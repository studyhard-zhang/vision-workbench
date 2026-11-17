"""Keep candidate authentication and verification separate from the active target."""
import base64
import json
import re
import shlex
import threading
import uuid
from pathlib import Path

from .ssh_credentials import protect
from .ssh_setup import connection_input
from .storage import ProjectError, timestamp
from tools.remote_environment import ROOTS, WEIGHTS, RECIPES

_files = threading.Lock()


def read(path, default=None):
    with _files:
        return json.loads(path.read_text('utf-8')) if path.is_file() else default


def write(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    with _files:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(payload, 'utf-8')
        temporary.replace(path)


def active(directory):
    return read(directory/'active-compute.json')


def run_check(transport, action='inspect', python=None, **arguments):
    if action not in ('inspect','validate'):raise ProjectError(409,'ENVIRONMENT_USER_MANAGED','请由用户自行准备环境，软件仅提供校验。')
    code = (Path(__file__).resolve().parents[1]/'tools/remote_environment.py').read_bytes()
    source = 'exec(__import__("base64").b64decode('+repr(base64.b64encode(code).decode())+'))'
    payload = dict(action=action, **arguments)
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    tail = shlex.join(['-c', source, encoded])
    if python:
        unavailable=json.dumps({'ok':True,'value':{'blockers':['指定的Python不可执行，请由用户安装Python 3.10–3.12或填写已有虚拟环境的正确绝对路径：'+python],'missing':[],'missing_weights':[],'gpu':None}})
        command = f'unset PYTHONPATH PYTHONHOME; if [ -x {shlex.quote(python)} ]; then PYTHONNOUSERSITE=1 {shlex.quote(python)} {tail}; else printf "%s\\n" {shlex.quote(unavailable)}; fi'
    else:
        no_python = json.dumps({'ok': True, 'value': {'blockers': ['实例缺少 Python 3，请选择带 Python 3.10–3.12 的镜像。'], 'missing': [], 'missing_weights': [], 'gpu': None}})
        command = ('unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1; '
                   'for p in /root/miniconda3/bin/python python3; do '
                   f'if command -v "$p" >/dev/null 2>&1; then exec "$p" {tail}; fi; done; '
                   f'printf "%s\\n" {shlex.quote(no_python)}')
    try:
        response = json.loads(transport.execute(command, 120 if action=='validate' else 40))
        if response['ok'] is not True:
            raise ProjectError(409, 'ENVIRONMENT_CHECK_FAILED', str(response['error'])[:1000])
        return response['value']
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectError(503, 'ENVIRONMENT_RESPONSE_INVALID', '无法读取实例环境报告，请重新检查连接。') from exc


class ComputeInstances:
    @staticmethod
    def inspect_transfer(transport,plan):
        import paramiko
        try:
            with transport.client() as client,client.open_sftp() as sftp:
                sftp.get_channel().settimeout(12);sftp.stat('.')
        except (OSError,paramiko.SSHException):
            plan['blockers'].append('SSH已连接，但SFTP不可用。请由用户在主机启用SSH服务的SFTP子系统、允许当前账号访问工作目录，然后重新检查。')
    def __init__(self, remote):
        self.remote = remote
        self.directory = remote.storage.path.parent
        self.candidate = self.directory/'ssh-candidate.json'
        self.saved = self.directory/'compute-instances'
        self.worker = None

    def status(self):
        candidate = read(self.candidate)
        current = active(self.directory)
        available = []
        for path in sorted(self.saved.glob('*.json')):
            record = read(path)
            available.append({'id': record['target']['id'], 'name': record['target']['name'], 'endpoint': record['endpoint']})
        if self.remote.config.is_file():
            available.insert(0, {'id': 'legacy', 'name': '原训练目标'})
        result = {'active': current['target']['id'] if current else 'legacy', 'available': available, 'candidate': None}
        if candidate:
            public = {key: candidate[key] for key in ('id','endpoint','plan','state')}
            public['error'] = candidate.get('error','')
            if public['state'] in ('preparing','validating') and not (self.worker and self.worker.is_alive()):
                public.update(state='interrupted', error='环境校验已中断，请重新连接并校验。')
            result['candidate'] = public
        return result

    def connect(self, values):
        from .remote import PasswordSSHTransport
        if self.worker and self.worker.is_alive():
            raise ProjectError(409,'ENVIRONMENT_BUSY','正在配置实例，请等待操作完成。')
        python=values.get('python')
        root=values.get('root')
        profile, password = connection_input({k:v for k,v in values.items() if k not in ('python','root','connection_kind')})
        if values.get('connection_kind','ssh') not in ('ssh','local'):raise ProjectError(422,'INVALID_CONNECTION_KIND','请选择本机或SSH连接。')
        if values.get('connection_kind')=='local':
            if profile['host'] not in ('127.0.0.1','localhost'):raise ProjectError(422,'INVALID_LOCAL_HOST','本机连接请使用127.0.0.1或localhost。')
            profile['connection_kind']='local'
        if python and (not isinstance(python,str) or not python.startswith('/') or any(c in python for c in '\n\r\x00')):raise ProjectError(422,'INVALID_PYTHON','请输入Linux Python绝对路径。')
        known = [read(self.remote.config), read(self.candidate), *[read(path) for path in self.saved.glob('*.json')]]
        for entry in filter(None, known):
            other = entry.get('profile',entry)
            if all(other.get(k)==profile[k] for k in ('host','port','user')) and other.get('host_key_fingerprint') not in (None,profile['host_key_fingerprint']):
                raise ProjectError(409,'SSH_HOST_KEY_FAILED','该地址的主机公钥发生变化，请先核对实例。')
        for entry in filter(None, known):
            other = entry.get('profile',entry)
            if all(other.get(k)==profile[k] for k in ('host','port','user')) and other.get('local_wsl')=='Ubuntu-24.04':
                profile['local_wsl']='Ubuntu-24.04'
        transport = PasswordSSHTransport(profile, password=password)
        plan = run_check(transport,python=python,root=root) if python or root else run_check(transport)
        self.inspect_transfer(transport,plan)
        identifier = uuid.uuid4().hex
        secrets = self.directory/'.secrets'
        secrets.mkdir(exist_ok=True)
        credential = secrets/(identifier+'.dpapi')
        credential.write_bytes(protect(password))
        profile['credential_file'] = str(credential.resolve())
        write(self.candidate, dict(id=identifier, endpoint=transport.endpoint, profile=profile, plan=plan, state='checked'))
        return self.status()

    def prepare(self, identifier):
        """Compatibility entry point: validation only, without environment changes."""
        from .remote import PasswordSSHTransport
        candidate=read(self.candidate)
        if not candidate or identifier!=candidate['id']:raise ProjectError(409,'SSH_CANDIDATE_CHANGED','候选环境已变化，请重新连接。')
        if self.worker and self.worker.is_alive():return self.status()
        if candidate['state']=='ready':return self.status()
        plan=candidate['plan']
        if plan['blockers'] or plan['missing'] or plan['missing_weights']:raise ProjectError(409,'ENVIRONMENT_BLOCKED','环境尚未就绪，请按缺失项提示自行准备后重新连接。')
        candidate.update(state='validating',error='');write(self.candidate,candidate)
        def work():
            try:
                result=run_check(PasswordSSHTransport(candidate['profile']),'validate',python=plan['executable'],root=plan['root'])
                if result.get('ready') is False:raise ProjectError(409,'ENVIRONMENT_BLOCKED','环境发生变化，请重新连接并查看缺失项。')
                record=self.verified_record(candidate,result);write(self.saved/(record['target']['id']+'.json'),record);candidate['state']='ready'
            except Exception as exc:candidate.update(state='failed',error=str(exc)[:1600])
            write(self.candidate,candidate)
        self.worker=threading.Thread(target=work,name='compute-validation',daemon=True);self.worker.start()
        return self.status()

    def connect_local(self, values):
        if self.worker and self.worker.is_alive():raise ProjectError(409,'ENVIRONMENT_BUSY','正在校验环境，请等待完成。')
        records=[active(self.directory),*[read(path) for path in self.saved.glob('*.json')]]
        record=next((record for record in records if record and (record['profile'].get('local_wsl')=='Ubuntu-24.04' or record['profile'].get('connection_kind')=='local')),None)
        if not record:raise ProjectError(409,'LOCAL_ENVIRONMENT_REQUIRED','请自行安装WSL2 Ubuntu24.04、CUDA版训练库，并准备本机SSH训练服务和登录信息；可使用SSH方式连接127.0.0.1完成首次登记。')
        from .remote import PasswordSSHTransport
        profile=dict(record['profile']);python=values.get('python') or profile['python_executable']
        if not isinstance(python,str) or not python.startswith('/') or any(c in python for c in '\n\r\x00'):raise ProjectError(422,'INVALID_PYTHON','请选择WSL中的Python绝对路径。')
        transport=PasswordSSHTransport(profile)
        plan=run_check(transport,python=python,root=values.get('root') or profile['workspace_root'])
        self.inspect_transfer(transport,plan)
        identifier=uuid.uuid4().hex;write(self.candidate,dict(id=identifier,endpoint=record['endpoint'],profile=profile,plan=plan,state='checked'))
        return self.status()

    @staticmethod
    def verified_record(candidate, result):
        plan = candidate['plan']
        runtime = result['runtime']
        environment_only=result.get('validation_kind')=='environment'
        valid_models=set(result['tests'])<=set(WEIGHTS) and 'yolo26n.pt' in result['tests'] if environment_only else set(result['tests'])==set(WEIGHTS)
        if plan['root'] not in ROOTS or not valid_models:
            raise ProjectError(409,'ENVIRONMENT_INCOMPLETE','检测与分割验证结果不完整。')
        expected = RECIPES[plan['recipe']]
        prefix = plan['root']+'/environment-v1/'
        python = result['python']
        if (runtime['torch'],runtime['torchvision']) != expected or runtime['ultralytics'] != '8.4.203' or not python.startswith('/') or (result.get('validation_kind')!='environment' and not python.startswith(prefix)):
            raise ProjectError(409,'ENVIRONMENT_INCOMPLETE','验证运行库与配置方案不符。')
        for name,test in result['tests'].items():
            if result.get('validation_kind')=='environment':
                if not test.get('environment_checked') or not test.get('forward_passed') or test.get('training_executed') is not False or test.get('weights_sha256')!=WEIGHTS[name]:raise ProjectError(409,'ENVIRONMENT_INCOMPLETE','环境计算校验未通过。')
            elif test['epochs'] != 1 or test['reload'] != ['best.pt','last.pt']:
                raise ProjectError(409,'ENVIRONMENT_INCOMPLETE','验证训练或权重重载未完成。')
        weights = {name: {'id':name[:-3], 'name':name, 'path':prefix+'weights/'+name, 'sha256':WEIGHTS[name]} for name in result['tests']}
        identity = plan['gpu']['uuid']
        target = dict(id='ssh-'+candidate['id'], name=plan['gpu']['name']+' · '+candidate['endpoint'].split(':')[-1], kind='self_hosted',
                      transport='ssh', gpu_selector='0', expected_gpu=plan['gpu']['name'], verified_gpu_uuid=identity,
                      weights=weights['yolo26n.pt'], runtime=runtime, executor_connected=True, enabled=True,
                      verified_at=timestamp(), managed_environment=True, python_executable=python, workspace_root=plan['root'])
        if environment_only:target['verification_kind']='environment'
        profile = dict(candidate['profile'], verified_gpu_uuid=identity, python_executable=python, workspace_root=plan['root'])
        return dict(profile=profile, target=target, weights=weights, tests=result['tests'], endpoint=candidate['endpoint'])

    def activate(self, identifier):
        import paramiko
        from .remote import PasswordSSHTransport
        from .training import TrainingRepository
        if identifier == 'legacy':
            transport = self.remote.transport(legacy=True)
            target = TrainingRepository(self.remote.storage).target(legacy=True)
            probe = transport.call('probe')
            if not any(line.split(',')[0].strip()==target['verified_gpu_uuid'] for line in probe.get('nvidia_smi','').splitlines()):
                raise ProjectError(409,'SSH_GPU_CHANGED','原地址当前 GPU 与原目标不符，请重新检查。')
            record = None
        else:
            if not isinstance(identifier,str) or not re.fullmatch(r'ssh-[a-f0-9]{32}',identifier):
                raise ProjectError(422,'INVALID_COMPUTE_INSTANCE','请选择已验证实例。')
            record = read(self.saved/(identifier+'.json'))
            if not record:
                raise ProjectError(404,'COMPUTE_INSTANCE_NOT_FOUND','未找到已验证实例。')
            target = record['target']
            transport = PasswordSSHTransport(record['profile'])
            plan = run_check(transport, python=target['python_executable'])
            current = {**plan['packages'],'python':plan['python']}
            if plan['blockers'] or not plan['gpu'] or plan['gpu']['uuid'] != target['verified_gpu_uuid'] or plan['missing_weights'] or plan['missing'] or any(name in plan.get('optional_missing_weights',[]) for name in record['weights']) or any(current.get(k)!=v for k,v in target['runtime'].items()):
                raise ProjectError(409,'ENVIRONMENT_CHANGED','GPU、依赖或权重已变化，请重新配置并验证该实例。')
            try:
                with transport.client() as client, client.open_sftp() as sftp:
                    sftp.get_channel().settimeout(12)
                    sftp.stat(target['workspace_root'])
            except (OSError,paramiko.SSHException) as exc:
                raise ProjectError(409,'SFTP_UNAVAILABLE','SSH 文件传输或工作目录不可用，请检查实例的 SFTP 服务。') from exc
        with self.remote.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if any(db.execute(f"SELECT 1 FROM {kind}_dispatches WHERE state='AWAITING_RESULT'").fetchone() for kind in ('training','evaluation','inference')):
                raise ProjectError(409,'SSH_CONNECTION_BUSY','请先完成原目标上未收尾的任务，再切换实例。')
            write(self.directory/'active-compute.json',record)
        return self.status()
