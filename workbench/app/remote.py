"""Pinned-key OpenSSH transport and durable remote-run reconciliation."""
import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

from .storage import ProjectError,timestamp
from .training import TARGET
from .executions import MAX_RESULT_BYTES


class SSHTransport:
    def __init__(self,profile):
        self.profile=profile
        if not re.fullmatch('[a-zA-Z0-9.-]+',profile.get('host','')) or profile.get('user')!='root' or type(profile.get('port')) is not int or not 1<=profile['port']<=65535:raise ProjectError(503,'SSH_CONFIG_INVALID','SSH目标配置无效。')
        self.endpoint=f"{profile['user']}@{profile['host']}:{profile['port']}"

    def options(self):
        p=self.profile
        return ['-i',p['identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
                '-o','UserKnownHostsFile='+p['known_hosts_file'],'-o','KexAlgorithms=curve25519-sha256',
                '-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','LogLevel=ERROR']

    def run(self,args,timeout=45):
        try:result=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except (OSError,subprocess.TimeoutExpired) as exc:raise ProjectError(503,'SSH_UNAVAILABLE','SSH连接或传输未完成，请检查实例和网络；远程训练可能仍在运行。') from exc
        if result.returncode:
            detail=result.stderr.decode('utf-8',errors='replace')[-1200:]
            if not detail:
                try:
                    response=json.loads(result.stdout)
                    if response.get('ok') is False:detail=str(response.get('error',''))[:1200]
                except (ValueError,TypeError,AttributeError):pass
            code='SSH_HOST_KEY_FAILED' if any(t in detail.lower() for t in ('host key verification failed','remote host identification has changed')) else 'SSH_AUTHENTICATION_FAILED' if 'permission denied' in detail.lower() else 'SSH_FAILED'
            raise ProjectError(503,code,'SSH操作失败：'+detail)
        return result.stdout

    def call(self,action,**values):
        code=(Path(__file__).resolve().parents[1]/'tools/ssh_control.py').read_bytes()
        source='exec(__import__("base64").b64decode('+repr(base64.b64encode(code).decode())+'))'
        payload=base64.b64encode(json.dumps({'action':action,'expected_gpu_uuid':self.profile.get('verified_gpu_uuid',TARGET['verified_gpu_uuid']),'workspace_root':self.profile.get('workspace_root','/root/autodl-tmp/vision-workbench'),**values}).encode()).decode()
        command=shlex.join([self.profile.get('python_executable','/root/miniconda3/bin/python'),'-c',source,payload])
        if self.profile.get('python_executable'):
            command='unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1; '+command
        raw=self.execute(command,70 if action=='stop' else 40)
        try:value=json.loads(raw.decode('utf-8'))
        except (ValueError,UnicodeError) as exc:raise ProjectError(503,'SSH_RESPONSE_INVALID','实例控制响应无法识别，请保留运行目录。') from exc
        if value.get('ok') is not True:raise ProjectError(503,'REMOTE_CONTROL_FAILED',str(value.get('error','远程操作未完成。'))[:1600])
        return value['value']

    def execute(self,command,timeout=40):
        from .local_wsl import manager
        manager.ensure(self.profile)
        return self.run([shutil.which('ssh') or 'ssh','-T','-p',str(self.profile['port']),*self.options(),self.profile['user']+'@'+self.profile['host'],command],timeout)

    def copy(self,local,job_id,upload):
        import paramiko
        if not re.fullmatch('[0-9a-f]{32}',job_id):raise ProjectError(422,'INVALID_JOB','训练标识无效。')
        root=self.profile.get('workspace_root','/root/autodl-tmp/vision-workbench')
        remote=f'{root}/{job_id}/'+('execution.zip.upload' if upload else 'return-result.zip')
        try:
            if not upload:
                from .ssh_transfer import download
                download(self.client,local,remote,self.endpoint);return
            client=self.client()
            try:
                with client.open_sftp() as sftp:
                    sftp.get_channel().settimeout(120);sftp.put(str(local),remote)
            finally:client.close()
        except (OSError,paramiko.SSHException) as exc:raise ProjectError(503,'SSH_TRANSFER_FAILED','SSH文件传输未完成，原任务及材料保留，可恢复同一任务。') from exc

    def client(self):
        from .local_wsl import manager
        manager.ensure(self.profile)
        import paramiko
        from .ssh_setup import fingerprint
        profile=self.profile;expected=profile.get('host_key_fingerprint');client=paramiko.SSHClient()
        class Pinned(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self,client,hostname,key):
                if fingerprint(key)!=expected:raise paramiko.BadHostKeyException(hostname,key,key)
        try:
            if expected:client.set_missing_host_key_policy(Pinned())
            else:client.load_host_keys(profile['known_hosts_file'])
            client.connect(profile['host'],port=profile['port'],username=profile['user'],key_filename=profile['identity_file'],allow_agent=False,look_for_keys=False,timeout=8,banner_timeout=10,auth_timeout=10)
            return client
        except paramiko.BadHostKeyException as exc:
            client.close();raise ProjectError(503,'SSH_HOST_KEY_FAILED','SSH服务器公钥与已确认指纹不符，未执行远程操作。') from exc
        except paramiko.AuthenticationException as exc:
            client.close();raise ProjectError(503,'SSH_AUTHENTICATION_FAILED','SSH免密认证失败，请核对原密钥与连接。') from exc
        except (OSError,paramiko.SSHException) as exc:
            client.close();raise ProjectError(503,'SSH_UNAVAILABLE','SSH文件连接未完成，原任务保持，可恢复同一任务。') from exc


class PasswordSSHTransport(SSHTransport):
    def __init__(self,profile,password=None):
        super().__init__(profile);self._password=password

    def client(self):
        from .local_wsl import manager
        manager.ensure(self.profile)
        import paramiko
        from .ssh_setup import fingerprint
        from .ssh_credentials import unprotect
        profile=self.profile;expected=profile.get('host_key_fingerprint')
        class Pinned(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self,client,hostname,key):
                if fingerprint(key)!=expected:raise paramiko.BadHostKeyException(hostname,key,key)
        client=paramiko.SSHClient();client.set_missing_host_key_policy(Pinned())
        try:
            password=self._password if self._password is not None else unprotect(Path(profile['credential_file']))
            client.connect(profile['host'],port=profile['port'],username=profile['user'],password=password,allow_agent=False,look_for_keys=False,timeout=8,banner_timeout=10,auth_timeout=10)
            return client
        except paramiko.BadHostKeyException as exc:
            client.close();raise ProjectError(503,'SSH_HOST_KEY_FAILED','SSH服务器公钥与已确认指纹不符，未执行远程操作。') from exc
        except paramiko.AuthenticationException as exc:
            client.close();raise ProjectError(503,'SSH_AUTHENTICATION_FAILED','SSH登录失败，请核对密码和登录指令。') from exc
        except ProjectError:client.close();raise
        except (OSError,paramiko.SSHException) as exc:
            client.close()
            from .local_wsl import is_local
            if is_local(profile):raise ProjectError(503,'LOCAL_EXECUTOR_UNAVAILABLE','本机训练连接未完成，请检查 WSL 训练服务；原任务保持，不会自动重跑。') from exc
            raise ProjectError(503,'SSH_UNAVAILABLE','SSH连接未完成，请核对实例和网络；原任务不会自动重跑。') from exc

    def execute(self,command,timeout=40):
        import time
        import paramiko
        client=self.client();channel=None
        try:
            channel=client.get_transport().open_session(timeout=10);channel.exec_command(command);channel.shutdown_write()
            chunks=[];errors=[];size=0;deadline=time.monotonic()+timeout
            while True:
                for ready,receive,output in ((channel.recv_ready,channel.recv,chunks),(channel.recv_stderr_ready,channel.recv_stderr,errors)):
                    while ready():
                        chunk=receive(65536);size+=len(chunk)
                        if size>4*1024**2:raise ProjectError(503,'SSH_RESPONSE_INVALID','SSH控制响应超出上限，保留原任务。')
                        output.append(chunk)
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():break
                if time.monotonic()>deadline:raise ProjectError(503,'SSH_UNAVAILABLE','SSH控制响应超时，保留原任务。')
                time.sleep(.02)
            if channel.recv_exit_status()!=0:
                detail=b''.join(errors).decode('utf-8',errors='replace')[-1200:]
                try:
                    response=json.loads(b''.join(chunks))
                    if response.get('ok') is False:detail=str(response.get('error',''))[-1200:]
                except (ValueError,TypeError,AttributeError):
                    if not detail:detail=b''.join(chunks).decode('utf-8',errors='replace')[-1200:]
                raise ProjectError(503,'SSH_FAILED','远端控制未完成：'+detail)
            return b''.join(chunks)
        except (OSError,paramiko.SSHException) as exc:raise ProjectError(503,'SSH_UNAVAILABLE','SSH操作中断，请恢复原连接核对结果。') from exc
        finally:
            if channel:channel.close()
            client.close()

    def copy(self,local,job_id,upload):
        return super().copy(local,job_id,upload)


class RemoteRepository:
    def __init__(self,storage,training,executions,kind="training"):
        assert kind in ("training","evaluation","inference")
        self.kind=kind
        self.storage=storage;self.training=training;self.executions=executions
        self.config=storage.path.parent/'ssh-target.json';self.stop_event=threading.Event();self.thread=None;self.guard=threading.Lock()
        self.pending=storage.path.parent/'ssh-target-change.json'

    def execute(self,db,sql,parameters=()):
        if self.kind!='training':sql=sql.replace('training_',self.kind+'_')
        return db.execute(sql,parameters)

    def transport(self, legacy=False, job=None):
        from .compute_instances import active
        record=active(self.storage.path.parent) if not legacy else None
        if record:
            profile=record['profile']
        elif self.config.is_file():
            profile=json.loads(self.config.read_text(encoding='utf-8'))
        else:
            raise ProjectError(503,'SSH_NOT_CONFIGURED','请先配置 SSH 连接。')
        if job and job.get('snapshot',{}).get('backend')=='tensorrt':
            acceleration=profile.get('acceleration')
            if not acceleration:raise ProjectError(409,'TENSORRT_NOT_READY','当前设备未准备TensorRT运行环境。')
            profile={**profile,'python_executable':acceleration['python_executable']}
        return (PasswordSSHTransport if profile.get('authentication')=='password' else SSHTransport)(profile)

    def info(self):
        try:transport=self.transport()
        except ProjectError:return {'configured':False,'message':'SSH待配置；手动执行包仍可用。'}
        last=self.storage.path.parent/'ssh-last-check.json'
        pending=json.loads(self.pending.read_text(encoding='utf-8')) if self.pending.is_file() else None
        with self.storage.connect() as db:
            reservations=[dict(r) for r in self.execute(db,"SELECT d.job_id,j.name,j.state,s.endpoint,s.phase,s.cancel_requested FROM training_dispatches d JOIN training_jobs j ON j.id=d.job_id LEFT JOIN training_remote_sessions s ON s.job_id=j.id WHERE d.state='AWAITING_RESULT'")]
        profile=getattr(transport,'profile',{})
        command=f"ssh -p {profile['port']} {profile['user']}@{profile['host']}" if profile else ''
        return {'configured':True,'endpoint':transport.endpoint,'command':command,'authentication':profile.get('authentication','key'),'last_check':json.loads(last.read_text(encoding='utf-8')) if last.is_file() else None,
            'execution_enabled':not pending,'pending_target':pending,'reservations':reservations}

    def check(self):
        if self.pending.is_file():raise ProjectError(409,'SSH_TARGET_CHANGE_PENDING','新计算目标尚未完成核对，旧配置保留；请先完成新实例的公钥、身份和环境确认。')
        transport=self.transport();value=transport.call('probe')
        target=self.training.target()
        if not any(line.split(',')[0].strip()==target['verified_gpu_uuid'] for line in value.get('nvidia_smi','').splitlines()):raise ProjectError(409,'SSH_GPU_CHANGED','已连接，但实例GPU身份与当前目标不同，请先重新核对。')
        if target.get('managed_environment') and value.get('python')!=target['runtime']['python']:
            raise ProjectError(409,'SSH_RUNTIME_CHANGED','实例的工作台 Python 环境已变化，请重新检查并验证。')
        value.update(checked_at=timestamp(),endpoint=transport.endpoint,training_executed=False)
        (self.storage.path.parent/'ssh-last-check.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        return value

    def submit(self,project_id,task_id,job_id):
        transport=self.transport();job=self.training.get(project_id,task_id,job_id)
        if self.pending.is_file() and not job['remote']:raise ProjectError(409,'SSH_TARGET_CHANGE_PENDING','计算目标切换待核对，未向旧目标发出训练。')
        target=self.training.target()
        if job['snapshot']['target']['id']!=target['id'] or job['snapshot']['target'].get('verified_gpu_uuid')!=target['verified_gpu_uuid']:
            raise ProjectError(409,'SSH_RUN_TARGET_MISMATCH','此训练属于其他GPU目标，不能将旧任务改绑到新实例。请按配置新建。')
        if job['state'] not in ('QUEUED','DISPATCHED','RUNNING'):raise ProjectError(409,'REMOTE_TERMINAL','该训练已经结束，请按配置新建。')
        if job['remote'] and job['remote']['phase']=='COMPLETED':raise ProjectError(409,'REMOTE_TERMINAL','此会话已经完成。')
        with self.storage.connect() as db:existing=self.execute(db,'SELECT endpoint FROM training_remote_sessions WHERE job_id=?',(job_id,)).fetchone()
        if existing and existing['endpoint']!=transport.endpoint:raise ProjectError(409,'SSH_ENDPOINT_CHANGED','此任务绑定其他SSH地址，不会连接新实例恢复旧任务。')
        self.check()  # CPU-only, before reserving a new job.
        self.executions.dispatch(project_id,task_id,job_id)
        with self.storage.connect() as db:
            self.execute(db,'BEGIN IMMEDIATE');row=self.execute(db,'SELECT * FROM training_remote_sessions WHERE job_id=?',(job_id,)).fetchone()
            if row and row['endpoint']!=transport.endpoint:raise ProjectError(409,'SSH_ENDPOINT_CHANGED','此任务绑定其他SSH地址，未自动迁移或重跑。')
            if row:self.execute(db,"UPDATE training_remote_sessions SET phase='RECONNECTING',error='',updated_at=? WHERE job_id=?",(timestamp(),job_id))
            else:self.execute(db,"INSERT INTO training_remote_sessions(job_id,endpoint,phase,created_at,updated_at) VALUES(?,?,'UPLOADING',?,?)",(job_id,transport.endpoint,timestamp(),timestamp()))
        self.start_worker();return self.training.get(project_id,task_id,job_id)

    def pause(self,project_id,task_id,job_id):
        job=self.training.get(project_id,task_id,job_id)
        if job['state']!='RUNNING' or not job['remote']:raise ProjectError(409,'REMOTE_NOT_ACTIVE','仅已确认启动的实例运行可暂停监控。')
        with self.storage.connect() as db:
            self.execute(db,'BEGIN IMMEDIATE')
            current=self.training.row(db,project_id,task_id,job_id)
            if current['state']!='RUNNING':raise ProjectError(409,'REMOTE_NOT_ACTIVE','运行状态已变化，请刷新。')
            row=self.execute(db,'SELECT cancel_requested FROM training_remote_sessions WHERE job_id=?',(job_id,)).fetchone()
            if row['cancel_requested']:raise ProjectError(409,'REMOTE_CANCELLING','正在中止，需继续核对进程与资源，不能暂停监控。')
            self.execute(db,"UPDATE training_remote_sessions SET phase='PAUSED',updated_at=? WHERE job_id=?",(timestamp(),job_id))
        return self.training.get(project_id,task_id,job_id)

    def cancel(self,project_id,task_id,job_id):
        job=self.training.get(project_id,task_id,job_id)
        if job['state'] not in ('DISPATCHED','RUNNING'):return job
        with self.storage.connect() as db:
            if not self.execute(db,'SELECT 1 FROM training_remote_sessions WHERE job_id=?',(job_id,)).fetchone():raise ProjectError(409,'REMOTE_SESSION_REQUIRED','此记录尚未通过SSH连接，可先连接后确认中止。')
            self.execute(db,"UPDATE training_remote_sessions SET cancel_requested=1,phase='CANCELLING',error='',updated_at=? WHERE job_id=? AND phase!='COMPLETED'",(timestamp(),job_id))
        self.start_worker();return self.training.get(project_id,task_id,job_id)

    def abandon(self,project_id,task_id,job_id,values):
        if self.kind!='training':raise ProjectError(422,'TRAINING_REQUIRED','此操作仅适用于训练。')
        if set(values)!={'confirmation_name','visibility_revision','instance_unavailable'} or values['instance_unavailable'] is not True:
            raise ProjectError(422,'INSTANCE_CONFIRMATION_REQUIRED','请确认设备无法使用，再结束并删除本地训练记录。')
        with self.guard, self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=self.training.row(db,project_id,task_id,job_id);job=self.training.public(db,row)
            if job['remote'] and job['remote']['progress'].get('local_abandonment'):
                return {'job':job,'reused':True}
            if values['confirmation_name']!=job['name'] or type(values['visibility_revision']) is not int or values['visibility_revision']!=job['visibility_revision']:
                raise ProjectError(409,'TRAINING_VISIBILITY_CHANGED','训练名称或状态已变化，请刷新后重试。')
            if not job['can_abandon']:raise ProjectError(409,'TRAINING_NOT_ABANDONABLE','请先暂停监控，或等待连接失败后，再确认设备不可用。')
            now=timestamp()
            audit={'origin':'controller','reason':'instance_unavailable','confirmed_at':now,'remote_stop_confirmed':False,'job_id':job_id,'target_id':job['snapshot']['target']['id'],'gpu_uuid':job['snapshot']['target']['verified_gpu_uuid']}
            progress={**job['remote']['progress'],'local_abandonment':audit}
            report={'kind':'LOCAL_ABANDONMENT','report':{'status':'abandoned','completion_reason':'instance_unavailable','remote_stop_confirmed':False},'artifacts':{},'audit':audit}
            raw=json.dumps(report,ensure_ascii=False,sort_keys=True).encode('utf-8')
            db.execute("UPDATE training_dispatches SET state='FAILED',result_json=?,result_sha256=?,updated_at=? WHERE job_id=? AND state='AWAITING_RESULT'",(raw.decode('utf-8'),hashlib.sha256(raw).hexdigest(),now,job_id))
            db.execute("UPDATE training_jobs SET state='FAILED',updated_at=? WHERE id=?",(now,job_id))
            db.execute("UPDATE training_remote_sessions SET phase='COMPLETED',progress_json=?,error=?,updated_at=? WHERE job_id=?",(json.dumps(progress,ensure_ascii=False),'设备不可用，本地已结束跟踪；远端停止状态未核实。',now,job_id))
            db.execute('INSERT INTO training_record_visibility(job_id,deleted_at,revision,updated_at) VALUES(?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET deleted_at=excluded.deleted_at,revision=excluded.revision,updated_at=excluded.updated_at',(job_id,now,job['visibility_revision']+1,now))
        return {'job':self.training.get(project_id,task_id,job_id),'reused':False}

    @staticmethod
    def release_confirmed(state):
        resources=state.get('resources',{})
        return state.get('alive') is False and resources.get('checked') is True and resources.get('free') is True

    def finish_stopped(self,project_id,task_id,job_id,state,confirmation=None,reason=''):
        if self.kind!='training' or not self.release_confirmed(state):
            raise ProjectError(409,'REMOTE_STOP_NOT_CONFIRMED','尚未确认训练进程退出及设备锁释放，请稍后重新核实。')
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job=self.training.public(db,self.training.row(db,project_id,task_id,job_id))
            if confirmation is not None and (set(confirmation)!={'confirmation_name','visibility_revision'} or confirmation['confirmation_name']!=job['name'] or type(confirmation['visibility_revision']) is not int or confirmation['visibility_revision']!=job['visibility_revision']):
                raise ProjectError(409,'TRAINING_VISIBILITY_CHANGED','训练记录已变化，请刷新后重试。')
            if not job['remote'] or not job['remote']['cancel_requested']:raise ProjectError(409,'REMOTE_NOT_STOPPING','请先中止该任务。')
            if job['dispatch']['state']=='AWAITING_RESULT':
                now=timestamp();audit={'confirmed_at':now,'remote_stop_confirmed':True,'resources':state['resources'],'reason':reason[:1600]}
                progress={**job['remote']['progress'],**state.get('progress',{}),'local_interruption':audit,'resources':state['resources'],'remote_process_exited':True}
                receipt={'kind':'LOCAL_INTERRUPTION','report':{'status':'interrupted','remote_stop_confirmed':True},'artifacts':{},'audit':audit}
                payload=json.dumps(receipt,ensure_ascii=False,sort_keys=True).encode('utf8')
                db.execute("UPDATE training_jobs SET state='INTERRUPTED',updated_at=? WHERE id=?",(now,job_id))
                db.execute("UPDATE training_dispatches SET state='INTERRUPTED',result_json=?,result_sha256=?,updated_at=? WHERE job_id=?",(payload.decode('utf8'),hashlib.sha256(payload).hexdigest(),now,job_id))
                db.execute("UPDATE training_remote_sessions SET phase='COMPLETED',progress_json=?,error='',updated_at=? WHERE job_id=?",(json.dumps(progress,ensure_ascii=False),now,job_id))
            if confirmation is not None:
                current=self.training.public(db,self.training.row(db,project_id,task_id,job_id))
                if not current['deleted_at']:
                    if not current['can_delete']:raise ProjectError(409,'TRAINING_ACTIVE','任务仍未收尾，请刷新后核实。')
                    now=timestamp()
                    db.execute('INSERT INTO training_record_visibility VALUES(?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET deleted_at=excluded.deleted_at,revision=excluded.revision,updated_at=excluded.updated_at',(job_id,now,current['visibility_revision']+1,now))
        return {'job':self.training.get(project_id,task_id,job_id)}

    def stop_and_trash(self,project_id,task_id,job_id,values):
        with self.guard:
            job=self.training.get(project_id,task_id,job_id);transport=self.transport()
            with self.storage.connect() as db:session=db.execute('SELECT endpoint FROM training_remote_sessions WHERE job_id=?',(job_id,)).fetchone()
            if not session or session['endpoint']!=transport.endpoint:raise ProjectError(409,'SSH_ENDPOINT_CHANGED','请核对原任务绑定的设备。')
            state=transport.call('verify-release',job_id=job_id,expected_gpu_uuid=job['snapshot']['target']['verified_gpu_uuid'])
            return self.finish_stopped(project_id,task_id,job_id,state,values,'用户核实中止后删除，原始日志与权重保留在执行目录。')

    def recover_device(self):
        if self.kind!='training':raise ProjectError(422,'TRAINING_REQUIRED','请从训练设备入口核对。')
        with self.guard:
            self.check();target=self.training.target();transport=self.transport()
            with self.storage.connect() as db:
                rows=[dict(row) for row in db.execute("SELECT s.*,j.snapshot_json FROM training_remote_sessions s JOIN training_jobs j ON j.id=s.job_id WHERE json_extract(s.progress_json,'$.local_abandonment.remote_stop_confirmed')=0 AND coalesce(json_extract(s.progress_json,'$.local_abandonment_resolution.resources_released'),0)!=1 AND json_extract(j.snapshot_json,'$.target.id')=? AND json_extract(j.snapshot_json,'$.target.verified_gpu_uuid')=?",(target['id'],target['verified_gpu_uuid']))]
            proofs={}
            for row in rows:
                if row['endpoint']!=transport.endpoint:raise ProjectError(409,'SSH_ENDPOINT_CHANGED','原任务绑定的连接地址不同，请先核对设备。')
                state=transport.call('verify-release',job_id=row['job_id'],expected_gpu_uuid=target['verified_gpu_uuid'])
                if not self.release_confirmed(state):raise ProjectError(409,'REMOTE_STOP_NOT_CONFIRMED','原任务进程或设备锁尚未释放，请先完成中止。')
                proofs[row['job_id']]=state['resources']
            with self.storage.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                for row in rows:
                    current=db.execute('SELECT progress_json FROM training_remote_sessions WHERE job_id=?',(row['job_id'],)).fetchone()
                    progress=json.loads(current[0]);progress['local_abandonment_resolution']={'checked_at':timestamp(),'resources_released':True,'gpu_uuid':target['verified_gpu_uuid'],'target_id':target['id'],'resources':proofs[row['job_id']]}
                    db.execute("UPDATE training_remote_sessions SET progress_json=?,error='',updated_at=? WHERE job_id=?",(json.dumps(progress,ensure_ascii=False),timestamp(),row['job_id']))
            return {'recovered':len(rows),'target_id':target['id'],'gpu_uuid':target['verified_gpu_uuid']}

    def update(self,job_id,phase,progress=None,error=''):
        with self.storage.connect() as db:
            current=self.execute(db,'SELECT phase FROM training_remote_sessions WHERE job_id=?',(job_id,)).fetchone()
            if current and current['phase']=='COMPLETED':return
            if current and current['phase']=='PAUSED' and phase!='COMPLETED':phase='PAUSED'
            if progress is None:
                row=self.execute(db,'SELECT progress_json FROM training_remote_sessions WHERE job_id=?',(job_id,)).fetchone()
                progress=json.loads(row['progress_json']) if row else {}
            self.execute(db,'UPDATE training_remote_sessions SET phase=?,progress_json=?,error=?,updated_at=? WHERE job_id=?',
                       (phase,json.dumps(progress or {},ensure_ascii=False),error[:3000],timestamp(),job_id))

    def reconcile(self,row):
        with self.guard:
            return self._reconcile(row)

    def _reconcile(self,row):
        with self.storage.connect() as db:current=self.execute(db,'SELECT phase FROM training_remote_sessions WHERE job_id=?',(row['job_id'],)).fetchone()
        if current and current['phase'] in ('PAUSED','COMPLETED'):return
        if self.pending.is_file() and not row['launched'] and not row['cancel_requested']:
            raise ProjectError(409,'SSH_TARGET_CHANGE_PENDING','计算目标切换待核对，原上传会话保留，未启动旧实例训练。')
        transport=self.transport()
        if transport.endpoint!=row['endpoint']:raise ProjectError(409,'SSH_ENDPOINT_CHANGED','SSH地址已变化，未自动重跑原任务。')
        job_id=row['job_id'];pid=row['project_id'];tid=row['task_id']
        job=self.training.get(pid,tid,job_id)
        if job['snapshot'].get('backend')=='tensorrt':transport=self.transport(job=job)
        identity={'expected_gpu_uuid':job['snapshot']['target']['verified_gpu_uuid']}
        state=transport.call('status',job_id=job_id,**identity)
        if row['cancel_requested'] and not state['result_ready']:
            if not state['directory_exists']:
                # A failed transfer has not started the runner; prepare the same package before CPU-only closure.
                transport.call('prepare',job_id=job_id,**identity)
            if not state.get('launch'):
                self.update(job_id,'UPLOADING')
                path,_,_=self.executions.package(pid,tid,job_id);transport.copy(path,job_id,True)
                # Install without starting a training process.
                transport.call('install',job_id=job_id,package_sha256=row['package_sha256'],**identity)
            try:state=transport.call('stop',job_id=job_id,**identity)
            except ProjectError as exc:
                if self.kind!='training':raise
                proof=transport.call('verify-release',job_id=job_id,**identity)
                if not self.release_confirmed(proof):raise exc
                self.finish_stopped(pid,tid,job_id,proof,reason=str(exc));return
        elif not state.get('launch') and not state['result_ready']:
            if row['launched'] or state['started']:raise ProjectError(409,'REMOTE_UNCERTAIN','原运行目录或启动信息缺失，不自动重跑；请确认实例状态后中止并取回结果。')
            self.update(job_id,'UPLOADING');transport.call('prepare',job_id=job_id,**identity)
            path,_,_=self.executions.package(pid,tid,job_id);transport.copy(path,job_id,True)
            with self.storage.connect() as db:cancel=self.execute(db,'SELECT cancel_requested FROM training_remote_sessions WHERE job_id=?',(job_id,)).fetchone()[0]
            if cancel:
                transport.call('install',job_id=job_id,package_sha256=row['package_sha256'],**identity);state=transport.call('stop',job_id=job_id,**identity)
            else:
                self.update(job_id,'STARTING')
                with self.storage.connect() as db:self.execute(db,'UPDATE training_remote_sessions SET launched=1 WHERE job_id=?',(job_id,))
                state=transport.call('launch',job_id=job_id,package_sha256=row['package_sha256'],**identity)
        if state['result_ready']:
            resources=state.get('resources',{})
            if state.get('alive') is not False or resources.get('checked') is not True or resources.get('free') is not True:
                self.update(job_id,'RELEASING',{**state.get('progress',{}),'log_tail':state.get('log_tail',''),'resources':resources},error='结果已生成，尚未确认受管进程和设备锁全部释放；保留目标预留，未放行后续任务。')
                return
            self.update(job_id,'DOWNLOADING',state.get('progress'))
            destination=self.executions.owned(job_id,'ssh-return.zip')
            transport.copy(destination,job_id,False)
            if not destination.is_file() or destination.stat().st_size>MAX_RESULT_BYTES or hashlib.sha256(destination.read_bytes()).hexdigest()!=state['result_sha256']:raise ProjectError(409,'SSH_RESULT_DAMAGED','结果传输校验不符，未登记成功；可恢复连接重新获取。')
            self.executions.receive(pid,tid,job_id,destination.read_bytes(),remote_resources_verified=True)
            self.update(job_id,'COMPLETED',{**state.get('progress',{}),'log_tail':state.get('log_tail',''),'resources':resources,
                'remote_process_exited':True,'resource_checked_at':timestamp(),'remote_pid':state.get('launch',{}).get('pid')});return
        progress={**state.get('progress',{}),'log_tail':state.get('log_tail',''),'remote_pid':state.get('launch',{}).get('pid')}
        if state['alive']:
            with self.storage.connect() as db:self.execute(db,"UPDATE training_jobs SET state='RUNNING',updated_at=? WHERE id=? AND state='DISPATCHED'",(timestamp(),job_id))
            self.update(job_id,'RUNNING',progress)
        else:
            diagnostic=progress.get('diagnostic',{})
            storage=state.get('storage',{});free=storage.get('free_bytes')
            low_space=storage.get('checked') is True and type(free) is int and 0<=free<64*1024**2
            progress.update(error_code=diagnostic.get('code') or ('REMOTE_STORAGE_LOW' if low_space else 'REMOTE_RESULT_NOT_FINALIZED'),recovery_hint=diagnostic.get('next_step') or '保留目标占用；先核对原进程、实例空间与权限，再恢复原任务收尾，不自动重跑。')
            message='实例剩余空间低于64MiB，结果未收尾；请先释放空间，再恢复原任务的结果收尾。' if low_space else '实例进程尚未确认运行或已结束但结果未收尾；不会自动重跑。可继续检查，或确认后中止并回收。'
            self.update(job_id,'WAITING_RESULT',progress,error=message)

    def work(self):
        while not self.stop_event.is_set():
            with self.storage.connect() as db:
                rows=[dict(r) for r in self.execute(db,"SELECT s.*,t.project_id,j.task_id,d.package_sha256 FROM training_remote_sessions s JOIN training_jobs j ON j.id=s.job_id JOIN vision_tasks t ON t.id=j.task_id JOIN training_dispatches d ON d.job_id=j.id WHERE j.state IN ('DISPATCHED','RUNNING') AND s.phase NOT IN ('ERROR','PAUSED')")]
            if not rows:
                if self.stop_event.wait(3):return
                continue
            for row in rows:
                if self.stop_event.is_set():return
                try:self.reconcile(row)
                except Exception as exc:
                    with self.storage.connect() as db:current=self.execute(db,'SELECT progress_json FROM training_remote_sessions WHERE job_id=?',(row['job_id'],)).fetchone()
                    last=json.loads(current['progress_json']) if current else {}
                    self.update(row['job_id'],'ERROR',{**last,'error_code':getattr(exc,'code','REMOTE_RECONCILIATION_FAILED'),'recovery_hint':'连接错误不等于训练失败。目标预留保留；恢复连接只核对原进程与结果，不会重新训练。'},error=str(exc))
            if self.stop_event.wait(3):return

    def start_worker(self):
        with self.guard:
            if self.thread and self.thread.is_alive():return
            self.thread=threading.Thread(target=self.work,name='vision-ssh-worker',daemon=True);self.thread.start()

    def close(self):self.stop_event.set()
