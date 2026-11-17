"""Explicit SSH endpoint setup with pinned server identity and encrypted password."""
import base64,hashlib,json,re,shlex,socket,uuid
from pathlib import Path
from .storage import ProjectError,timestamp
from .ssh_credentials import protect

def parse_command(command):
    if not isinstance(command,str) or len(command)>400 or any(c in command for c in '\r\n\x00'):
        raise ProjectError(422,'INVALID_SSH_COMMAND','请输入ssh -p 端口 root@主机形式的登录指令。')
    try:parts=shlex.split(command)
    except ValueError:parts=[]
    match=re.fullmatch(r'root@([A-Za-z0-9][A-Za-z0-9.-]{0,252})',parts[-1]) if parts else None
    if len(parts)==2 and parts[0]=='ssh':port=22
    elif len(parts)==4 and parts[:2]==['ssh','-p'] and parts[2].isdigit():port=int(parts[2])
    else:port=0
    if not match or not 1<=port<=65535:
        raise ProjectError(422,'INVALID_SSH_COMMAND','请输入ssh -p 端口 root@主机；当前执行环境使用root账号，不支持附加命令或SSH选项。')
    return {'host':match[1],'user':'root','port':port}

def fingerprint(key):return 'SHA256:'+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')

def host_key(command,current=None):
    import paramiko
    profile=parse_command(command);transport=None
    try:
        sock=socket.create_connection((profile['host'],profile['port']),timeout=8)
        transport=paramiko.Transport(sock);transport.start_client(timeout=10);key=transport.get_remote_server_key()
        value={**profile,'fingerprint':fingerprint(key),'key_type':key.get_name(),'authenticated':False}
        same=current and all(profile[k]==current[k] for k in ('host','port','user'))
        if same and current.get('host_key_fingerprint') and current['host_key_fingerprint']!=value['fingerprint']:
            raise ProjectError(409,'SSH_HOST_KEY_FAILED','服务器主机公钥与已有信任记录不同，原连接保持；请核对实例后处理。')
        return value
    except ProjectError:raise
    except (OSError,paramiko.SSHException) as exc:raise ProjectError(503,'SSH_UNAVAILABLE','无法读取SSH主机公钥，请核对登录指令和实例网络。') from exc
    finally:
        if transport:transport.close()

def connection_input(values):
    if set(values)-{'command','password','fingerprint','confirmed_host'}:raise ProjectError(422,'UNKNOWN_FIELDS','SSH设置含不支持的字段。')
    profile=parse_command(values.get('command'));password=values.get('password')
    if not isinstance(password,str) or not 1<=len(password)<=1024 or any(c in password for c in '\x00\r\n'):
        raise ProjectError(422,'INVALID_SSH_PASSWORD','请输入SSH登录密码。')
    if values.get('confirmed_host') is not True or not isinstance(values.get('fingerprint'),str) or not re.fullmatch(r'SHA256:[A-Za-z0-9+/]{43}',values['fingerprint']):
        raise ProjectError(422,'SSH_HOST_CONFIRMATION_REQUIRED','请先读取并确认服务器主机公钥。')
    profile.update(authentication='password', host_key_fingerprint=values['fingerprint'])
    return profile, password


def configure(remote,values):
    from .remote import PasswordSSHTransport
    from .compute_instances import active
    if active(remote.storage.path.parent):
        raise ProjectError(409,'USE_COMPUTE_CANDIDATE','请通过“连接并检查环境”更新此实例。')
    profile,password=connection_input(values)
    if remote.pending.exists():raise ProjectError(409,'SSH_TARGET_CHANGE_PENDING','当前实例切换尚未完成核对，原连接保留。')
    with remote.guard:
        with remote.storage.connect() as db:
            if any(db.execute(f"SELECT 1 FROM {kind}_dispatches WHERE state='AWAITING_RESULT'").fetchone() for kind in ('training','evaluation','inference')):
                raise ProjectError(409,'SSH_CONNECTION_BUSY','当前有未收尾的训练、评估或推理，不能切换连接；请先完成原任务。')
        current=json.loads(remote.config.read_text('utf-8')) if remote.config.exists() else None
        key=host_key(values['command'],current)
        if key['fingerprint']!=values['fingerprint']:raise ProjectError(409,'SSH_HOST_KEY_FAILED','读取后服务器主机公钥发生变化，未保存连接。')
        profile.update(authentication='password',host_key_fingerprint=key['fingerprint'],verified_gpu_uuid=remote.training.target()['verified_gpu_uuid'])
        candidate=PasswordSSHTransport(profile,password=password)
        result=candidate.call('probe')  # Authentication and CPU-only identity check before any commit.
        if profile['verified_gpu_uuid'] not in result.get('nvidia_smi',''):
            raise ProjectError(409,'SSH_GPU_CHANGED','SSH已连接，但GPU身份发生变化；原配置保留，请先确认新实例并完成环境验证。')
        if result.get('python')!=remote.training.target()['runtime']['python']:
            raise ProjectError(409,'SSH_RUNTIME_CHANGED','实例Python环境与已验证目标不同，未替换原连接。')
        sealed=protect(password)
        with remote.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if any(db.execute(f"SELECT 1 FROM {kind}_dispatches WHERE state='AWAITING_RESULT'").fetchone() for kind in ('training','evaluation','inference')):
                raise ProjectError(409,'SSH_CONNECTION_BUSY','检查期间有新任务开始，原连接保留，请先完成该任务。')
            folder=remote.storage.path.parent/'.secrets';folder.mkdir(exist_ok=True)
            secret=folder/(uuid.uuid4().hex+'.dpapi');secret.write_bytes(sealed)
            profile['credential_file']=str(secret.resolve());temporary=remote.config.with_suffix('.json.tmp')
            try:
                if remote.config.exists():
                    history=remote.storage.path.parent/'ssh-history';history.mkdir(exist_ok=True)
                    (history/(uuid.uuid4().hex+'.json')).write_bytes(remote.config.read_bytes())
                temporary.write_text(json.dumps(profile,ensure_ascii=False,indent=2),'utf-8');temporary.replace(remote.config)
            except OSError:
                secret.unlink(missing_ok=True);raise
        # Previous credential is deliberately retained for connection rollback; never remove old trust/keys.
        result.update(checked_at=timestamp(),endpoint=candidate.endpoint,training_executed=False)
        (remote.storage.path.parent/'ssh-last-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
        return remote.info()
