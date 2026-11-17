import json,socket,threading,time,os,hashlib
from pathlib import Path
from unittest.mock import patch
import paramiko
from test_versions import VersionFixture
from app.training import TrainingRepository,TARGET
from app.executions import ExecutionRepository
from app.remote import RemoteRepository,PasswordSSHTransport
from app.ssh_setup import parse_command,host_key,configure,fingerprint
from app.ssh_credentials import protect,unprotect
from app.storage import ProjectError

class LocalSSH:
    """Real password SSH handshake/exec, entirely on loopback; no subprocess or GPU."""
    def __init__(self,gpu=TARGET['verified_gpu_uuid']):
        self.key=paramiko.RSAKey.generate(2048);self.password='QA-only-'+str(time.monotonic_ns());self.gpu=gpu;self.stop=threading.Event();self.clients=[];self.auth_attempts=0;self.source_path=None
        self.socket=socket.socket();self.socket.bind(('127.0.0.1',0));self.socket.listen();self.socket.settimeout(.1);self.port=self.socket.getsockname()[1];self.thread=threading.Thread(target=self.serve,daemon=True);self.thread.start()
    @property
    def command(self):return f'ssh -p {self.port} root@127.0.0.1'
    def serve(self):
        owner=self
        class Files(paramiko.SFTPServerInterface):
            def stat(self,path):
                if owner.source_path is None:return paramiko.SFTP_NO_SUCH_FILE
                return paramiko.SFTPAttributes.from_stat(owner.source_path.stat())
            def open(self,path,flags,attr):
                if owner.source_path is None or flags!=os.O_RDONLY:return paramiko.SFTP_PERMISSION_DENIED
                handle=paramiko.SFTPHandle(flags);handle.readfile=owner.source_path.open('rb');return handle
        class Server(paramiko.ServerInterface):
            def get_allowed_auths(self,username):return 'password'
            def check_auth_password(self,username,password):
                owner.auth_attempts+=1
                return paramiko.AUTH_SUCCESSFUL if username=='root' and password==owner.password else paramiko.AUTH_FAILED
            def check_channel_request(self,kind,chanid):return paramiko.OPEN_SUCCEEDED if kind=='session' else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
            def check_channel_exec_request(self,channel,command):
                def reply():
                    time.sleep(.03);channel.sendall(json.dumps({'ok':True,'value':{'python':TARGET['runtime']['python'],'nvidia_smi':owner.gpu}}).encode());channel.send_exit_status(0);channel.close()
                threading.Thread(target=reply,daemon=True).start();return True
        while not self.stop.is_set():
            try:client,_=self.socket.accept()
            except socket.timeout:continue
            except OSError:break
            transport=paramiko.Transport(client);self.clients.append(transport);transport.add_server_key(self.key);transport.set_subsystem_handler('sftp',paramiko.SFTPServer,Files)
            try:transport.start_server(server=Server())
            except (OSError,paramiko.SSHException,EOFError):transport.close()
    def close(self):
        self.stop.set();self.socket.close()
        for client in self.clients:client.close()
        self.thread.join(timeout=2)

class SSHSetupTests(VersionFixture):
    def setUp(self):
        super().setUp();self.training=TrainingRepository(self.store,self.versions);self.remote=RemoteRepository(self.store,self.training,ExecutionRepository(self.store,self.training));self.ssh=LocalSSH()
    def tearDown(self):self.ssh.close();self.remote.close();super().tearDown()
    def values(self,**changes):return {'command':self.ssh.command,'password':self.ssh.password,'fingerprint':fingerprint(self.ssh.key),'confirmed_host':True,**changes}
    def test_parser_never_accepts_shell_remote_commands_or_ssh_options(self):
        self.assertEqual(parse_command('ssh root@example.com')['port'],22)
        for command in ('ssh -p 22 root@host;calc','ssh -p 22 root@host whoami','ssh -o ProxyCommand=calc root@host','ssh -p 0 root@host','ssh -p 65536 root@host','ssh -p 22 guest@host','ssh root@host\ncalc'):
            with self.subTest(command=command),self.assertRaises(ProjectError):parse_command(command)
    def test_password_auth_saved_encrypted_restarts_without_plaintext_leak(self):
        key=host_key(self.ssh.command);self.assertFalse(key['authenticated']);self.assertEqual(self.ssh.auth_attempts,0)
        info=configure(self.remote,self.values());self.assertEqual(info['authentication'],'password');self.assertEqual(info['command'],self.ssh.command)
        profile=json.loads(self.remote.config.read_text('utf-8'));secret=Path(profile['credential_file'])
        self.assertNotIn(self.ssh.password,self.remote.config.read_text('utf-8'));self.assertNotIn(self.ssh.password,json.dumps(info));self.assertNotIn(self.ssh.password.encode(),secret.read_bytes())
        self.assertEqual(unprotect(secret),self.ssh.password);self.assertEqual(self.remote.check()['nvidia_smi'],TARGET['verified_gpu_uuid']);self.assertGreaterEqual(self.ssh.auth_attempts,2)
    def test_bad_password_does_not_commit_or_echo_secret(self):
        password=self.ssh.password+'wrong'
        with self.assertRaises(ProjectError) as error:configure(self.remote,self.values(password=password))
        self.assertEqual(error.exception.code,'SSH_AUTHENTICATION_FAILED');self.assertNotIn(password,str(error.exception));self.assertFalse(self.remote.config.exists())
    def test_wrong_or_unconfirmed_host_never_authenticates(self):
        for changes in ({'fingerprint':'SHA256:'+'A'*43},{'fingerprint':[]},{'confirmed_host':False}):
            with self.subTest(changes=changes),self.assertRaises(ProjectError):configure(self.remote,self.values(**changes))
        self.assertEqual(self.ssh.auth_attempts,0);self.assertFalse(self.remote.config.exists())
    def test_gpu_change_retains_original_configuration(self):
        configure(self.remote,self.values());original=self.remote.config.read_bytes();self.ssh.gpu='GPU-other'
        with self.assertRaises(ProjectError) as error:configure(self.remote,self.values())
        self.assertEqual(error.exception.code,'SSH_GPU_CHANGED');self.assertEqual(self.remote.config.read_bytes(),original)
    def test_password_channel_copy_uses_sftp_and_no_subprocess(self):
        configure(self.remote,self.values());transport=self.remote.transport();local=self.path.parent/'input.zip';local.write_bytes(b'fixture')
        with patch.object(transport,'client') as client,patch('app.remote.subprocess.run') as run:
            transport.copy(local,'a'*32,True);client.return_value.open_sftp.return_value.__enter__.return_value.put.assert_called_once();run.assert_not_called()
    def test_large_result_transfers_exact_bytes_with_bounded_parallel_sftp(self):
        configure(self.remote,self.values());self.ssh.source_path=self.path.parent/'remote-result.zip';data=bytes(range(256))*(36*1024);self.ssh.source_path.write_bytes(data)
        destination=self.path.parent/'returned-result.zip';self.remote.transport().copy(destination,'a'*32,False)
        self.assertEqual(hashlib.sha256(destination.read_bytes()).digest(),hashlib.sha256(data).digest());self.assertFalse(list(self.path.parent.glob('.sftp-parts-*')))
