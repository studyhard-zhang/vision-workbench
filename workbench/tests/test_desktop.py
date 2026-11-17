import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from app.storage import Storage
from app.server import create_server
from app.version import VERSION
from app.workspace_lock import WorkspaceLease
from desktop import Session,validate_workspace,workspace_from_profile,initialize_workspace

class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.var=self.root/'var';Storage(self.var/'platform.db')
    def tearDown(self):self.tmp.cleanup()
    def test_missing_workspace_does_not_create_empty_database(self):
        missing=self.root/'missing'
        with self.assertRaises(ValueError):validate_workspace(missing)
        self.assertFalse(missing.exists())
    def test_explicit_new_workspace_is_empty_and_has_no_compute_identity(self):
        path=initialize_workspace(self.root/'new workspace')
        self.assertEqual(validate_workspace(path),path)
        store=Storage(path/'platform.db')
        with store.connect() as db:
            for table in ('projects','datasets','vision_tasks','training_jobs','model_versions'):
                self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
        self.assertFalse((path/'active-compute.json').exists())
        from app.training import TrainingRepository
        training=TrainingRepository(store)
        self.assertIsNone(training.target()['verified_run_id'])
        self.assertFalse(any(training.capabilities()[0].values()))
    def test_create_refuses_existing_data_and_nonempty_directories(self):
        before=(self.var/'platform.db').read_bytes()
        with self.assertRaises(ValueError):initialize_workspace(self.var)
        self.assertEqual((self.var/'platform.db').read_bytes(),before)
        other=self.root/'photos';other.mkdir();(other/'keep.txt').write_text('keep')
        with self.assertRaises(ValueError):initialize_workspace(other)
        self.assertEqual((other/'keep.txt').read_text(),'keep')
        self.assertFalse((other/'platform.db').exists())
    def test_profile_preserves_unicode_and_percent_in_path(self):
        self.root.joinpath('desktop.ini').write_text('[workspace]\ndirectory=E:\\原工作区%20\\var\n','utf-8-sig')
        self.assertEqual(str(workspace_from_profile(self.root)),r'E:\原工作区%20\var')
    def test_same_workspace_with_old_version_refuses_second_writer(self):
        value={'product':'vision-workbench','data_dir':str(self.var),'version':'old','schema':15}
        with patch('desktop.runtime_info',return_value=value),patch('app.server.create_server') as create,self.assertRaises(ValueError) as error:
            Session(self.var,8774).start()
        self.assertIn(str(self.var),str(error.exception))
        create.assert_not_called()
    def test_other_workspace_uses_its_own_port_and_keeps_existing_service(self):
        other=self.root/'other';Storage(other/'platform.db')
        server=create_server(0,other/'platform.db');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        session=Session(self.var,server.server_port)
        from desktop import runtime_info
        try:
            url=session.start();self.assertFalse(session.reused);self.assertNotEqual(session.port,server.server_port)
            self.assertEqual(url,f'http://127.0.0.1:{session.port}/')
            self.assertEqual(Path(runtime_info(session.port)['data_dir']),self.var)
            body=json.dumps({'name':'Desktop isolated fixture','description':''}).encode()
            with urlopen(Request(url+'api/projects',data=body,headers={'Content-Type':'application/json','Origin':url.rstrip('/')}),timeout=5) as response:
                self.assertEqual(response.status,201)
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url+'api/projects',data=body,headers={'Content-Type':'application/json','Origin':f'http://127.0.0.1:{server.server_port}'}),timeout=5)
            self.assertEqual(error.exception.code,403)
            with urlopen(f'http://127.0.0.1:{server.server_port}/api/projects',timeout=5) as response:
                self.assertEqual(json.load(response)['projects'],[])
            session.close()
            self.assertEqual(Path(runtime_info(server.server_port)['data_dir']),other)
        finally:session.close();server.shutdown();server.server_close();thread.join()
    def test_reused_service_is_not_stopped_by_window_close(self):
        server=create_server(0,self.var/'platform.db');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            session=Session(self.var,server.server_port);session.start();self.assertTrue(session.reused);session.close()
            from desktop import runtime_info
            self.assertEqual(runtime_info(server.server_port)['version'],VERSION)
        finally:server.shutdown();server.server_close();thread.join()
    def test_busy_unknown_port_uses_free_port_without_closing_unknown_listener(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen()
            occupied=listener.getsockname()[1];session=Session(self.var,occupied)
            try:
                with patch('desktop.runtime_info',return_value=None):session.start()
                self.assertNotEqual(session.port,occupied)
                from desktop import runtime_info
                self.assertEqual(Path(runtime_info(session.port)['data_dir']),self.var)
            finally:session.close()
            with socket.socket() as probe:self.assertEqual(probe.connect_ex(('127.0.0.1',occupied)),0)
    def test_workspace_busy_on_another_port_is_not_bypassed(self):
        lease=WorkspaceLease(self.var/'platform.db')
        try:
            with patch('desktop.runtime_info',return_value=None),self.assertRaisesRegex(ValueError,'无法打开所选工作区'):
                Session(self.var,0).start()
        finally:lease.close()
    def test_workspace_lease_refuses_second_writer_and_releases(self):
        first=WorkspaceLease(self.var/'platform.db')
        try:
            with self.assertRaises(OSError):WorkspaceLease(self.var/'platform.db')
        finally:first.close()
        second=WorkspaceLease(self.var/'platform.db');second.close()
    def test_runtime_identifies_workspace_and_owned_session_can_restart(self):
        with socket.socket() as temporary:temporary.bind(('127.0.0.1',0));port=temporary.getsockname()[1]
        session=Session(self.var,port)
        try:
            session.start();self.assertFalse(session.reused)
            from desktop import runtime_info
            self.assertEqual(Path(runtime_info(port)['data_dir']),self.var)
        finally:session.close()
        again=Session(self.var,port);again.start();again.close()
