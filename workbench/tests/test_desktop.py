import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
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
    def test_wrong_workspace_or_version_refuses_existing_service(self):
        for value in [{'product':'vision-workbench','data_dir':str(self.root/'other'),'version':VERSION,'schema':15},{'product':'vision-workbench','data_dir':str(self.var),'version':'old','schema':15}]:
            with patch('desktop.runtime_info',return_value=value),self.assertRaises(ValueError):Session(self.var,8774).start()
    def test_reused_service_is_not_stopped_by_window_close(self):
        server=create_server(0,self.var/'platform.db');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            session=Session(self.var,server.server_port);session.start();self.assertTrue(session.reused);session.close()
            from desktop import runtime_info
            self.assertEqual(runtime_info(server.server_port)['version'],VERSION)
        finally:server.shutdown();server.server_close();thread.join()
    def test_busy_unknown_port_does_not_initialize_workspace(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen()
            with patch('desktop.runtime_info',return_value=None),patch('app.server.create_server') as create,self.assertRaises(ValueError):Session(self.var,listener.getsockname()[1]).start()
            create.assert_not_called()
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
