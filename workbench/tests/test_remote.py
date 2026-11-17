import hashlib
import json
import sys
import errno
from types import SimpleNamespace
from unittest.mock import Mock,patch

from test_versions import VersionFixture
import test_executions as contract
from app.storage import ProjectError
from app.training import TARGET,TrainingRepository
from app.executions import ExecutionRepository
from app.remote import RemoteRepository,SSHTransport
from tools import ssh_control


class RemoteTests(VersionFixture):
    job=contract.ExecutionTests.job
    request=contract.ExecutionTests.request
    receipt=contract.ExecutionTests.receipt

    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions)
        self.executions=ExecutionRepository(self.store,self.training);self.remote=RemoteRepository(self.store,self.training,self.executions)
        self.fake=SimpleNamespace(endpoint='root@fixture.invalid:22',call=Mock(return_value={'nvidia_smi':TARGET['verified_gpu_uuid']}),copy=Mock())
        self.patch_transport=patch.object(self.remote,'transport',return_value=self.fake);self.patch_transport.start()
        self.patch_worker=patch.object(self.remote,'start_worker');self.patch_worker.start()

    def tearDown(self):
        self.patch_worker.stop();self.patch_transport.stop();self.remote.close();super().tearDown()

    def row(self,job):
        with self.store.connect() as db:return dict(db.execute('SELECT s.*,t.project_id,j.task_id,d.package_sha256 FROM training_remote_sessions s JOIN training_jobs j ON j.id=s.job_id JOIN vision_tasks t ON t.id=j.task_id JOIN training_dispatches d ON d.job_id=j.id WHERE s.job_id=?',(job['id'],)).fetchone())

    def test_install_accepts_shared_configuration_module_without_starting_process(self):
        job=self.job();self.executions.dispatch(self.pid,self.a,job['id'])
        path,_,_=self.executions.package(self.pid,self.a,job['id'])
        root=self.path.parent/'remote-install';directory=root/job['id'];directory.mkdir(parents=True)
        (directory/'execution.zip.upload').write_bytes(path.read_bytes())
        with patch.object(ssh_control,'ROOT',root),patch.object(ssh_control.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=TARGET['verified_gpu_uuid'])),patch.object(ssh_control.subprocess,'Popen') as process:
            result=ssh_control.handle({'action':'install','job_id':job['id'],'expected_gpu_uuid':TARGET['verified_gpu_uuid'],'package_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        self.assertTrue((directory/'training_config.py').is_file());self.assertFalse(result['started']);process.assert_not_called()

    def test_connection_check_is_cpu_only_and_wrong_gpu_never_dispatches(self):
        value=self.remote.check();self.assertFalse(value['training_executed']);self.fake.call.assert_called_with('probe')
        self.fake.call.return_value={'nvidia_smi':'other GPU'};job=self.job()
        with self.assertRaises(ProjectError):self.remote.submit(self.pid,self.a,job['id'])
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'QUEUED')
        self.assertIsNone(self.training.get(self.pid,self.a,job['id'])['dispatch'])

    def test_duplicate_submission_reuses_reservation_and_does_not_start_via_http(self):
        job=self.job();first=self.remote.submit(self.pid,self.a,job['id']);again=self.remote.submit(self.pid,self.a,job['id'])
        self.assertEqual(first['dispatch']['id'],again['dispatch']['id']);self.fake.copy.assert_not_called()
        self.assertEqual([c.args[0] for c in self.fake.call.call_args_list],['probe','probe'])

    def test_launch_once_and_reconnect_poll_existing_process(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id'])
        live={'directory_exists':True,'launch':{'pid':123},'alive':True,'started':True,'result_ready':False,'progress':{'epochs_completed':0},'log_tail':'CPU mocked process'}
        initial={**live,'launch':{},'alive':False,'started':False}
        self.fake.call.side_effect=lambda action,**kw: initial if action in ('status','prepare') else live
        self.remote.reconcile(self.row(job));self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'RUNNING')
        self.fake.call.side_effect=lambda action,**kw:live
        self.remote.reconcile(self.row(job))
        self.assertEqual(sum(c.args[0]=='launch' for c in self.fake.call.call_args_list),1)

    def test_lost_launch_metadata_is_never_replayed_after_reconnect(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id'])
        with self.store.connect() as db:db.execute("UPDATE training_remote_sessions SET launched=1,phase='RECONNECTING' WHERE job_id=?",(job['id'],))
        self.fake.call.return_value={'directory_exists':False,'launch':{},'alive':False,'started':False,'result_ready':False}
        with self.assertRaises(ProjectError) as e:self.remote.reconcile(self.row(job))
        self.assertEqual(e.exception.code,'REMOTE_UNCERTAIN');self.fake.copy.assert_not_called()

    def test_download_validates_then_commits_success_and_completes_session(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id']);raw=self.receipt(job)
        self.fake.call.return_value={'result_ready':True,'alive':False,'resources':{'checked':True,'free':True},'result_sha256':hashlib.sha256(raw).hexdigest(),'progress':{'epochs_completed':1}}
        self.fake.copy.side_effect=lambda path,job_id,upload:path.write_bytes(raw)
        self.remote.reconcile(self.row(job));record=self.training.get(self.pid,self.a,job['id'])
        self.assertEqual(record['state'],'SUCCEEDED');self.assertEqual(record['remote']['phase'],'COMPLETED')
        self.assertTrue(record['remote']['progress']['remote_process_exited']);self.assertTrue(record['remote']['progress']['resources']['free'])

    def test_failed_copy_does_not_release_target_or_mark_success(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id'])
        self.fake.call.return_value={'result_ready':True,'alive':False,'resources':{'checked':True,'free':True},'result_sha256':'0'*64,'progress':{}}
        self.fake.copy.side_effect=lambda path,job_id,upload:path.write_bytes(b'corrupt copy')
        with self.assertRaises(ProjectError):self.remote.reconcile(self.row(job))
        record=self.training.get(self.pid,self.a,job['id']);self.assertEqual(record['state'],'DISPATCHED');self.assertEqual(record['dispatch']['state'],'AWAITING_RESULT')

    def test_ready_result_keeps_reservation_until_process_and_both_locks_are_free(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id']);raw=self.receipt(job)
        ready={'result_ready':True,'result_sha256':hashlib.sha256(raw).hexdigest(),'progress':{'epochs_completed':1}}
        for proof in ({'alive':True,'resources':{'checked':True,'free':True}},
                      {'alive':False,'resources':{'checked':True,'free':False}},
                      {'alive':False,'resources':{'checked':False,'free':False}},
                      {'resources':{'checked':True,'free':True}}, {'alive':False}):
            with self.subTest(proof=proof):
                self.fake.call.return_value={**ready,**proof};self.remote.reconcile(self.row(job))
                record=self.training.get(self.pid,self.a,job['id'])
                self.assertEqual(record['remote']['phase'],'RELEASING');self.assertEqual(record['dispatch']['state'],'AWAITING_RESULT')
                self.fake.copy.assert_not_called()
        self.fake.call.return_value={**ready,'alive':False,'resources':{'checked':True,'free':True}}
        self.fake.copy.side_effect=lambda path,job_id,upload:path.write_bytes(raw)
        self.remote.reconcile(self.row(job));self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'SUCCEEDED')
        self.assertEqual(self.fake.copy.call_count,1)

    def test_manual_upload_cannot_bypass_resource_proof_for_an_active_ssh_session(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id'])
        with self.assertRaises(ProjectError) as e:self.executions.receive(self.pid,self.a,job['id'],self.receipt(job))
        self.assertEqual(e.exception.code,'SSH_RESOURCE_PROOF_REQUIRED')
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['dispatch']['state'],'AWAITING_RESULT')

    def test_pause_survives_stale_poll_and_repository_restart_then_resumes_same_pid(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id'])
        live={'directory_exists':True,'launch':{'pid':123},'alive':True,'started':True,'result_ready':False,'progress':{'epochs_completed':1},'log_tail':'last known log'}
        self.fake.call.return_value=live;self.remote.reconcile(self.row(job));stale=self.row(job)
        self.remote.pause(self.pid,self.a,job['id']);self.fake.call.reset_mock()
        restarted=RemoteRepository(self.store,self.training,self.executions)
        with patch.object(restarted,'transport',return_value=self.fake):restarted.reconcile(stale)
        self.fake.call.assert_not_called();self.remote.update(job['id'],'RUNNING')
        record=self.training.get(self.pid,self.a,job['id']);self.assertEqual(record['remote']['phase'],'PAUSED')
        self.assertEqual(record['remote']['progress']['log_tail'],'last known log');self.assertEqual(record['dispatch']['state'],'AWAITING_RESULT')
        self.fake.call.return_value={'nvidia_smi':TARGET['verified_gpu_uuid']};self.remote.submit(self.pid,self.a,job['id'])
        self.fake.call.return_value=live;self.remote.reconcile(self.row(job))
        self.assertEqual([c.args[0] for c in self.fake.call.call_args_list],['probe','status'])
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['remote']['progress']['remote_pid'],123)

    def test_cancel_overrides_pause_and_cannot_be_paused_again(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id'])
        with self.store.connect() as db:db.execute("UPDATE training_jobs SET state='RUNNING' WHERE id=?",(job['id'],))
        self.remote.pause(self.pid,self.a,job['id']);self.remote.cancel(self.pid,self.a,job['id'])
        record=self.training.get(self.pid,self.a,job['id']);self.assertEqual(record['remote']['phase'],'CANCELLING')
        with self.assertRaises(ProjectError) as e:self.remote.pause(self.pid,self.a,job['id'])
        self.assertEqual(e.exception.code,'REMOTE_CANCELLING');self.assertEqual(record['dispatch']['state'],'AWAITING_RESULT')

    def test_changed_endpoint_is_refused_before_any_probe_or_transfer(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id']);self.fake.call.reset_mock()
        self.fake.endpoint='root@new-fixture.invalid:9999'
        with self.assertRaises(ProjectError) as e:self.remote.submit(self.pid,self.a,job['id'])
        self.assertEqual(e.exception.code,'SSH_ENDPOINT_CHANGED');self.fake.call.assert_not_called();self.fake.copy.assert_not_called()
        with self.assertRaises(ProjectError):self.remote.reconcile(self.row(job))
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['dispatch']['state'],'AWAITING_RESULT')

    def test_pending_target_blocks_old_target_submission_and_manual_dispatch(self):
        job=self.job();self.remote.pending.write_text(json.dumps({'endpoint':'new:22','status':'DEPENDENCIES_MISSING'}),encoding='utf-8')
        self.fake.call.reset_mock()
        for action in (self.remote.submit,self.executions.dispatch):
            with self.assertRaises(ProjectError) as e:action(self.pid,self.a,job['id'])
            self.assertEqual(e.exception.code,'SSH_TARGET_CHANGE_PENDING')
        self.assertFalse(self.remote.info()['execution_enabled']);self.assertFalse(self.training.options()['targets'][0]['enabled'])
        self.fake.call.assert_not_called();self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'QUEUED')

    def test_pending_target_also_blocks_a_preexisting_upload_session_before_launch(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id']);self.fake.call.reset_mock()
        self.remote.pending.write_text('{}',encoding='utf-8')
        with self.assertRaises(ProjectError) as e:self.remote.reconcile(self.row(job))
        self.assertEqual(e.exception.code,'SSH_TARGET_CHANGE_PENDING');self.fake.call.assert_not_called();self.fake.copy.assert_not_called()
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['dispatch']['state'],'AWAITING_RESULT')

    def test_worker_retains_progress_and_reservation_after_connection_failure(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id']);self.remote.update(job['id'],'RUNNING',{'epochs_completed':1,'log_tail':'kept'})
        with patch.object(self.remote,'reconcile',side_effect=ProjectError(503,'SSH_AUTHENTICATION_FAILED','denied')),patch.object(self.remote.stop_event,'wait',return_value=True):self.remote.work()
        record=self.training.get(self.pid,self.a,job['id'])
        self.assertEqual(record['remote']['phase'],'ERROR');self.assertEqual(record['remote']['progress']['log_tail'],'kept')
        self.assertEqual(record['remote']['progress']['error_code'],'SSH_AUTHENTICATION_FAILED')
        self.assertEqual(record['dispatch']['state'],'AWAITING_RESULT');self.assertEqual(record['state'],'DISPATCHED')

    def test_transport_distinguishes_authentication_and_host_key_failures(self):
        transport=SSHTransport({'host':'fixture.invalid','port':22,'user':'root','identity_file':'key','known_hosts_file':'hosts'})
        for stderr,code in [(b'Host key verification failed.','SSH_HOST_KEY_FAILED'),(b'Permission denied (publickey).','SSH_AUTHENTICATION_FAILED'),(b'Connection refused','SSH_FAILED')]:
            with self.subTest(code=code),patch('app.remote.subprocess.run',return_value=SimpleNamespace(returncode=255,stderr=stderr)):
                with self.assertRaises(ProjectError) as e:transport.run(['ssh'])
                self.assertEqual(e.exception.code,code)

    def test_resource_probe_requires_both_existing_locks_and_closes_descriptors(self):
        directory=self.path.parent/'locks-job';directory.mkdir();(directory/'.job.lock').touch()
        home=self.path.parent/'remote-home';locks=home/'.vision-workbench-locks';locks.mkdir(parents=True);(locks/'gpu-0.lock').touch()
        fcntl=SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=Mock())
        with patch.dict(sys.modules,{'fcntl':fcntl}),patch.object(ssh_control.Path,'home',return_value=home):
            self.assertTrue(ssh_control.resource_state(directory)['free']);self.assertEqual(fcntl.flock.call_count,2)
            fcntl.flock.side_effect=[None,BlockingIOError(errno.EAGAIN,'held')]
            self.assertEqual(ssh_control.resource_state(directory),{'checked':True,'free':False,'reason':'managed_lock_held'})
            (locks/'gpu-0.lock').unlink();fcntl.flock.side_effect=None
            self.assertFalse(ssh_control.resource_state(directory)['checked'])
            self.assertFalse((locks/'gpu-0.lock').exists())

    def test_cancel_waits_for_remote_interruption_receipt(self):
        job=self.job();self.remote.submit(self.pid,self.a,job['id']);self.remote.cancel(self.pid,self.a,job['id'])
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'DISPATCHED')
        raw=self.receipt(job,status='interrupted');pending={'directory_exists':True,'launch':{'pid':123},'alive':True,'result_ready':False}
        completed={'result_ready':True,'alive':False,'resources':{'checked':True,'free':True},'result_sha256':hashlib.sha256(raw).hexdigest(),'progress':{}}
        self.fake.call.side_effect=lambda action,**kw: completed if action=='stop' else pending
        self.fake.copy.side_effect=lambda path,job_id,upload:path.write_bytes(raw)
        self.remote.reconcile(self.row(job));self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'INTERRUPTED')

    def test_transport_pins_host_key_and_does_not_construct_local_shell_command(self):
        transport=SSHTransport({'host':'fixture.invalid','port':22,'user':'root','identity_file':'local.key','known_hosts_file':'pinned.hosts'})
        with patch.object(transport,'run',return_value=b'{"ok":true,"value":{}}') as run:transport.call('probe')
        argv=run.call_args.args[0];self.assertIn('StrictHostKeyChecking=yes',argv);self.assertIn('BatchMode=yes',argv)
        self.assertIn('IdentitiesOnly=yes',argv);self.assertIn('UserKnownHostsFile=pinned.hosts',argv)
        with self.assertRaises(ProjectError):SSHTransport({'host':'host;command','port':22,'user':'root'})

    def test_remote_stop_refuses_recycled_pid_before_sending_signals(self):
        directory=self.path.parent/'remote-root';directory.mkdir();job='a'*32
        before={'result_ready':False,'launch':{'pid':999,'start_tick':'old'},'alive':True}
        with (patch.object(ssh_control,'ROOT',directory),patch.object(ssh_control,'status',return_value=before),
              patch.object(ssh_control.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=TARGET['verified_gpu_uuid']+'\n')),
              patch.object(ssh_control,'identity',return_value='new'),patch.object(ssh_control.os,'killpg',create=True) as kill):
            with self.assertRaises(RuntimeError):ssh_control.handle({'action':'stop','job_id':job,'expected_gpu_uuid':TARGET['verified_gpu_uuid']})
            kill.assert_not_called()

    def test_second_workbench_cannot_listen_on_the_same_port(self):
        from app.server import create_server
        first=create_server(0,self.path)
        try:
            with self.assertRaises(OSError):create_server(first.server_port,self.path)
        finally:first.server_close()
