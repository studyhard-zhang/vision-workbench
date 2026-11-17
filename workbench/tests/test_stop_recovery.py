import json
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
from test_versions import VersionFixture
import test_executions as fixtures
from app.training import TrainingRepository, TARGET, unavailable_target
from app.executions import ExecutionRepository
from app.remote import RemoteRepository
from app.storage import ProjectError
from tools.run_training_job import pack_result
from tools import ssh_control


class StopRecoveryTests(VersionFixture):
    job = fixtures.ExecutionTests.job

    def setUp(self):
        super().setUp()
        self.vid = self.publish()['version']['id']
        self.training = TrainingRepository(self.store, self.versions)
        self.executions = ExecutionRepository(self.store, self.training)
        self.remote = RemoteRepository(self.store, self.training, self.executions)
        self.transport = SimpleNamespace(endpoint='root@fixture.invalid:22',call=Mock(return_value={'nvidia_smi':TARGET['verified_gpu_uuid']}),copy=Mock())
        self.transport_patch = patch.object(self.remote,'transport',return_value=self.transport)
        self.worker_patch = patch.object(self.remote,'start_worker')
        self.transport_patch.start();self.worker_patch.start()
        self.original = self.job()
        self.remote.submit(self.pid,self.a,self.original['id'])
        self.addCleanup(self.transport_patch.stop);self.addCleanup(self.worker_patch.stop);self.addCleanup(self.remote.close)

    def row(self):
        with self.store.connect() as db:
            return dict(db.execute('SELECT s.*,t.project_id,j.task_id,d.package_sha256 FROM training_remote_sessions s JOIN training_jobs j ON j.id=s.job_id JOIN vision_tasks t ON t.id=j.task_id JOIN training_dispatches d ON d.job_id=j.id WHERE s.job_id=?',(self.original['id'],)).fetchone())

    def proof(self,**values):
        return dict(alive=False,resources={'checked':True,'free':True},progress={'epochs_completed':2},**values)

    def cancelling(self):
        with self.store.connect() as db:db.execute("UPDATE training_remote_sessions SET cancel_requested=1,phase='ERROR'")

    def abandon(self):
        self.remote.update(self.original['id'],'ERROR')
        return self.remote.abandon(self.pid,self.a,self.original['id'],dict(confirmation_name=self.original['name'],visibility_revision=0,instance_unavailable=True))['job']

    def test_failed_result_closure_with_release_proof_can_be_deleted_and_gpu_reused(self):
        self.cancelling()
        self.transport.call.side_effect=[dict(result_ready=False,directory_exists=True,launch={'pid':12}),ProjectError(503,'SSH_FAILED','oversized checkpoint'),self.proof()]
        self.remote.reconcile(self.row())
        job=self.training.get(self.pid,self.a,self.original['id'])
        self.assertEqual(job['state'],'INTERRUPTED');self.assertTrue(job['can_delete'])
        self.assertEqual(job['dispatch']['result']['kind'],'LOCAL_INTERRUPTION')
        with self.store.connect() as db:self.assertFalse(unavailable_target(db,job['snapshot']['target']))
        self.training.change_visibility(self.pid,self.a,[dict(id=job['id'],confirmation_name=job['name'],visibility_revision=job['visibility_revision'])])
        self.assertEqual(self.job()['state'],'QUEUED')

    def test_missing_release_proof_never_finishes_or_deletes(self):
        self.cancelling()
        for proof in ({'alive':True,'resources':{'checked':True,'free':True}}, {'alive':False,'resources':{'checked':True,'free':False}}, {'alive':False,'resources':{'checked':False,'free':True}}):
            with self.assertRaises(ProjectError):self.remote.finish_stopped(self.pid,self.a,self.original['id'],proof)
        self.assertEqual(self.training.get(self.pid,self.a,self.original['id'])['dispatch']['state'],'AWAITING_RESULT')

    def test_stop_and_trash_is_normal_deletion_and_preserves_source(self):
        self.cancelling();self.transport.call.return_value=self.proof()
        values=dict(confirmation_name=self.original['name'],visibility_revision=0)
        result=self.remote.stop_and_trash(self.pid,self.a,self.original['id'],values)['job']
        self.assertIsNotNone(result['deleted_at']);self.assertEqual(result['snapshot'],self.original['snapshot'])
        self.assertEqual(result['state'],'INTERRUPTED')

    def test_recover_disabled_gpu_keeps_historical_abandonment(self):
        abandoned=self.abandon()
        self.transport.call.side_effect=[{'nvidia_smi':TARGET['verified_gpu_uuid']},self.proof()]
        self.assertEqual(self.remote.recover_device()['recovered'],1)
        job=self.training.get(self.pid,self.a,self.original['id'])
        self.assertEqual(job['deleted_at'],abandoned['deleted_at']);self.assertEqual(job['state'],'FAILED')
        self.assertFalse(job['remote']['progress']['local_abandonment']['remote_stop_confirmed'])
        self.assertTrue(job['remote']['progress']['local_abandonment_resolution']['resources_released'])
        self.assertEqual(self.job()['state'],'QUEUED')

    def test_recover_busy_gpu_does_not_clear_block(self):
        self.abandon()
        self.transport.call.side_effect=[{'nvidia_smi':TARGET['verified_gpu_uuid']},{'alive':False,'resources':{'checked':True,'free':False}}]
        with self.assertRaises(ProjectError):self.remote.recover_device()
        with self.store.connect() as db:self.assertTrue(unavailable_target(db,self.original['snapshot']['target']))

    def test_large_interrupted_weights_retained_without_blocking_receipt(self):
        folder=self.path.parent/'pack';weights=folder/'work/train/weights';weights.mkdir(parents=True)
        with (weights/'last.pt').open('wb') as stream:stream.truncate(121*1024**2)
        (folder/'console.log').write_text('stopped')
        request={key:'fixture' for key in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256','return_token')}
        report={'status':'interrupted'};pack_result(folder,request,report)
        with zipfile.ZipFile(folder/'return-result.zip') as archive:
            receipt=json.loads(archive.read('receipt.json'));self.assertNotIn('last.pt',archive.namelist())
            self.assertEqual(receipt['report']['retained_artifacts']['last.pt']['bytes'],121*1024**2)
            self.assertIn('console.log',archive.namelist())
        self.assertTrue((weights/'last.pt').exists())

    def test_stop_waits_for_inherited_locks_before_cpu_closure(self):
        root=self.path.parent/'remote';directory=root/self.original['id'];directory.mkdir(parents=True);(directory/'.job.lock').touch()
        before={'result_ready':False,'launch':{},'started':True}
        after={'result_ready':True,'alive':False}
        with patch.object(ssh_control,'ROOT',root),patch.object(ssh_control,'status',side_effect=[before,after]),patch.object(ssh_control,'resource_state',side_effect=[{'checked':True,'free':False},{'checked':True,'free':True}]) as resources,patch.object(ssh_control.time,'sleep'),patch.object(ssh_control.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=TARGET['verified_gpu_uuid'])) as run:
            result=ssh_control.handle({'action':'stop','job_id':self.original['id'],'expected_gpu_uuid':TARGET['verified_gpu_uuid']})
        self.assertTrue(result['result_ready']);self.assertEqual(resources.call_count,2)
        self.assertIn('--recover-interrupted',run.call_args.args[0])
