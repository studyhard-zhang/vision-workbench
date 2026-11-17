import contextlib,errno,hashlib,io,json,shutil,sys,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from test_versions import VersionFixture
import test_executions as contract
from app.storage import ProjectError
from app.training import TrainingRepository,TARGET
from app.executions import ExecutionRepository
from app.remote import RemoteRepository
from tools import run_training_job as runner


class FailureClassificationTests(unittest.TestCase):
    def test_observed_resource_dependency_and_unknown_errors_have_distinct_actions(self):
        cases=[(RuntimeError('CUDA out of memory. Tried to allocate 2 GiB'),'CUDA_OUT_OF_MEMORY'),(MemoryError(),'HOST_MEMORY_EXHAUSTED'),
            (OSError(errno.ENOSPC,'No space left on device'),'STORAGE_FULL'),(PermissionError(errno.EACCES,'denied'),'STORAGE_PERMISSION_DENIED'),
            (ModuleNotFoundError("No module named 'dill'"),'RUNTIME_DEPENDENCY_MISSING'),(RuntimeError('unrecognized training exception'),'TRAINING_EXECUTION_FAILED')]
        for error,code in cases:
            with self.subTest(code=code):
                diagnostic=runner.failure_details(error);self.assertEqual(diagnostic['code'],code);self.assertTrue(diagnostic['title']);self.assertTrue(diagnostic['next_step'])
        self.assertEqual(runner.failure_details(RuntimeError('unknown'),'finalization')['code'],'RESULT_FINALIZATION_FAILED')
        self.assertEqual(runner.failure_details(RuntimeError('unknown'),existing_code='CHECKPOINT_DEPENDENCY_MISSING')['code'],'CHECKPOINT_DEPENDENCY_MISSING')

    def test_chained_storage_failure_has_priority_without_importing_torch(self):
        inner=OSError(errno.ENOSPC,'full');outer=RuntimeError('stream writer failed');outer.__cause__=inner
        self.assertEqual(runner.failure_details(outer)['code'],'STORAGE_FULL')


class FailureRecoveryTests(VersionFixture):
    job=contract.ExecutionTests.job;receipt=contract.ExecutionTests.receipt;request=contract.ExecutionTests.request;unpack=contract.ExecutionTests.unpack
    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions);self.executions=ExecutionRepository(self.store,self.training)
    def execute(self,folder,train,*,recover=False,pack=None,progress=None,lock=None,registry=None):
        argv=['runner','--directory',str(folder)]+(['--recover-interrupted'] if recover else [])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys,'argv',argv));stack.enter_context(patch.object(runner,'device_lock',return_value=contextlib.nullcontext()) if lock is None else patch.object(runner,'device_lock',side_effect=lock))
            stack.enter_context(patch.object(runner,'verify_lock'));stack.enter_context(patch.object(runner,'registry_path',return_value=registry or folder/'fixture-registry.json'))
            mock=stack.enter_context(patch.object(runner,'train',side_effect=train))
            if pack is not None:stack.enter_context(patch.object(runner,'pack_result',side_effect=pack))
            if progress is not None:stack.enter_context(patch.object(runner,'progress',side_effect=progress))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()));stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            result=runner.main()
        return result,mock

    def test_cpu_injected_oom_returns_classified_failure_not_available_model(self):
        job=self.job();self.executions.dispatch(self.pid,self.a,job['id']);folder=self.unpack(job)
        def fail(directory,request,report):report.update(training_executed=True);raise RuntimeError('CUDA out of memory. CPU injected fixture, no GPU used.')
        code,_=self.execute(folder,fail);self.assertEqual(code,2)
        with zipfile.ZipFile(folder/'return-result.zip') as archive:receipt=json.loads(archive.read('receipt.json'))
        self.assertEqual(receipt['report']['diagnostic']['code'],'CUDA_OUT_OF_MEMORY')
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['dispatch']['state'],'AWAITING_RESULT')
        self.executions.receive(self.pid,self.a,job['id'],(folder/'return-result.zip').read_bytes())
        record=self.training.get(self.pid,self.a,job['id']);self.assertEqual(record['state'],'FAILED');self.assertIsNone(record['model_version'])

    def test_pack_disk_failure_retains_reservation_then_cpu_recovery_closes_original_run(self):
        job=self.job();self.executions.dispatch(self.pid,self.a,job['id']);folder=self.unpack(job)
        def completed(directory,request,report):report.update(status='completed',training_executed=True,epochs_completed=1,batches_completed=13)
        result,_=self.execute(folder,completed,pack=OSError(errno.ENOSPC,'injected full disk'))
        self.assertEqual(result,2);self.assertFalse((folder/'return-result.zip').exists())
        progress=json.loads((folder/'progress.json').read_text(encoding='utf-8'));self.assertEqual(progress['diagnostic']['code'],'STORAGE_FULL')
        remote=RemoteRepository(self.store,self.training,self.executions)
        fake=SimpleNamespace(endpoint='root@fixture.invalid:22',call=Mock(return_value={'nvidia_smi':TARGET['verified_gpu_uuid']}),copy=Mock())
        with patch.object(remote,'transport',return_value=fake),patch.object(remote,'start_worker'):
            remote.submit(self.pid,self.a,job['id'])
            with self.store.connect() as db:row=dict(db.execute('SELECT s.*,? project_id,? task_id,d.package_sha256 FROM training_remote_sessions s JOIN training_dispatches d ON d.job_id=s.job_id WHERE s.job_id=?',(self.pid,self.a,job['id'])).fetchone())
            fake.call.return_value={'result_ready':False,'alive':False,'started':True,'directory_exists':True,'launch':{'pid':123},'progress':progress,'storage':{'checked':True,'free_bytes':0}}
            remote.reconcile(row)
            record=self.training.get(self.pid,self.a,job['id']);self.assertEqual(record['remote']['phase'],'WAITING_RESULT');self.assertEqual(record['remote']['progress']['error_code'],'STORAGE_FULL');self.assertEqual(record['dispatch']['state'],'AWAITING_RESULT')
            with self.assertRaises(ProjectError):self.training.change_visibility(self.pid,self.a,[{'id':job['id'],'confirmation_name':job['name'],'visibility_revision':0}])
            code,train=self.execute(folder,AssertionError('Recovery must never train'),recover=True);self.assertEqual(code,2);train.assert_not_called()
            raw=(folder/'return-result.zip').read_bytes();fake.call.return_value={'result_ready':True,'alive':False,'launch':{'pid':123},'resources':{'checked':True,'free':True},'result_sha256':hashlib.sha256(raw).hexdigest(),'progress':{}}
            fake.copy.side_effect=lambda path,*args:path.write_bytes(raw)
            remote.reconcile(row)
            final=self.training.get(self.pid,self.a,job['id']);self.assertEqual(final['state'],'INTERRUPTED');self.assertEqual(final['dispatch']['result']['report']['epochs_completed'],1)
        remote.close()

    def test_saved_result_is_preserved_if_progress_volume_fails_after_pack(self):
        job=self.job();self.executions.dispatch(self.pid,self.a,job['id']);folder=self.unpack(job)
        code,_=self.execute(folder,lambda d,q,r:r.update(status='completed'),progress=OSError(errno.ENOSPC,'metadata write failed'))
        self.assertEqual(code,0);self.assertTrue((folder/'return-result.zip').is_file())

    def test_other_directory_recovery_cannot_replace_saved_result_with_incomplete_ledger(self):
        job=self.job();self.executions.dispatch(self.pid,self.a,job['id']);folder=self.unpack(job)
        self.execute(folder,lambda d,q,r:r.update(status='completed'),progress=OSError(errno.ENOSPC,'metadata full'))
        original=(folder/'return-result.zip').read_bytes();registry=folder/'fixture-registry.json'
        self.assertNotIn('result_path',json.loads(registry.read_text(encoding='utf-8')))
        other=folder.parent/'another-directory';shutil.copytree(folder,other,ignore=shutil.ignore_patterns('return-result.zip'))
        code,train=self.execute(other,AssertionError('Never retrain'),recover=True,registry=registry)
        self.assertEqual(code,2);train.assert_not_called();self.assertFalse((other/'return-result.zip').exists());self.assertEqual((folder/'return-result.zip').read_bytes(),original)

    def test_duplicate_lock_failure_cannot_overwrite_original_progress(self):
        job=self.job();self.executions.dispatch(self.pid,self.a,job['id']);folder=self.unpack(job)
        original=b'{"stage":"training","epochs_completed":1}';(folder/'progress.json').write_bytes(original)
        code,train=self.execute(folder,AssertionError('Must not train'),lock=RuntimeError('existing lock held'))
        self.assertEqual(code,2);train.assert_not_called();self.assertEqual((folder/'progress.json').read_bytes(),original)

    def test_malformed_diagnostic_does_not_release_dispatch(self):
        job=self.job();self.executions.dispatch(self.pid,self.a,job['id'])
        raw=self.receipt(job,status='failed',change=lambda r:r['report'].update(error_code='STORAGE_FULL',diagnostic={'code':'CUDA_OUT_OF_MEMORY','title':'x','next_step':'x','phase':'training'}))
        with self.assertRaises(ProjectError):self.executions.receive(self.pid,self.a,job['id'],raw)
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['dispatch']['state'],'AWAITING_RESULT')
