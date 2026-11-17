"""CPU contracts; the real 1824 AutoBatch evidence is kept outside test fixtures."""
import copy
import json
import uuid
from unittest.mock import patch
from test_versions import VersionFixture
import test_executions as fixtures
from app.training import TrainingRepository
from app.executions import ExecutionRepository
from app.models import ModelRepository
from app.evaluations import EvaluationRepository,EvaluationExecution
from app.inferences import InferenceInputRepository,InferenceRepository,InferenceExecution
from app.storage import ProjectError
from tools.training_config import PRECISION_CONFIG,configuration,parameters,validate_snapshot
from tools import run_evaluation_job,run_inference_job


class PrecisionTests(VersionFixture):
    dispatch=fixtures.ExecutionTests.dispatch
    request=fixtures.ExecutionTests.request
    receipt=fixtures.ExecutionTests.receipt

    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions);self.executions=ExecutionRepository(self.store,self.training)

    def job(self):
        return self.training.create(self.pid,self.a,{'name':'CPU precision contract','version_id':self.vid,'target_id':self.training.target()['id'],'epochs':1,'configuration':PRECISION_CONFIG,'idempotency_key':uuid.uuid4().hex},allow_validation=True)['job']

    def complete(self,job):
        def change(r):
            r['report'].update(geometry_check_passed=True,autobatch_estimation_passed=True,autobatch_memory_fraction=.6,resolved_batch_size=2,actual_parameters={**job['snapshot']['requested_parameters'],'batch':2})
        return self.receipt(job,change=change)

    def test_default_size_keeps_native_detail_and_manual_values_are_bounded(self):
        self.assertEqual(self.training.options()['default_configuration'],{'profile':'official26'})
        p=parameters(PRECISION_CONFIG,50);self.assertEqual((p['imgsz'],p['batch']),(1824,-1))
        validate_snapshot({'configuration':PRECISION_CONFIG,'requested_parameters':p})
        validate_snapshot({'configuration':PRECISION_CONFIG,'requested_parameters':p,'task_type':'segment'})
        for extra in ({'imgsz':1810},{'imgsz':True},{'imgsz':2080},{'batch':0},{'batch':True},{'batch':.6}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):configuration({**PRECISION_CONFIG,**extra})
        self.assertEqual(parameters({**PRECISION_CONFIG,'imgsz':1280,'batch':4},2)['batch'],4)

    def test_autobatch_receipt_requires_resolved_batch_and_estimation(self):
        job=self.job();self.dispatch(job)
        with self.assertRaises(ProjectError):self.executions.receive(self.pid,self.a,job['id'],self.receipt(job,change=lambda r:r['report'].update(geometry_check_passed=True)))
        raw=self.complete(job)
        import io,zipfile
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            receipt=json.loads(z.read('receipt.json'));files={n:z.read(n) for n in z.namelist() if n!='receipt.json'}
        from app.versions import encoded
        for bad in (True,0,-1):
            r=copy.deepcopy(receipt);r['report']['resolved_batch_size']=bad;out=io.BytesIO()
            with zipfile.ZipFile(out,'w') as z:
                z.writestr('receipt.json',encoded(r))
                for n,b in files.items():z.writestr(n,b)
            with self.subTest(bad=bad),self.assertRaises(ProjectError):self.executions.receive(self.pid,self.a,job['id'],out.getvalue())
        self.assertEqual(self.executions.receive(self.pid,self.a,job['id'],raw)['job']['state'],'SUCCEEDED')

    def check_model_size(self,kind):
        job=self.job();self.dispatch(job);self.executions.receive(self.pid,self.a,job['id'],self.complete(job));models=ModelRepository(self.store,self.executions);model=models.register(self.pid,self.a,job['id'])['model']
        evaluation=EvaluationRepository(self.store,models);eexec=EvaluationExecution(self.store,evaluation)
        inputs=InferenceInputRepository(self.store,self.tasks);inference=InferenceRepository(self.store,models,inputs);iexec=InferenceExecution(self.store,inference)
        frozen=self.versions.read(self.pid,self.a,self.vid)['manifest'];image=frozen['images'][0];item=inputs.from_dataset(self.pid,self.a,frozen['dataset_id'],image['id'])['input']
        self.assertEqual(evaluation.options(self.pid,self.a,model['id'])['defaults']['imgsz'],1824)
        self.assertEqual(inference.options(self.pid,self.a,model['id'])['defaults']['imgsz'],1824)
        e=evaluation.create(self.pid,self.a,{'model_id':model['id'],'subset':'val','idempotency_key':uuid.uuid4().hex})['job']
        i=inference.create(self.pid,self.a,{'model_id':model['id'],'input_ids':[item['id']],'idempotency_key':uuid.uuid4().hex})['job']
        if kind=='inference':evaluation.cancel(self.pid,self.a,e['id'])
        import zipfile
        for repo,executor,record,runner in (((evaluation,eexec,e,run_evaluation_job),) if kind=='evaluation' else ((inference,iexec,i,run_inference_job),)):
            self.assertEqual(record['snapshot']['requested_parameters']['imgsz'],1824);executor.dispatch(self.pid,self.a,record['id']);archive,_,_=executor.package(self.pid,self.a,record['id']);folder=self.path.parent/record['id'];folder.mkdir()
            with zipfile.ZipFile(archive) as z:z.extractall(folder)
            self.assertEqual(runner.check_input(folder)['snapshot']['requested_parameters']['imgsz'],1824)

    def test_evaluation_package_follows_model_size(self):self.check_model_size('evaluation')

    def test_inference_package_follows_model_size(self):self.check_model_size('inference')

    def test_precision_evidence_is_bound_to_task_type_and_gpu(self):
        target=self.training.target();root=self.path.parent/'compute-evidence';root.mkdir()
        proof={'verified':True,'run_id':'cpu-proof-format-fixture','gpu_uuid':target['verified_gpu_uuid'],'runtime':target['runtime'],'weights_sha256':target['weights']['sha256'],'requested_configuration':PRECISION_CONFIG}
        (root/'n21-segment.json').write_text(json.dumps(proof),'utf-8')
        self.assertFalse(self.training.capabilities()[0]['precision'])
        (root/'n21-detect.json').write_text(json.dumps(proof),'utf-8');self.assertTrue(self.training.capabilities()[0]['precision'])
        proof['gpu_uuid']='GPU-changed';(root/'n21-detect.json').write_text(json.dumps(proof),'utf-8');self.assertFalse(self.training.capabilities()[0]['precision'])
