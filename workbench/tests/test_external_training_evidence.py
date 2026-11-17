"""External QA evidence must remain bound to task kind, GPU and lifecycle."""
import copy
import json
from test_versions import VersionFixture
from app.training import TrainingRepository
from tools.training_config import SEGMENT_CONFIG,parameters

class ExternalEvidenceTests(VersionFixture):
    def setUp(self):
        super().setUp();self.training=TrainingRepository(self.store,self.versions);target=self.training.target()
        self.path=self.store.path.parent/'compute-evidence/training-validation.json';self.path.parent.mkdir(exist_ok=True)
        self.proof={'verified':True,'task_type':'detect','run_id':'isolated-run','target':target,'configuration':SEGMENT_CONFIG,'remote_process_exited':True,'resources':{'checked':True,'free':True},
            'report':{'status':'completed','training_executed':True,'forward_passed':True,'geometry_check_passed':True,'weights_sha256':target['weights']['sha256'],'requested_parameters':parameters(SEGMENT_CONFIG,1),'environment':{'ready':True,'nvidia_smi':target['verified_gpu_uuid']},'reload_passed':{'best.pt':True,'last.pt':True}}}
    def store_proof(self,p):self.path.write_text(json.dumps([p]),'utf-8')
    def test_isolated_geometry_proof_unlocks_only_verified_profile_and_optimizer(self):
        self.store_proof(self.proof);passed,opts=self.training.capabilities()
        self.assertEqual(passed['geometry640'],['isolated-run']);self.assertEqual(opts['geometry640'],{'auto'});self.assertFalse(passed['geometry512']);self.assertFalse(passed['precision'])
    def test_other_gpu_runtime_weights_and_task_kind_cannot_unlock(self):
        for path,value in [(('target','verified_gpu_uuid'),'GPU-other'),(('target','runtime'),{}),(('target','weights','sha256'),'0'*64),(('task_type',),'segment')]:
            p=copy.deepcopy(self.proof);d=p
            for part in path[:-1]:d=d[part]
            d[path[-1]]=value;self.store_proof(p);self.assertFalse(self.training.capabilities()[0]['geometry640'])
    def test_missing_reload_resource_and_execution_proof_cannot_unlock(self):
        for path in [('remote_process_exited',),('resources','free'),('resources','checked'),('report','forward_passed'),('report','geometry_check_passed'),('report','reload_passed','best.pt'),('report','environment','ready')]:
            p=copy.deepcopy(self.proof);d=p
            for part in path[:-1]:d=d[part]
            d[path[-1]]=False;self.store_proof(p);self.assertFalse(self.training.capabilities()[0]['geometry640'])
    def test_malformed_evidence_does_not_unlock(self):
        self.path.write_text('{broken','utf-8');self.assertFalse(self.training.capabilities()[0]['geometry640'])
