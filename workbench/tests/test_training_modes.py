import json,uuid
from test_versions import VersionFixture
from app.training import TrainingRepository,TARGET
from app.storage import ProjectError
from tools.training_config import parameters

class TrainingModeTests(VersionFixture):
    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions)
        self.cfg={'profile':'precision','imgsz':640,'batch':-1,'fliplr':0.,'flipud':0.,'seed':42,'patience':100,'optimizer':'auto'}
        self.weight={'id':'qa-medium','name':'DetectDL','path':'/remote/yolo11m.pt','sha256':'c'*64}
    def settings(self,verified=True):
        value={'detect':{'weights':self.weight,'runtime':TARGET['runtime'],'gpu_uuid':TARGET['verified_gpu_uuid'],'verified':verified}}
        (self.path.parent/'high-accuracy-training.json').write_text(json.dumps(value),'utf-8')
    def body(self,**changes):return {'name':'高精度训练','version_id':self.vid,'target_id':TARGET['id'],'epochs':1,'idempotency_key':uuid.uuid4().hex,'training_mode':'high_accuracy','execution_method':'ssh','configuration':self.cfg,**changes}
    def proof(self,weight=None,gpu=None):
        weight=weight or self.weight;gpu=gpu or TARGET['verified_gpu_uuid'];target={**TARGET,'weights':weight,'verified_gpu_uuid':gpu}
        proof={'verified':True,'task_type':'detect','run_id':'isolated-medium','target':target,'configuration':self.cfg,'report':{'status':'completed','training_executed':True,'forward_passed':True,'geometry_check_passed':True,'weights_sha256':weight['sha256'],'requested_parameters':parameters(self.cfg,1),'environment':{'ready':True,'nvidia_smi':gpu},'autobatch_estimation_passed':True,'resolved_batch_size':2,'reload_passed':{'best.pt':True,'last.pt':True}},'remote_process_exited':True,'resources':{'checked':True,'free':True}}
        folder=self.path.parent/'compute-evidence';folder.mkdir(exist_ok=True);(folder/'training-validation.json').write_text(json.dumps([proof]),'utf-8')
    def test_no_medium_validation_never_inherits_nano_capability(self):
        self.settings();self.proof(TARGET['weights'])
        with self.assertRaises(ProjectError) as error:self.training.create(self.pid,self.a,self.body())
        self.assertEqual(error.exception.code,'CONFIGURATION_NOT_VERIFIED');self.assertFalse(self.training.capabilities('detect','high_accuracy')[0]['precision'])
    def test_verified_medium_snapshot_preserves_model_mode_and_manual_method(self):
        self.settings();self.proof();body=self.body(execution_method='manual');job=self.training.create(self.pid,self.a,body)['job']
        self.assertEqual(job['snapshot']['training_mode'],'high_accuracy');self.assertEqual(job['snapshot']['execution_method'],'manual');self.assertEqual(job['snapshot']['target']['weights'],self.weight);self.assertEqual(job['state'],'QUEUED')
        self.assertEqual(self.training.create(self.pid,self.a,body)['job']['id'],job['id']);self.assertEqual(self.training.options(self.pid,self.a,self.vid,'high_accuracy')['model_name'],'DetectDL')
    def test_new_gpu_and_unverified_weights_block_high_mode(self):
        self.settings(False)
        with self.assertRaises(ProjectError) as error:self.training.create(self.pid,self.a,self.body())
        self.assertEqual(error.exception.code,'HIGH_ACCURACY_NOT_VERIFIED')
        self.settings();self.proof(gpu='GPU-other');self.assertFalse(self.training.capabilities('detect','high_accuracy')[0]['precision'])
    def test_old_request_defaults_to_standard_and_remains_compatible(self):
        body=self.body();body.pop('training_mode');body.pop('configuration');job=self.training.create(self.pid,self.a,body)['job']
        self.assertEqual(job['snapshot']['training_mode'],'standard');self.assertEqual(job['snapshot']['target']['weights'],TARGET['weights'])
