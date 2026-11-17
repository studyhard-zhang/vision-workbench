import copy
import json
import uuid
from unittest.mock import patch
from test_versions import VersionFixture
import test_executions as fixtures
from app.training import TrainingRepository
from app.executions import ExecutionRepository
from app.training_recommendations import recommend
from app.training_diagnosis import diagnose
from app.storage import ProjectError


class RecommendationTests(VersionFixture):
    dispatch=fixtures.ExecutionTests.dispatch
    request=fixtures.ExecutionTests.request
    receipt=fixtures.ExecutionTests.receipt

    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions);self.executions=ExecutionRepository(self.store,self.training)

    def completed(self,map95=.7,precision=.95,validation=False):
        job=self.training.create(self.pid,self.a,{'name':'CPU historical candidate','version_id':self.vid,'target_id':self.training.target()['id'],'epochs':50,'idempotency_key':uuid.uuid4().hex},allow_validation=validation)['job'];self.dispatch(job)
        metrics={'metrics/mAP50-95(B)':map95,'metrics/mAP50(B)':.9,'metrics/precision(B)':precision,'metrics/recall(B)':1.}
        self.executions.receive(self.pid,self.a,job['id'],self.receipt(job,change=lambda r:r['report'].update(metrics=metrics,epochs_completed=50)))
        return self.training.get(self.pid,self.a,job['id'])

    def test_history_prefers_recorded_quality_and_keeps_complete_parameters(self):
        good=self.completed();self.completed(.2,.03);result=recommend(self.training,self.pid,self.a,self.vid)
        self.assertEqual(result['mode'],'official');self.assertIsNone(result['source']);self.assertEqual(result['configuration'],{'profile':'official26'});self.assertEqual(result['epochs'],100)

    def test_qa_validation_and_trash_cannot_set_recommendation(self):
        self.completed(validation=True);self.assertEqual(recommend(self.training,self.pid,self.a,self.vid)['mode'],'official')
        good=self.completed();self.training.change_visibility(self.pid,self.a,[{'id':good['id'],'confirmation_name':good['name'],'visibility_revision':0}])
        self.assertEqual(recommend(self.training,self.pid,self.a,self.vid)['mode'],'official')

    def test_history_is_scoped_to_same_task_and_version(self):
        good=self.completed()
        other=self.publish(self.generate(seed=123))['version']['id']
        self.assertNotEqual(other,self.vid)
        self.assertEqual(recommend(self.training,self.pid,self.a,other)['mode'],'official')

    def test_unseen_data_uses_object_size_and_sample_count_without_gpu(self):
        original=self.versions.read(self.pid,self.a,self.vid);data=copy.deepcopy(original)
        for image in data['manifest']['images']:
            image.update(width=1800,height=1800)
            for box in image['boxes']:box.update(width=10,height=10)
        with patch.object(self.versions,'read',return_value=data):small=recommend(self.training,self.pid,self.a,self.vid)
        for image in data['manifest']['images']:
            for box in image['boxes']:box.update(width=500,height=500)
        with patch.object(self.versions,'read',return_value=data):large=recommend(self.training,self.pid,self.a,self.vid)
        self.assertEqual(small['parameters']['imgsz'],640);self.assertEqual(large['parameters']['imgsz'],640);self.assertEqual(large['parameters']['batch'],16);self.assertEqual(large['epochs'],100);self.assertEqual(self.training.list(self.pid,self.a),[])

    def test_damaged_or_other_task_version_is_rejected(self):
        data=self.versions.read(self.pid,self.a,self.vid);data['integrity']['ok']=False
        with patch.object(self.versions,'read',return_value=data),self.assertRaises(ProjectError):recommend(self.training,self.pid,self.a,self.vid)
        with self.assertRaises(ProjectError):recommend(self.training,self.pid,self.b,self.vid)

    def test_diagnosis_preserves_metric_scope_and_actual_batch(self):
        job=self.completed(.4,.03);job['snapshot']['requested_parameters'].update(batch=-1,imgsz=1824);job['dispatch']['result']['report']['resolved_batch_size']=1
        rows=[{'epoch':1,'metrics/precision(B)':.9,'metrics/recall(B)':1.,'metrics/mAP50(B)':.99},{'epoch':50,'metrics/precision(B)':.03,'metrics/recall(B)':1.,'metrics/mAP50(B)':.49}]
        result=diagnose(job,rows);text='\n'.join(result['notes']);self.assertIn('实际批大小为1',text);self.assertIn('误检',text);self.assertIn('轮间波动',text);self.assertIn('不是独立测试',text);self.assertEqual(result['checkpoint_metrics']['precision'],.03)
