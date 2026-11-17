import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch
from PIL import Image
import test_executions as fixtures
import test_evaluations as evaluation_fixtures
from test_segmentation import populate
from app.training import TrainingRepository
from app.executions import ExecutionRepository
from app.models import ModelRepository
from app.evaluations import EvaluationRepository,EvaluationExecution
from app.inferences import InferenceInputRepository,InferenceRepository,InferenceExecution
from app.prediction_masks import validate_segments
from app.storage import ProjectError


class SegmentValidationTests(unittest.TestCase):
    request=fixtures.ExecutionTests.request
    dispatch=fixtures.ExecutionTests.dispatch
    receipt=fixtures.ExecutionTests.receipt
    erequest=evaluation_fixtures.EvaluationTests.erequest
    eresult=evaluation_fixtures.EvaluationTests.eresult
    detail_report=evaluation_fixtures.EvaluationTests.detail_report
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store,ids=populate(self.tmp.name);self.ids=ids;self.path=self.store.path;self.pid=ids['project_id'];self.a=ids['task_id'];self.vid=ids['version_id'];self.training=TrainingRepository(self.store);self.versions=self.training.versions;target=self.training.target()
        proof={'verified':True,'runtime':target['runtime'],'gpu_uuid':target['verified_gpu_uuid'],'weights':{'path':'/fixture/yolo11n-seg.pt','sha256':'1'*64}}
        (self.path.parent/'segment-training.json').write_text(json.dumps(proof),'utf-8');self.executions=ExecutionRepository(self.store,self.training);self.models=ModelRepository(self.store,self.executions);self.executions.on_success=self.models.automatic
        job=self.training.create(self.pid,self.a,{'name':'seg fixture','version_id':self.vid,'target_id':target['id'],'epochs':1,'idempotency_key':uuid.uuid4().hex},allow_validation=True)['job'];self.dispatch(job)
        def change(receipt):receipt['report'].update(weights_sha256='1'*64,geometry_check_passed=True,mask_geometry_passed=True,actual_parameters={**job['snapshot']['requested_parameters'],'task':'segment'})
        self.model=self.executions.receive(self.pid,self.a,job['id'],self.receipt(job,change=change))['model'];self.evaluations=EvaluationRepository(self.store,self.models);self.eexec=EvaluationExecution(self.store,self.evaluations)
        self.inputs=InferenceInputRepository(self.store,self.training.versions.tasks);self.inferences=InferenceRepository(self.store,self.models,self.inputs);self.iexec=InferenceExecution(self.store,self.inferences)
    def tearDown(self):self.tmp.cleanup()
    def test_segment_evaluation_requires_mask_metrics_and_native_contours(self):
        job=self.evaluations.create(self.pid,self.a,{'model_id':self.model['id'],'subset':'val','idempotency_key':uuid.uuid4().hex})['job'];self.assertEqual(job['snapshot']['task_type'],'segment');self.eexec.dispatch(self.pid,self.a,job['id']);detail=self.detail_report(job)
        with self.assertRaises(ValueError):self.eexec.validate_report(detail,self.erequest(job))
        detail['metric_scope']='mask';image=detail['images'][0];pred={'class_index':0,'confidence':.9,'xyxy':[10,10,90,90],'segments':[{'points':[[10,10],[90,10],[90,90],[10,90]],'hole':False}]};image['predictions']=[pred];self.eexec.validate_report(detail,self.erequest(job));pred['segments'][0]['points'][0]=[-1,10]
        with self.assertRaises(ValueError):self.eexec.validate_report(detail,self.erequest(job))
    def test_mask_holes_and_complexity_are_bounded(self):
        validate_segments({'segments':[{'points':[[0,0],[10,0],[0,10]],'hole':False},{'points':[[1,1],[2,1],[1,2]],'hole':True}]},20,20)
        for segments in [None,[{'points':[[0,0],[10,0]],'hole':False}],[{'points':[[0,0],[10,0],[0,float('nan')]],'hole':False}]]:
            with self.assertRaises(ValueError):validate_segments({'segments':segments},20,20)
    def test_segment_inference_snapshot_and_report_cannot_silently_return_boxes(self):
        input=self.inputs.from_dataset(self.pid,self.a,self.ids['dataset_id'],self.ids['image_ids'][0])['input'];job=self.inferences.create(self.pid,self.a,{'model_id':self.model['id'],'input_ids':[input['id']],'idempotency_key':uuid.uuid4().hex})['job'];snap=job['snapshot'];self.assertEqual(snap['task_type'],'segment')
        row={'id':input['id'],'filename':input['filename'],'width':input['width'],'height':input['height'],'status':'SUCCEEDED','predictions':[{'class_index':0,'class_name':snap['classes'][0]['name'],'confidence':.8,'xyxy':[0,0,10,10]}]};data={'snapshot':snap,'runtime':'8.3.241','coordinate_system':'exif-oriented-pixels','complete':True,'images':[row]}
        with self.assertRaises(ValueError):self.iexec.validate_report(data,{'snapshot':snap},True)
        row['predictions'][0]['segments']=[];self.iexec.validate_report(data,{'snapshot':snap},True)

    def test_png_overlay_preserves_holes_in_original_image_pixels(self):
        image=self.inputs.from_dataset(self.pid,self.a,self.ids['dataset_id'],self.ids['image_ids'][0])['input']
        prediction={'class_index':0,'confidence':.8,'xyxy':[10,10,100,100],'segments':[{'points':[[10,10],[100,10],[100,100],[10,100]],'hole':False},{'points':[[30,30],[60,30],[60,60],[30,60]],'hole':True}]}
        report={'images':[{'id':image['id'],'status':'SUCCEEDED','predictions':[prediction]}]}
        path,_,_=self.inputs.file(self.pid,self.a,image['id'],'view')
        with Image.open(path) as original:outside=original.getpixel((20,20));hole=original.getpixel((40,40))
        with patch.object(self.iexec,'export',return_value=report):raw=self.iexec.overlay(self.pid,self.a,'unused',image['id'],.25)
        with Image.open(io.BytesIO(raw)) as overlay:
            self.assertEqual(overlay.size,(320,256));self.assertNotEqual(overlay.getpixel((20,20)),outside);self.assertEqual(overlay.getpixel((40,40)),hole)
