import base64
import contextlib
import hashlib
import io
import json
import sys
import threading
import uuid
import unittest
import zipfile
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from PIL import Image

from test_versions import VersionFixture
import test_executions as fixtures
from app.storage import ProjectError
from app.training import TrainingRepository,TARGET,PARAMETERS
from app.executions import ExecutionRepository
from app.augmentation import AugmentationRepository
from app.versions import encoded
from app.server import create_server
from tools.training_config import configuration,parameters,validate_snapshot,geometry
from tools import run_training_job as runner


class ConfigurationContractTests(unittest.TestCase):
    def test_legacy_configuration_is_byte_semantically_unchanged(self):
        self.assertEqual(parameters(None,50),{**PARAMETERS,'epochs':50})
        validate_snapshot({'requested_parameters':parameters(None,1)})

    def test_contract_fixes_compute_and_disables_unpreviewed_geometry(self):
        cfg=configuration({'profile':'geometry512','optimizer':'AdamW','lr0':.001,'momentum':.9})
        p=parameters(cfg,1);self.assertEqual((p['imgsz'],p['batch'],p['amp']),(512,2,False))
        for key in ('mosaic','scale','translate','degrees','hsv_h'):self.assertEqual(p[key],0)
        validate_snapshot({'configuration':cfg,'requested_parameters':p})
        for k,v in [('imgsz',1024),('device','cpu'),('amp',True)]:
            with self.assertRaises(ValueError):validate_snapshot({'configuration':cfg,'requested_parameters':{**p,k:v}})

    def test_invalid_fields_nan_bool_and_auto_overrides_are_rejected(self):
        for extra in [{'profile':{}},{'fliplr':True},{'flipud':-1},{'fliplr':float('nan')},{'seed':True},
                      {'patience':0},{'optimizer':'SGD'},{'optimizer':'auto','lr0':.1},
                      {'optimizer':'AdamW','lr0':0},{'optimizer':'AdamW','momentum':1},{'device':'cpu'}]:
            with self.subTest(extra=extra),self.assertRaises(ValueError):configuration({'profile':'geometry640',**extra})

    def test_non_square_resize_padding_and_double_flip_match_known_coordinates(self):
        boxes=[{'class_id':'a','x':10,'y':5,'width':20,'height':10}]
        a=geometry(100,50,640,boxes);b=geometry(100,50,640,boxes,True,True)
        self.assertEqual((a['resized_width'],a['resized_height'],a['left'],a['top']),(640,320,0,160))
        self.assertEqual((a['boxes'][0]['x'],a['boxes'][0]['y']),(64,192))
        self.assertEqual((b['boxes'][0]['x'],b['boxes'][0]['y']),(448,384))
        self.assertEqual(boxes[0]['x'],10)
        odd=geometry(17,31,512,[]);self.assertEqual(odd['resized_width'],281);self.assertEqual(odd['left'],115)


class ConfigurationTests(VersionFixture):
    job=fixtures.ExecutionTests.job;dispatch=fixtures.ExecutionTests.dispatch;request=fixtures.ExecutionTests.request;receipt=fixtures.ExecutionTests.receipt

    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions)
        self.executions=ExecutionRepository(self.store,self.training);self.preview=AugmentationRepository(self.versions)

    def body(self,**cfg):return {'name':'CPU-only configuration fixture','version_id':self.vid,'target_id':TARGET['id'],'epochs':1,'idempotency_key':uuid.uuid4().hex,'configuration':{'profile':'geometry640',**cfg}}

    def test_unverified_public_config_is_blocked_and_internal_validation_is_idempotent(self):
        body=self.body()
        with self.assertRaises(ProjectError) as e:self.training.create(self.pid,self.a,body)
        self.assertEqual(e.exception.code,'CONFIGURATION_NOT_VERIFIED');self.assertEqual(self.training.list(self.pid,self.a),[])
        a=self.training.create(self.pid,self.a,body,allow_validation=True);self.assertTrue(a['job']['snapshot']['validation_only'])
        self.assertTrue(self.training.create(self.pid,self.a,body)['reused'])
        with self.assertRaises(ProjectError):self.training.create(self.pid,self.a,{**body,'configuration':{'profile':'geometry512'}})

    def test_new_gpu_cannot_reuse_old_gpu_configuration_evidence_or_old_queued_job(self):
        job=self.training.create(self.pid,self.a,self.body(),allow_validation=True)['job'];self.dispatch(job)
        self.executions.receive(self.pid,self.a,job['id'],self.receipt(job,change=lambda r:r['report'].update(geometry_check_passed=True)))
        self.assertTrue(next(p for p in self.training.options()['configuration_profiles'] if p['id']=='geometry640')['enabled'])
        queued=self.job();target={**TARGET,'id':'autodl-new-gpu','verified_gpu_uuid':'GPU-new-identity','enabled':False}
        (self.path.parent/'compute-target.json').write_text(json.dumps(target),encoding='utf-8')
        options=self.training.options();self.assertFalse(any(p['enabled'] for p in options['configuration_profiles']))
        with self.assertRaises(ProjectError) as e:self.executions.dispatch(self.pid,self.a,queued['id'])
        self.assertEqual(e.exception.code,'SSH_RUN_TARGET_MISMATCH')
        body={**self.body(),'target_id':target['id']}
        with self.assertRaises(ProjectError) as e:self.training.create(self.pid,self.a,body)
        self.assertEqual(e.exception.code,'COMPUTE_TARGET_NOT_VALIDATED')
        validation=self.training.create(self.pid,self.a,body,allow_validation=True)['job'];self.assertEqual(validation['snapshot']['target']['verified_gpu_uuid'],'GPU-new-identity')
        self.training.cancel(self.pid,self.a,queued['id']);self.dispatch(validation)
        self.assertEqual(self.request(validation)['expected_gpu_uuid'],'GPU-new-identity')

    def test_pending_target_preserves_idempotent_old_submission_but_refuses_new_jobs(self):
        body=self.body();job=self.training.create(self.pid,self.a,body,allow_validation=True)['job']
        (self.path.parent/'ssh-target-change.json').write_text('{}',encoding='utf-8')
        self.assertEqual(self.training.create(self.pid,self.a,body)['job']['id'],job['id'])
        with self.assertRaises(ProjectError) as e:self.training.create(self.pid,self.a,self.body(),allow_validation=True)
        self.assertEqual(e.exception.code,'SSH_TARGET_CHANGE_PENDING')

    def test_malformed_target_is_rejected_without_creating_job(self):
        (self.path.parent/'compute-target.json').write_text('{"id":"unverified"}',encoding='utf-8')
        with self.assertRaises(ProjectError) as e:self.training.options()
        self.assertEqual(e.exception.code,'COMPUTE_TARGET_INVALID');self.assertEqual(self.training.list(self.pid,self.a),[])

    def test_geometry_validation_does_not_unlock_unverified_legacy_augmentation_on_new_gpu(self):
        target={**TARGET,'id':'autodl-new-gpu','verified_gpu_uuid':'GPU-new-identity','enabled':True}
        (self.path.parent/'compute-target.json').write_text(json.dumps(target),encoding='utf-8')
        body={**self.body(),'target_id':target['id']};body.pop('configuration')
        self.assertFalse(self.training.options()['legacy_enabled'])
        with self.assertRaises(ProjectError) as e:self.training.create(self.pid,self.a,body)
        self.assertEqual(e.exception.code,'CONFIGURATION_NOT_VERIFIED')
        validation=self.training.create(self.pid,self.a,body,allow_validation=True)['job']
        self.assertEqual(validation['snapshot']['target']['id'],target['id'])

    def test_historical_snapshot_without_gpu_uuid_stays_readable_and_cannot_unlock_config(self):
        job=self.job();self.dispatch(job);self.executions.receive(self.pid,self.a,job['id'],self.receipt(job))
        snapshot=job['snapshot'];snapshot['target'].pop('verified_gpu_uuid')
        with self.store.connect() as db:
            historical=dict(db.execute('SELECT * FROM training_jobs WHERE id=?',(job['id'],)).fetchone())
            historical.update(id=uuid.uuid4().hex,sequence=historical['sequence']+1,idempotency_key=uuid.uuid4().hex,snapshot_json=json.dumps(snapshot))
            db.execute('INSERT INTO training_jobs ('+','.join(historical)+') VALUES ('+','.join('?' for _ in historical)+')',tuple(historical.values()))
            dispatch=dict(db.execute('SELECT * FROM training_dispatches WHERE job_id=?',(job['id'],)).fetchone());dispatch.update(job_id=historical['id'],id=uuid.uuid4().hex)
            db.execute('INSERT INTO training_dispatches ('+','.join(dispatch)+') VALUES ('+','.join('?' for _ in dispatch)+')',tuple(dispatch.values()))
        self.assertFalse(any(p['enabled'] for p in self.training.options()['configuration_profiles']))
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'SUCCEEDED')

    def test_new_execution_package_checks_shared_contract_without_importing_training_runtime(self):
        job=self.training.create(self.pid,self.a,self.body(profile='geometry512',optimizer='AdamW'),allow_validation=True)['job'];self.dispatch(job)
        path,_,_=self.executions.package(self.pid,self.a,job['id']);folder=self.path.parent/'remote';folder.mkdir()
        with zipfile.ZipFile(path) as z:z.extractall(folder)
        original=__import__
        def guard(name,*args,**kwargs):
            if name.startswith(('torch','ultralytics','numpy')):raise AssertionError('training import on controller')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',guard),patch.object(sys,'argv',['run_training_job.py','--directory',str(folder),'--prepare-only']),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(runner.main(),0)
        self.assertFalse((folder/'started.json').exists());self.assertFalse((folder/'return-result.zip').exists())

    def test_result_needs_geometry_parity_and_exact_parameters_before_unlocking(self):
        job=self.training.create(self.pid,self.a,self.body(),allow_validation=True)['job'];self.dispatch(job)
        raw=self.receipt(job)
        with self.assertRaises(ProjectError):self.executions.receive(self.pid,self.a,job['id'],raw)
        raw=self.receipt(job,change=lambda r:r['report'].update(geometry_check_passed=True))
        self.executions.receive(self.pid,self.a,job['id'],raw)
        options=self.training.options();self.assertTrue(next(p for p in options['configuration_profiles'] if p['id']=='geometry640')['enabled'])
        self.assertFalse(next(p for p in options['configuration_profiles'] if p['id']=='geometry512')['enabled'])
        self.assertEqual(options['verified_optimizers'],['auto'])
        queued=self.training.create(self.pid,self.a,self.body())['job'];self.assertEqual(queued['state'],'QUEUED')
        self.training.cancel(self.pid,self.a,queued['id'])
        with self.assertRaises(ProjectError):self.training.create(self.pid,self.a,self.body(optimizer='AdamW'))
        other=self.training.create(self.pid,self.a,self.body(profile='geometry512',optimizer='AdamW'),allow_validation=True)['job'];self.dispatch(other)
        raw=self.receipt(other,change=lambda r:r['report'].update(geometry_check_passed=True,actual_optimizer='AdamW'))
        self.executions.receive(self.pid,self.a,other['id'],raw)
        # Passing two pairs must not authorize their untested cross combinations.
        with self.assertRaises(ProjectError):self.training.create(self.pid,self.a,self.body(optimizer='AdamW'))
        with self.assertRaises(ProjectError):self.training.create(self.pid,self.a,self.body(profile='geometry512'))

    def test_preview_uses_frozen_labels_and_never_flips_validation_members(self):
        frozen=self.versions.read(self.pid,self.a,self.vid)['manifest'];before=json.dumps(frozen,sort_keys=True)
        image=next(i for i in frozen['images'] if i['subset']=='train')
        body={'version_id':self.vid,'image_id':image['id'],'configuration':{'profile':'geometry640','fliplr':1,'flipud':1}}
        a=self.preview.preview(self.pid,self.a,body);self.assertTrue(a['geometry']['horizontal']);self.assertTrue(a['geometry']['vertical'])
        self.assertEqual(a,self.preview.preview(self.pid,self.a,body))
        with Image.open(io.BytesIO(base64.b64decode(a['image_data'].split(',')[1]))) as im:self.assertEqual(im.size,(640,640))
        self.repo.change_class(self.pid,self.a,{'name':'工作区变更','revision':1},self.cids[0])
        self.assertEqual(a,self.preview.preview(self.pid,self.a,body))
        image=next(i for i in frozen['images'] if i['subset']=='val');b=self.preview.preview(self.pid,self.a,{**body,'image_id':image['id']})
        self.assertFalse(b['geometry']['horizontal']);self.assertFalse(b['geometry']['vertical'])
        self.assertEqual(json.dumps(self.versions.read(self.pid,self.a,self.vid)['manifest'],sort_keys=True),before)
        self.assertEqual(self.training.list(self.pid,self.a),[])

    def test_http_preview_scope_and_unverified_submission_cannot_be_bypassed(self):
        image=self.versions.read(self.pid,self.a,self.vid)['manifest']['images'][0]
        server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();origin=f'http://127.0.0.1:{server.server_port}'
        base=f'/api/projects/{self.pid}/tasks/{self.a}'
        def call(path,body):return urlopen(Request(origin+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}))
        body={'version_id':self.vid,'image_id':image['id'],'configuration':{'profile':'geometry640'}}
        try:
            with call(base+'/training-preview',body) as r:self.assertEqual(json.load(r)['version_id'],self.vid)
            with self.assertRaises(HTTPError) as e:call(base.replace(self.a,self.b)+'/training-preview',body)
            self.assertEqual(e.exception.code,404)
            with self.assertRaises(HTTPError) as e:call(base+'/training',{**self.body(),'allow_validation':True})
            self.assertEqual(e.exception.code,422)
            with self.assertRaises(HTTPError) as e:call(base+'/training',self.body())
            self.assertEqual(e.exception.code,409)
        finally:server.shutdown();server.server_close();thread.join()
