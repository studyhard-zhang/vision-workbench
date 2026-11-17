import copy
import json
import uuid
from pathlib import Path
from unittest.mock import patch

from test_versions import VersionFixture
from app.training import TrainingRepository,TARGET
from app.executions import ExecutionRepository
from tools.training_config import OFFICIAL_CONFIG,OFFICIAL26,parameters,validate_snapshot
from tools.remote_environment import WEIGHTS
from app.storage import ProjectError
import test_executions as fixtures


class YOLO26Tests(VersionFixture):
    request=fixtures.ExecutionTests.request
    def setUp(self):
        super().setUp()
        self.vid=self.publish()['version']['id']
        self.training=TrainingRepository(self.store,self.versions)
        target={**TARGET,'id':'ssh-yolo26-fixture','managed_environment':True,
                'runtime':{**TARGET['runtime'],'ultralytics':'8.4.203'}}
        weights={name:{'id':name,'name':name,'path':'/remote/'+name,'sha256':sha} for name,sha in WEIGHTS.items()}
        target['weights']=weights['yolo26n.pt']
        record={'target':target,'weights':weights,'tests':{name:{'run_id':'fixture-'+name,'epochs':1,'reload':['best.pt','last.pt']} for name in WEIGHTS}}
        (self.path.parent/'active-compute.json').write_text(json.dumps(record),'utf-8')

    def test_both_modes_use_yolo26_and_dispatch_official_parameters(self):
        for mode,filename in [('standard','yolo26n.pt'),('high_accuracy','yolo26m.pt')]:
            target=self.training.mode_target('detect',mode)
            self.assertEqual(Path(target['weights']['path']).name,filename)
            result=self.training.create(self.pid,self.a,dict(name=mode,version_id=self.vid,target_id=target['id'],epochs=1,training_mode=mode,idempotency_key=uuid.uuid4().hex))
            job=result['job'];snap=job['snapshot']
            self.assertEqual(snap['configuration'],OFFICIAL_CONFIG)
            self.assertEqual(validate_snapshot(snap),{**OFFICIAL26,'epochs':1})
            self.assertEqual(snap['requested_parameters']['batch'],16)
            self.assertEqual(snap['requested_parameters']['mosaic'],1.)
            if mode=='standard':
                execution=ExecutionRepository(self.store,self.training)
                self.executions=execution
                dispatched=execution.dispatch(self.pid,self.a,job['id'])
                self.assertIsNotNone(dispatched)
                def resolved(receipt):
                    report=receipt['report']
                    report.update(weights_sha256=target['weights']['sha256'],geometry_check_passed=True,
                                  actual_optimizer='AdamW',actual_parameters={**snap['requested_parameters'],
                                  'device':'0','optimizer':'AdamW','lr0':.001667,'warmup_bias_lr':0.})
                raw=fixtures.ExecutionTests.receipt(self,job,change=resolved)
                execution.validate_result(raw,self.request(job))
            damaged=copy.deepcopy(snap);damaged['requested_parameters']['amp']=False
            with self.assertRaises(ValueError):validate_snapshot(damaged)
        for mode,filename in [('standard','yolo26n-seg.pt'),('high_accuracy','yolo26m-seg.pt')]:
            self.assertEqual(Path(self.training.mode_target('segment',mode)['weights']['path']).name,filename)

    def test_recommendation_stays_official_and_legacy_snapshot_still_validates(self):
        options=self.training.options(self.pid,self.a,self.vid)
        self.assertEqual(options['recommendation']['mode'],'official')
        self.assertEqual(options['default_epochs'],100)
        self.assertEqual(options['recommendation']['parameters'],parameters(OFFICIAL_CONFIG,100))
        self.assertTrue(next(p for p in options['configuration_profiles'] if p['id']=='official26')['enabled'])
        legacy={'requested_parameters':parameters(None,1),'target':TARGET}
        self.assertEqual(validate_snapshot(legacy),parameters(None,1))
