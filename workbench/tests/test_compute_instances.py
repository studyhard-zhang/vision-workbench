import uuid
from unittest.mock import patch

from app.compute_instances import ComputeInstances
from app.executions import ExecutionRepository
from app.remote import RemoteRepository, PasswordSSHTransport
from app.storage import ProjectError
from app.training import TrainingRepository, TARGET
from tools.remote_environment import WEIGHTS, DEPENDENCIES, ROOTS
from tools.training_config import PRECISION_CONFIG,OFFICIAL_CONFIG
from test_versions import VersionFixture


class ComputeInstanceTests(VersionFixture):
    def setUp(self):
        super().setUp()
        self.version = self.publish()['version']['id']
        self.training = TrainingRepository(self.store, self.versions)
        self.executions = ExecutionRepository(self.store,self.training)
        self.remote = RemoteRepository(self.store,self.training,self.executions)
        self.instances = ComputeInstances(self.remote)
        self.plan = dict(gpu={'uuid':'GPU-new-instance','name':'NVIDIA Test'}, python='3.12.3', executable='/usr/bin/python3',
                         root=ROOTS[1], free_bytes=20*1024**3, blockers=[], missing=[], missing_weights=list(WEIGHTS), recipe='cu121', reuse_torch=True)

    def verified_candidate(self):
        self.plan['missing_weights']=[]  # The user has prepared the weights externally.
        connection = dict(command='ssh -p 2233 root@example.org', password='fixture only', fingerprint='SHA256:'+'A'*43, confirmed_host=True)
        with patch('app.compute_instances.run_check', return_value=self.plan), patch('app.compute_instances.protect',return_value=b'encrypted'),patch.object(PasswordSSHTransport,'client'):
            draft = self.instances.connect(connection)['candidate']
        self.assertNotIn(connection['password'], self.instances.candidate.read_text('utf-8'))
        result = {'runtime':{**TARGET['runtime'],'ultralytics':'8.4.203'},'python':ROOTS[1]+'/environment-v1/venv/bin/python',
                  'tests':{name:{'run_id':'test-run','epochs':1,'reload':['best.pt','last.pt']} for name in WEIGHTS}}
        with patch('app.compute_instances.run_check',return_value=result):
            self.instances.prepare(draft['id'])
            self.instances.worker.join(5)
        self.assertEqual(self.instances.status()['candidate']['state'],'ready')
        self.assertEqual(self.training.target()['id'],TARGET['id'])
        return 'ssh-'+draft['id']

    def job(self, target, **extras):
        return self.training.create(self.pid,self.a,dict(name='queue',version_id=self.version,target_id=target,
            epochs=1,idempotency_key=uuid.uuid4().hex,**extras),allow_validation=target==TARGET['id'])['job']

    def activate(self, identifier, **changes):
        plan = {**self.plan,'packages':{**TARGET['runtime'],**DEPENDENCIES},'missing_weights':[],**changes}
        with patch('app.compute_instances.run_check',return_value=plan),patch.object(PasswordSSHTransport,'client'):
            return self.instances.activate(identifier)

    def test_new_instance_routes_new_jobs_and_keeps_previous_queue(self):
        old = self.job(TARGET['id'])
        identifier = self.verified_candidate()
        self.activate(identifier)
        self.assertEqual(self.training.target()['verified_gpu_uuid'],'GPU-new-instance')
        self.assertEqual(self.remote.transport().profile['workspace_root'], ROOTS[1])
        new = self.job(identifier, configuration=OFFICIAL_CONFIG)
        self.assertEqual(new['queue_position'],1)
        self.executions.dispatch(self.pid,self.a,new['id'])
        self.assertEqual(self.training.get(self.pid,self.a,old['id'])['snapshot'],old['snapshot'])

    def test_drift_and_active_dispatch_block_selection(self):
        identifier = self.verified_candidate()
        with self.assertRaises(ProjectError) as error:
            self.activate(identifier, missing_weights=['yolo11n.pt'])
        self.assertEqual(error.exception.code,'ENVIRONMENT_CHANGED')
        old = self.job(TARGET['id'])
        self.executions.dispatch(self.pid,self.a,old['id'])
        with self.assertRaises(ProjectError) as error:
            self.activate(identifier)
        self.assertEqual(error.exception.code,'SSH_CONNECTION_BUSY')
        self.assertEqual(self.training.target()['id'],TARGET['id'])

    def test_custom26_uses_verified_model_and_only_overrides_chosen_values(self):
        identifier=self.verified_candidate();self.activate(identifier)
        job=self.job(identifier,configuration={'profile':'custom26','imgsz':1024,'batch':2})
        p=job['snapshot']['requested_parameters']
        from tools.training_config import OFFICIAL26
        self.assertEqual(p,{**OFFICIAL26,'epochs':1,'imgsz':1024,'batch':2})
        self.assertEqual(job['snapshot']['configuration'],{'profile':'custom26','imgsz':1024,'batch':2})
        self.assertTrue(p['amp']);self.assertEqual(p['mosaic'],1.0)

    def test_fresh_local_registration_can_be_rechecked_without_managed_service_marker(self):
        self.plan['missing_weights']=[]
        values=dict(command='ssh -p 22226 root@127.0.0.1',password='fixture',fingerprint='SHA256:'+'A'*43,confirmed_host=True,connection_kind='local')
        with patch('app.compute_instances.run_check',return_value=self.plan),patch('app.compute_instances.protect',return_value=b'encrypted'),patch.object(PasswordSSHTransport,'client'):
            candidate=self.instances.connect(values)['candidate']
        result={'validation_kind':'environment','ready':True,'python':'/opt/existing/bin/python','runtime':{**TARGET['runtime'],'ultralytics':'8.4.203'},'tests':{'yolo26n.pt':{'run_id':'env-fixture','environment_checked':True,'forward_passed':True,'training_executed':False,'weights_sha256':WEIGHTS['yolo26n.pt']}}}
        with patch('app.compute_instances.run_check',return_value=result):
            self.instances.prepare(candidate['id']);self.instances.worker.join(5)
        self.assertEqual(self.instances.status()['candidate']['state'],'ready')
        with patch('app.compute_instances.run_check',return_value=self.plan),patch.object(PasswordSSHTransport,'client'):
            self.assertEqual(self.instances.connect_local({})['candidate']['state'],'checked')
