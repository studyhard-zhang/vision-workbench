import hashlib
import uuid
import zipfile
import copy
from unittest.mock import patch
import test_models as model_fixtures
from test_versions import VersionFixture
from app.storage import ProjectError
from app.training import TARGET


class ModelIterationTests(VersionFixture):
    job = model_fixtures.ModelTests.job
    dispatch = model_fixtures.ModelTests.dispatch
    request = model_fixtures.ModelTests.request
    receipt = model_fixtures.ModelTests.receipt
    success = model_fixtures.ModelTests.success
    register = model_fixtures.ModelTests.register

    def setUp(self):
        super().setUp()
        from app.training import TrainingRepository
        from app.executions import ExecutionRepository
        from app.models import ModelRepository
        self.vid = self.publish()['version']['id']
        self.training = TrainingRepository(self.store, self.versions)
        self.executions = ExecutionRepository(self.store, self.training)
        self.models = ModelRepository(self.store, self.executions)

    def iteration(self, model, **overrides):
        values = dict(name='Next model', version_id=self.vid, target_id=TARGET['id'],
                      epochs=1, idempotency_key=uuid.uuid4().hex, source_model_id=model['id'])
        values.update(overrides)
        return self.training.create(self.pid, self.a, values, allow_validation=True)['job']

    def test_chain_packages_parent_best_and_records_lineage(self):
        parent = self.register(self.success())['model']
        for number in (2, 3):
            job = self.iteration(parent)
            self.assertEqual(job['snapshot']['initial_model']['id'], parent['id'])
            self.dispatch(job)
            package, _, _ = self.executions.package(self.pid, self.a, job['id'])
            with zipfile.ZipFile(package) as archive:
                self.assertEqual(hashlib.sha256(archive.read('initial.pt')).hexdigest(), parent['manifest']['artifacts']['best.pt']['sha256'])
            initial = job['snapshot']['initial_model']
            raw = self.receipt(job, change=lambda r: r['report'].update(weights_sha256=initial['weights_sha256'], initial_model=initial))
            self.executions.receive(self.pid, self.a, job['id'], raw)
            parent = self.register(job)['model']
            self.assertEqual(parent['number'], number)
            self.assertEqual(parent['manifest']['training_snapshot']['initial_model'], initial)

    def test_missing_or_cross_task_model_is_rejected(self):
        with self.assertRaises(ProjectError):self.iteration({'id':uuid.uuid4().hex})

    def test_changed_classes_are_rejected(self):
        parent = self.register(self.success())['model']
        self.repo.change_class(self.pid, self.a, {'name':'changed','revision':1}, self.cids[0])
        self.vid = self.publish(self.generate(seed=99))['version']['id']
        with self.assertRaises(ProjectError) as failure:self.iteration(parent)
        self.assertEqual(failure.exception.code, 'SOURCE_MODEL_CLASSES')

    def test_damaged_parent_blocks_dispatch(self):
        parent = self.register(self.success())['model'];job = self.iteration(parent)
        self.models.owned(parent['id'], 'best.pt').write_bytes(b'tampered')
        with self.assertRaises(ProjectError):self.dispatch(job)

    def test_parent_architecture_survives_changed_default_model(self):
        parent = self.register(self.success())['model']
        target = copy.deepcopy(TARGET)
        target['weights'].update(path='/root/models/yolo26n.pt',sha256='8'*64)
        with patch.object(self.training,'mode_target',return_value=target):
            job = self.iteration(parent)
        self.assertEqual(job['snapshot']['initial_model']['architecture'],'yolo11n.pt')

    def test_worker_refuses_changed_initial_weights(self):
        from tools.run_training_job import check_input
        parent = self.register(self.success())['model'];job = self.iteration(parent);self.dispatch(job)
        package,_,_ = self.executions.package(self.pid,self.a,job['id'])
        folder = self.path.parent/'worker';folder.mkdir()
        with zipfile.ZipFile(package) as archive:archive.extractall(folder)
        check_input(folder)
        (folder/'initial.pt').write_bytes(b'changed during transport')
        with self.assertRaises(ValueError):check_input(folder)
