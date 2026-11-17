"""Export boundary tests; fixture .pt bytes are not real model checkpoints."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from urllib.request import urlopen
from urllib.error import HTTPError
from unittest.mock import Mock
import zipfile

import test_models as fixtures
from test_versions import VersionFixture
from app.delivery import DeliveryRepository
from app.storage import ProjectError
from app.models import ModelRepository
from app.training import TrainingRepository
from app.executions import ExecutionRepository
from app.server import create_server


class DeliveryTests(VersionFixture):
    job = fixtures.ModelTests.job
    dispatch = fixtures.ModelTests.dispatch
    request = fixtures.ModelTests.request
    receipt = fixtures.ModelTests.receipt
    success = fixtures.ModelTests.success
    register = fixtures.ModelTests.register

    def setUp(self):
        super().setUp()
        self.vid = self.publish()['version']['id']
        self.training = TrainingRepository(self.store, self.versions)
        self.executions = ExecutionRepository(self.store, self.training)
        self.models = ModelRepository(self.store, self.executions)

    def package(self):
        model = self.register(self.success())['model']
        exports = DeliveryRepository(self.models, Mock())
        return model, exports, exports.build(self.pid, self.a, model['id'])

    def unpack(self, raw, directory):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            archive.extractall(directory)

    def test_allowlist_determinism_source_privacy_and_checksums(self):
        model, exports, (raw, name) = self.package()
        self.assertEqual(raw, exports.build(self.pid, self.a, model['id'])[0])
        self.assertIn('M001', name)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertEqual(set(archive.namelist()), {'best.pt', 'classes.json', 'inference_config.json',
                'metrics.json', 'source.json', 'environment.json', 'requirements.txt', 'predict.py', 'README.md', 'checksums.json', 'validation.json'})
            checks = json.loads(archive.read('checksums.json'))
            for key, spec in checks['files'].items():
                content = archive.read(key)
                self.assertEqual(len(content), spec['bytes'])
                self.assertEqual(hashlib.sha256(content).hexdigest(), spec['sha256'])
            self.assertNotIn('target', json.loads(archive.read('source.json')))
            self.assertEqual(json.loads(archive.read('metrics.json'))['status'], 'NOT_EVALUATED')
            self.assertEqual(archive.read('best.pt'), self.models.owned(model['id'], 'best.pt').read_bytes())

    def test_standard_library_check_and_corrupt_missing_extra_rejection(self):
        _, _, (raw, _) = self.package()
        with tempfile.TemporaryDirectory() as directory:
            self.unpack(raw, directory)
            path = Path(directory)
            def run():
                return subprocess.run([sys.executable, '-I', str(path/'predict.py'), '--check-only'], capture_output=True, timeout=15)
            self.assertEqual(run().returncode, 0)
            original = (path/'best.pt').read_bytes()
            (path/'best.pt').write_bytes(b'corrupt')
            self.assertNotEqual(run().returncode, 0)
            (path/'best.pt').unlink()
            self.assertNotEqual(run().returncode, 0)
            (path/'best.pt').write_bytes(original)
            (path/'unexpected.txt').write_text('extra')
            self.assertNotEqual(run().returncode, 0)

    def test_wrong_task_or_corrupted_weight_cannot_export(self):
        model, exports, _ = self.package()
        with self.assertRaises(ProjectError): exports.build(self.pid, self.b, model['id'])
        self.models.owned(model['id'], 'best.pt').write_bytes(b'corrupt')
        with self.assertRaises(ProjectError): exports.build(self.pid, self.a, model['id'])

    def test_http_download_and_scope(self):
        model, _, _ = self.package()
        server = create_server(0, self.path)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}/api/projects/{self.pid}/tasks/{self.a}/models/{model["id"]}/delivery'
        try:
            with urlopen(base) as response:
                self.assertEqual(response.headers['Content-Type'], 'application/zip')
                self.assertIn('python.zip', response.headers['Content-Disposition'])
                with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                    self.assertIn('predict.py', archive.namelist())
            with self.assertRaises(HTTPError) as error: urlopen(base.replace(self.a, self.b))
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_evaluated_and_latest_failed_attempt_keep_successful_metrics(self):
        model, _, _ = self.package()
        original_get = self.models.get
        value = original_get(self.pid, self.a, model['id'])
        value.update(latest_successful_evaluation_id='success', latest_evaluation={'id':'failed','state':'FAILED'})
        self.models.get = Mock(return_value=value)
        evaluations = Mock()
        evaluations.report.return_value = {'snapshot': {'weights_sha256': value['manifest']['artifacts']['best.pt']['sha256'],
            'subset': 'val', 'requested_parameters': {'conf':0.001}}, 'metrics':{'map50':0.5}, 'per_class':[]}
        exports = DeliveryRepository(self.models, evaluations)
        raw, _ = exports.build(self.pid, self.a, model['id'])
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            metrics = json.loads(archive.read('metrics.json'))
            self.assertEqual(metrics['status'], 'EVALUATED')
            self.assertEqual(metrics['metrics']['map50'], 0.5)
            self.assertEqual(metrics['latest_attempt']['state'], 'FAILED')
        evaluations.report.return_value['snapshot']['weights_sha256'] = 'wrong'
        with self.assertRaises(ProjectError): exports.build(self.pid, self.a, model['id'])
