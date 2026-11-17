import ast
import io
import json
import subprocess
import sys
import hashlib
import zipfile
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_versions import VersionFixture
from tools.compute_probe import inspect_environment,ultralytics_import,checkpoint_dependencies
from tools.smoke_train import execute,target_config,file_hash
from tools.resume_autodl import locate_weights,resolve_runtime
from tools import run_training_job as runner
from app.training import TARGET,PARAMETERS


class ComputeToolTests(VersionFixture):
    def setUp(self):
        super().setUp();self.profile=self.path.parent/'target.json'
        self.profile.write_text(json.dumps({'format_version':1,'target_id':'test-host','kind':'autodl_instance','transport':'manual_terminal','gpu_selector':'0','expected_gpu':'2080 Ti'}),encoding='utf-8')
        version=self.publish()['version']['id'];self.bundle=self.path.parent/'dataset.zip'
        with self.versions.archive(self.pid,self.a,version) as (stream,_,_):self.bundle.write_bytes(stream.read())

    def args(self,prepare_only=True):
        return SimpleNamespace(target=self.profile,bundle=self.bundle,output=self.path.parent/'runs',weights=None,prepare_only=prepare_only)

    def test_metadata_only_is_not_gpu_readiness_and_never_imports_torch(self):
        original=__import__
        def guarded(name,*args,**kwargs):
            if name.startswith(('torch','ultralytics')):raise AssertionError('GPU import in metadata-only mode')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',guarded):report=inspect_environment(self.path.parent,metadata_only=True)
        self.assertFalse(report['ready']);self.assertFalse(report['training_executed']);self.assertTrue(report['metadata_only'])
        self.assertNotIn('gpu',report)

    def test_checkpoint_optional_dependency_is_detected_without_unpickling_or_gpu_imports(self):
        weights=self.path.parent/'yolo11n.pt'
        with zipfile.ZipFile(weights,'w') as z:z.writestr('archive/data.pkl',b'\x80\x02cdill._dill\n_load_type\n.')
        original=__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','ultralytics','dill','numpy'):raise AssertionError('Unexpected runtime import: '+name)
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',guarded),patch('tools.compute_probe.importlib.metadata.version',side_effect=importlib.metadata.PackageNotFoundError),patch('pickle.loads',side_effect=AssertionError('No unpickling')):
            report=checkpoint_dependencies(weights)
        self.assertEqual(report['missing'],['dill']);self.assertFalse(report['checkpoint_deserialized']);self.assertFalse(report['gpu_used'])
        with patch('tools.compute_probe.importlib.metadata.version',return_value='0.3.8'):
            self.assertEqual(checkpoint_dependencies(weights)['missing'],[])

    def test_missing_checkpoint_dependency_stops_before_cuda_environment_probe(self):
        weights=self.path.parent/'yolo11n.pt'
        with zipfile.ZipFile(weights,'w') as z:z.writestr('archive/data.pkl',b'\x80\x02cdill._dill\n_load_type\n.')
        target={**TARGET,'weights':{**TARGET['weights'],'path':str(weights.resolve()),'sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}}
        request={'snapshot':{'target':target,'requested_parameters':{**PARAMETERS,'epochs':1}}};report={}
        with patch('tools.compute_probe.importlib.metadata.version',side_effect=importlib.metadata.PackageNotFoundError),patch('tools.compute_probe.inspect_environment',side_effect=AssertionError('CUDA probe must not run')) as probe:
            with self.assertRaisesRegex(RuntimeError,'dill'):runner.train(self.path.parent,request,report)
            probe.assert_not_called()
        self.assertEqual(report['error_code'],'CHECKPOINT_DEPENDENCY_MISSING');self.assertNotIn('environment',report)

    def test_target_profile_keeps_self_hosted_option_and_rejects_cpu_or_multiple_devices(self):
        value=json.loads(self.profile.read_text(encoding='utf-8'));value['kind']='self_hosted';value['expected_gpu']='RTX 5060'
        self.profile.write_text(json.dumps(value));self.assertEqual(target_config(self.profile)['kind'],'self_hosted')
        for selector in ('cpu','0,1','0; command',''):
            value['gpu_selector']=selector;self.profile.write_text(json.dumps(value))
            with self.assertRaises(ValueError):target_config(self.profile)

    def test_prepare_only_records_version_and_target_without_computing_or_overwriting_input(self):
        before=file_hash(self.bundle)
        with patch('tools.smoke_train.inspect_environment',side_effect=AssertionError('not allowed')):
            result,run=execute(self.args())
        self.assertEqual(result['status'],'prepared_only');self.assertFalse(result['training_executed'])
        self.assertEqual(result['target_id'],'test-host');self.assertEqual(file_hash(self.bundle),before)
        self.assertTrue((run/'dataset/data.yaml').is_file());self.assertEqual(json.loads((run/'result.json').read_text(encoding='utf-8'))['status'],'prepared_only')
        spec=json.loads((run/'run-spec.json').read_text(encoding='utf-8'));self.assertEqual(spec['parameters']['device'],0);self.assertEqual(spec['parameters']['epochs'],1)
        self.assertFalse((run/'train/weights/last.pt').exists())

    def test_failed_environment_never_starts_training_and_preserves_error_report(self):
        with patch('tools.smoke_train.inspect_environment',return_value={'ready':False}):result,run=execute(self.args(False))
        self.assertEqual(result['status'],'failed');self.assertFalse(result['training_executed'])
        self.assertIn('计算环境检查',result['error']);self.assertTrue((run/'compute-env.json').is_file())
        self.assertEqual(result['environment'],{'ready':False})

    def test_missing_weights_cannot_trigger_automatic_download_or_training(self):
        with patch('tools.smoke_train.inspect_environment',return_value={'ready':True}):result,run=execute(self.args(False))
        self.assertEqual(result['status'],'failed');self.assertFalse(result['training_executed']);self.assertIn('本地yolo11n.pt',result['error'])

    def test_python310_syntax_and_cli_metadata_report_do_not_claim_ready(self):
        root=Path(__file__).resolve().parents[1]
        for name in ['compute_probe.py','smoke_train.py']:ast.parse((root/'tools'/name).read_text(encoding='utf-8'),feature_version=(3,10))
        output=self.path.parent/'metadata.json'
        process=subprocess.run([sys.executable,str(root/'tools/compute_probe.py'),'--metadata-only','--workdir',str(self.path.parent),'--output',str(output)],capture_output=True)
        self.assertEqual(process.returncode,2,process.stderr)
        report=json.loads(output.read_text(encoding='utf-8'));self.assertTrue(report['metadata_only']);self.assertFalse(report['ready'])

    def test_source_import_without_distribution_metadata_is_a_valid_runtime(self):
        import importlib.metadata
        module=SimpleNamespace(__version__='8.3.source',__file__='/mirror/ultralytics/__init__.py',YOLO=lambda:None)
        with patch('tools.compute_probe.importlib.metadata.version',side_effect=importlib.metadata.PackageNotFoundError),patch('tools.compute_probe.importlib.import_module',return_value=module):
            details=ultralytics_import()
        self.assertTrue(details['importable']);self.assertIsNone(details['distribution_version']);self.assertEqual(details['origin'],module.__file__)

    def test_metadata_alone_cannot_hide_a_missing_or_broken_yolo_import(self):
        with patch('tools.compute_probe.importlib.metadata.version',return_value='8.3.241'),patch('tools.compute_probe.importlib.import_module',side_effect=ModuleNotFoundError('cv2')):
            details=ultralytics_import()
        self.assertFalse(details['importable']);self.assertIn('cv2',details['error'])

    def test_existing_weight_path_is_resolved_without_copying_or_downloading(self):
        weights=self.path.parent/'yolov11/model/yolo11n.pt';weights.parent.mkdir(parents=True);weights.write_bytes(b'existing test weight')
        before=weights.read_bytes();found=locate_weights('yolov11/model/yolo11n.pt',self.path.parent)
        self.assertEqual(found,weights.resolve());self.assertEqual(found.read_bytes(),before)
        with self.assertRaises(ValueError):locate_weights('yolov11/model/absent.pt',self.path.parent)

    def test_runtime_source_fallback_is_limited_to_selected_weight_repository(self):
        weights=self.path.parent/'yolov11/model/yolo11n.pt';weights.parent.mkdir(parents=True);weights.write_bytes(b'test')
        package=weights.parent.parent/'ultralytics';package.mkdir();(package/'__init__.py').write_text('# fixture',encoding='utf-8')
        with patch.object(sys,'path',list(sys.path)),patch.dict(sys.modules),patch('tools.resume_autodl.ultralytics_import',side_effect=[{'importable':False},{'importable':True,'origin':str(package/'__init__.py')}]):
            details=resolve_runtime(weights)
            self.assertTrue(details['importable']);self.assertEqual(sys.path[0],str(weights.parent.parent.resolve()))
