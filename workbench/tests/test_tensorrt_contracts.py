import copy,hashlib,io,json,uuid,zipfile
from pathlib import Path
from unittest.mock import patch
from test_inferences import InferenceFixture
from app.storage import ProjectError
from tools.deployment_build import verify
from tools.run_inference_job import pack_result
from tools.deployment_predict_trt import Detector

class TensorRTContracts(InferenceFixture):
    def acceleration(self):
        return {'profile':{'acceleration':{'python_executable':'/root/.vision-workbench/environment-v1/trt/bin/python','runtime':{'tensorrt':'10.9.0.34','onnx':'1.17.0'}}}}
    def test_unprepared_backend_does_not_create_a_job(self):
        before=len(self.inferences.list(self.pid,self.a))
        with self.assertRaises(ProjectError) as error:self.create(backend='tensorrt')
        self.assertEqual(error.exception.code,'TENSORRT_NOT_READY');self.assertEqual(before,len(self.inferences.list(self.pid,self.a)))
    def test_explicit_backend_and_profile_are_frozen_and_do_not_change_pt(self):
        native=self.create()
        self.assertNotIn('backend',native['snapshot']);self.assertFalse(native['snapshot']['requested_parameters']['half'])
        with patch('app.compute_instances.active',return_value=self.acceleration()):job=self.create(backend='tensorrt',deployment_profile='cuda11')
        self.assertTrue(job['snapshot']['requested_parameters']['half']);self.assertEqual(job['snapshot']['acceleration']['deployment_profile'],'cuda11')
        self.assertEqual(native,self.inferences.get(self.pid,self.a,native['id']))
    def test_invalid_backend_or_cuda_profile_rejected(self):
        for values in [{'backend':'auto'},{'backend':'tensorrt','deployment_profile':'cuda13'}]:
            with self.assertRaises(ProjectError):self.create(**values)
    def test_delivery_checksum_check_rejects_tampering_without_ml(self):
        folder=self.path.parent/'delivery';folder.mkdir();(folder/'model.onnx').write_bytes(b'fixture')
        spec={'files':{'model.onnx':{'bytes':7,'sha256':hashlib.sha256(b'fixture').hexdigest()}}};(folder/'checksums.json').write_text(json.dumps(spec));verify(folder)
        (folder/'model.onnx').write_bytes(b'changed')
        with self.assertRaises(RuntimeError):verify(folder)
    def test_oversized_delivery_fails_without_truncating_results(self):
        folder=self.path.parent/'large-delivery';folder.mkdir()
        original={'images':[{'predictions':[{'segments':[{'hole':True,'points':[[0,0],[3,0],[3,3]]}]}]}]}
        (folder/'report.json').write_text(json.dumps(original))
        with (folder/'deployment.zip').open('wb') as stream:stream.truncate(121*1024**2)
        request={k:'fixture' for k in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256','return_token')};request['snapshot']={'backend':'tensorrt'}
        report={'status':'completed'};pack_result(folder,request,report)
        with zipfile.ZipFile(folder/'return-result.zip') as archive:
            receipt=json.loads(archive.read('receipt.json'))
            self.assertEqual(receipt['report']['status'],'failed');self.assertNotIn('deployment.zip',archive.namelist())
            self.assertEqual(json.loads(archive.read('report.json')),original)
        self.assertTrue((folder/'deployment.zip').exists())
    def test_invalid_production_threshold_rejected_before_loading_ml(self):
        for conf in (True,float('nan'),float('inf'),0,1):
            with self.assertRaises(ValueError):Detector(conf=conf)
