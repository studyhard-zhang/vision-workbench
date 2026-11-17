import base64,contextlib,io,json,sys
from unittest import TestCase
from unittest.mock import patch,Mock
from app.compute_instances import ComputeInstances,run_check
from app.storage import ProjectError
from tools import remote_environment as environment


class EnvironmentValidationTests(TestCase):
    def test_removed_install_action_cannot_start_subprocess(self):
        payload=base64.b64encode(json.dumps({'action':'prepare'}).encode()).decode();output=io.StringIO()
        with patch.object(sys,'argv',['check',payload]),patch.object(environment.subprocess,'run') as process,contextlib.redirect_stdout(output):environment.main()
        self.assertFalse(json.loads(output.getvalue())['ok']);process.assert_not_called()
        with self.assertRaises(ProjectError):run_check(Mock(),'prepare')

    def test_missing_dependencies_return_before_importing_ml(self):
        plan={'blockers':[],'missing':[{'name':'torch'}],'missing_weights':[]}
        with patch.dict(sys.modules,{'torch':None}):result=environment.validate_runtime(plan)
        self.assertFalse(result['ready']);self.assertEqual(result['plan'],plan)

    def test_environment_only_proof_supports_existing_external_python_and_partial_models(self):
        candidate={'id':'a'*32,'endpoint':'root@host:22','profile':{},'plan':{'root':environment.ROOTS[1],'recipe':'cu128','gpu':{'uuid':'GPU-fixture','name':'NVIDIA GPU'}}}
        proof={'run_id':'environment-fixture','environment_checked':True,'forward_passed':True,'weights_sha256':environment.WEIGHTS['yolo26n.pt'],'training_executed':False}
        result={'ready':True,'validation_kind':'environment','python':'/opt/user-training/bin/python','runtime':{'python':'3.12.3','torch':'2.7.1+cu128','torchvision':'0.22.1+cu128','ultralytics':'8.4.203'},'tests':{'yolo26n.pt':proof}}
        record=ComputeInstances.verified_record(candidate,result)
        self.assertEqual(record['profile']['python_executable'],result['python']);self.assertEqual(set(record['weights']),{'yolo26n.pt'})
        result['tests']['yolo26n.pt']['forward_passed']=False
        with self.assertRaises(ProjectError):ComputeInstances.verified_record(candidate,result)
