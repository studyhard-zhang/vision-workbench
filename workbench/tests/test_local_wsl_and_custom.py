import unittest
from unittest.mock import patch,MagicMock
from app.local_wsl import LocalWSL
from tools.training_config import configuration,parameters,OFFICIAL26,PRECISION_CONFIG
from tools.run_training_job import failure_details

class CustomParametersTests(unittest.TestCase):
    def test_defaults_and_overrides_leave_augmentation_and_amp_unchanged(self):
        self.assertEqual(parameters({'profile':'custom26'},3),{**OFFICIAL26,'epochs':3})
        self.assertEqual(parameters({'profile':'custom26','batch':2,'imgsz':1024},3),{**OFFICIAL26,'epochs':3,'batch':2,'imgsz':1024})
        self.assertEqual(configuration({'profile':'custom26','batch':16}),{'profile':'custom26'})
        self.assertFalse(parameters(PRECISION_CONFIG,1)['amp'])
    def test_invalid_values_do_not_enter_a_snapshot(self):
        for key,value in [('batch',True),('batch',0),('batch',3),('imgsz',True),('imgsz',641),('imgsz',4096),('seed',-1),('lr0',.1),('amp',False)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):configuration({'profile':'custom26',key:value})
    def test_oom_has_specific_manual_recovery_guidance(self):
        r=failure_details(RuntimeError('CUDA out of memory'))
        self.assertEqual(r['code'],'CUDA_OUT_OF_MEMORY')
        self.assertIn('批大小',r['next_step']);self.assertIn('不会自动',r['next_step'])

class KeeperTests(unittest.TestCase):
    def test_external_target_never_runs_wsl(self):
        keeper=LocalWSL()
        with patch('app.local_wsl.subprocess.Popen') as popen:
            keeper.ensure({'host':'remote.example'});keeper.start({'host':'remote.example'});popen.assert_not_called()
    def test_keeper_is_reused_and_closes_its_stdin_without_stopping_jobs(self):
        keeper=LocalWSL();process=MagicMock();process.poll.return_value=None;process.stdin.closed=False
        sock=MagicMock();sock.__enter__.return_value.recv.return_value=b'SSH-2.0-OpenSSH'
        with patch('app.local_wsl.is_local',return_value=True),patch('app.local_wsl.subprocess.Popen',return_value=process) as popen,patch('app.local_wsl.socket.create_connection',return_value=sock):
            keeper.ensure({});keeper.ensure({});self.assertEqual(popen.call_count,1)
            self.assertIn('exec cat',popen.call_args.args[0][-1]);keeper.close();process.stdin.close.assert_called_once();process.terminate.assert_not_called()


    def test_missing_wsl_reports_local_error_and_does_not_connect(self):
        from app.storage import ProjectError
        keeper=LocalWSL()
        with patch('app.local_wsl.is_local',return_value=True),patch('app.local_wsl.subprocess.Popen',side_effect=FileNotFoundError()),patch('app.local_wsl.socket.create_connection') as connect:
            with self.assertRaises(ProjectError) as error:keeper.ensure({})
            self.assertEqual(error.exception.code,'LOCAL_EXECUTOR_UNAVAILABLE');connect.assert_not_called()
    def test_exited_keeper_is_replaced_before_connecting(self):
        keeper=LocalWSL();keeper.process=MagicMock();keeper.process.poll.return_value=1
        replacement=MagicMock();replacement.poll.return_value=None
        sock=MagicMock();sock.__enter__.return_value.recv.return_value=b'SSH-2.0-test'
        with patch('app.local_wsl.is_local',return_value=True),patch('app.local_wsl.subprocess.Popen',return_value=replacement) as popen,patch('app.local_wsl.socket.create_connection',return_value=sock):
            keeper.ensure({});popen.assert_called_once();self.assertIs(keeper.process,replacement)
