import unittest
from types import SimpleNamespace
from tools.run_training_job import resolved_parameter


class RuntimePrecisionTests(unittest.TestCase):
    def test_legacy_half_and_normalized_quantize_are_equivalent(self):
        for arguments,expected in (({'half':False},False),({'half':True},True),({'quantize':None},False),({'quantize':32},False),({'quantize':16},True)):
            with self.subTest(arguments=arguments):
                self.assertIs(resolved_parameter(SimpleNamespace(**arguments),'half'),expected)

    def test_other_parameters_are_not_normalized_and_unknown_precision_is_rejected(self):
        self.assertEqual(resolved_parameter(SimpleNamespace(imgsz=640),'imgsz'),640)
        for value in (8,'w8a16',False):
            with self.subTest(value=value),self.assertRaises(RuntimeError):
                resolved_parameter(SimpleNamespace(quantize=value),'half')
