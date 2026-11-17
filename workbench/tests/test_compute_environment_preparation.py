import json
import hashlib
import unittest
from tools.prepare_compute_environment import BASELINE,build_plan,apply_requirements,encoded


class EnvironmentPreparationTests(unittest.TestCase):
    def report(self):
        return {'install':[{'metadata':{'name':n,'version':v},'download_info':{'url':'https://example.invalid/'+n+'.whl','archive_info':{'hashes':{'sha256':'a'*64}}}} for n,v in [('ultralytics','8.3.241'),('opencv-python','4.11.0.86'),('dill','0.3.8')]]}

    def test_plan_is_hash_bound_and_install_never_resolves_or_replaces_existing_packages(self):
        raw=encoded(build_plan(self.report(),BASELINE));sha=hashlib.sha256(raw).hexdigest()
        requirements=apply_requirements(raw,sha,BASELINE)
        self.assertIn('ultralytics==8.3.241 --hash=sha256:',requirements);self.assertNotIn('torch==',requirements)
        for digest,installed in [('0'*64,BASELINE),(sha,{**BASELINE,'new-package':'1.0'})]:
            with self.assertRaises(ValueError):apply_requirements(raw,digest,installed)

    def test_existing_replacement_source_distribution_and_missing_baseline_are_refused(self):
        report=self.report()
        with self.assertRaises(ValueError):build_plan(report,{**BASELINE,'ultralytics':'8.0.0'})
        report['install'][0]['download_info']['url']='https://example.invalid/source.tar.gz'
        with self.assertRaises(ValueError):build_plan(report,BASELINE)
        with self.assertRaises(ValueError):build_plan(self.report(),{})

    def test_partial_or_unhashed_plan_is_not_installable(self):
        report=self.report();report['install'][0]['download_info']['archive_info']['hashes']={}
        with self.assertRaises(ValueError):build_plan(report,BASELINE)
        report=self.report();report['install'].pop()
        with self.assertRaises(ValueError):build_plan(report,BASELINE)
