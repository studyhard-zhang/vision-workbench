import hashlib
import io
import json
import threading
import unittest
import zipfile
from urllib.request import urlopen
from urllib.error import HTTPError

import test_executions as fixtures
from test_versions import VersionFixture
from app.storage import ProjectError
from app.training import TrainingRepository
from app.executions import ExecutionRepository
from app.training_reports import parse_metrics, TrainingReportRepository, MAX_CSV_BYTES
from app.versions import encoded, sha_file
from app.server import create_server


class MetricsParserTests(unittest.TestCase):
    def test_bom_spaces_and_numeric_values_keep_missing_distinct_from_zero(self):
        raw=b'\xef\xbb\xbf epoch , metrics/precision(B) , train/box_loss , lr/pg0\r\n 1 , 0 , , 0.000729904\r\n'
        parsed=parse_metrics(raw,50,1);row=parsed['rows'][0]
        self.assertEqual(row['epoch'],1);self.assertEqual(row['metrics/precision(B)'],0)
        self.assertIsNone(row['train/box_loss']);self.assertEqual(row['lr/pg0'],.000729904)
        self.assertIsNone(row['val/box_loss']);self.assertTrue(parsed['warnings'])

    def test_all_fifty_epochs_remain_ordered_with_exact_source_numbers(self):
        raw=('epoch,train/box_loss,metrics/mAP50-95(B),lr/pg0\n'+''.join(f'{i},{i/10},0.72725,0.000729904\n' for i in range(1,51))).encode()
        parsed=parse_metrics(raw,50,50)
        self.assertEqual([r['epoch'] for r in parsed['rows']],list(range(1,51)))
        self.assertEqual(parsed['rows'][-1]['metrics/mAP50-95(B)'],.72725)
        self.assertEqual(parsed['rows'][0]['lr/pg0'],.000729904)

    def test_missing_epochs_and_partial_results_are_explicit(self):
        parsed=parse_metrics(b'epoch,train/box_loss\n1,2\n3,1\n',50,5)
        self.assertEqual([r['epoch'] for r in parsed['rows']],[1,3])
        self.assertEqual(len(parsed['warnings']),3)

    def test_ambiguous_or_invalid_numbers_headers_and_epochs_are_rejected(self):
        bad=[b'',b'epoch,epoch\n1,1',b' epoch ,epoch\n1,1',b'loss\n1',b'epoch,\n1,2',
             b'epoch,train/box_loss\n1,NaN',b'epoch,train/box_loss\n1,inf',b'epoch,train/box_loss\n1,-1',
             b'epoch,metrics/recall(B)\n1,1.1',b'epoch\n1\n1',b'epoch\n2\n1',b'epoch\n0',
             b'epoch\n1.5',b'epoch\n51',b'epoch\ninf',b'epoch,train/box_loss\n1',b'epoch\n',b'\xff']
        for raw in bad:
            with self.subTest(raw=raw),self.assertRaises(ProjectError) as error:parse_metrics(raw,50)
            self.assertEqual(error.exception.code,'TRAINING_METRICS_INVALID')
        with self.assertRaises(ProjectError):parse_metrics(b'epoch\n2',50,1)

    def test_input_and_epoch_limits_are_enforced(self):
        with self.assertRaises(ProjectError):parse_metrics(b'x'*(MAX_CSV_BYTES+1),50)
        with self.assertRaises(ProjectError):parse_metrics(('epoch\n'+''.join(f'{i}\n' for i in range(1,302))).encode(),400)


class TrainingReportTests(VersionFixture):
    job=fixtures.ExecutionTests.job
    dispatch=fixtures.ExecutionTests.dispatch
    request=fixtures.ExecutionTests.request
    receipt=fixtures.ExecutionTests.receipt

    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions)
        self.executions=ExecutionRepository(self.store,self.training);self.reports=TrainingReportRepository(self.executions)

    def completed(self,raw=b'epoch,time,train/box_loss,metrics/mAP50-95(B),lr/pg0\n1,3.3,1.25,0.72725,0.000729904\n',status='completed'):
        job=self.job();self.dispatch(job)
        with zipfile.ZipFile(io.BytesIO(self.receipt(job,status=status))) as z:files={name:z.read(name) for name in z.namelist()}
        receipt=json.loads(files['receipt.json']);receipt['artifacts']['results.csv']={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        receipt['report']['actual_parameters'].update(lr0=.01,momentum=.937)
        files['results.csv']=raw;files['receipt.json']=encoded(receipt);out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            for name,content in files.items():z.writestr(name,content)
        return self.executions.receive(self.pid,self.a,job['id'],out.getvalue())['job']

    def test_report_is_read_only_bound_to_run_and_not_an_independent_evaluation(self):
        job=self.completed();before=self.training.get(self.pid,self.a,job['id']);path,_,_=self.executions.artifact(self.pid,self.a,job['id'],'results.csv');digest=sha_file(path)
        report=self.reports.get(self.pid,self.a,job['id'])
        self.assertEqual(report['metrics_scope'],'TRAINING_INTERNAL_VALIDATION');self.assertEqual(report['csv_sha256'],digest)
        self.assertEqual(report['rows'][0]['metrics/mAP50-95(B)'],.72725)
        self.assertEqual(report['actual_optimizer'],'CPU-test-fixture')
        lr=next(p for p in report['parameters'] if p['key']=='lr0')
        self.assertFalse(lr['requested_present']);self.assertTrue(lr['actual_present']);self.assertIn('auto',lr['note'])
        self.assertEqual(self.training.get(self.pid,self.a,job['id']),before);self.assertEqual(sha_file(path),digest)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM model_versions').fetchone()[0],0)

    def test_queued_cancelled_and_dispatched_have_no_fabricated_metrics(self):
        job=self.job()
        for state in ('QUEUED','CANCELLED'):
            report=self.reports.get(self.pid,self.a,job['id']);self.assertEqual(report['status'],'NOT_AVAILABLE');self.assertEqual(report['rows'],[])
            self.assertIsNone(report['actual_optimizer']);self.assertTrue(all(not p['actual_present'] for p in report['parameters']))
            if state=='QUEUED':self.training.cancel(self.pid,self.a,job['id'])
        job=self.job();self.dispatch(job);self.assertEqual(self.reports.get(self.pid,self.a,job['id'])['rows'],[])

    def test_partial_failed_results_have_a_warning_and_keep_failed_state(self):
        job=self.completed(status='failed');report=self.reports.get(self.pid,self.a,job['id'])
        self.assertEqual(report['status'],'AVAILABLE');self.assertEqual(report['job_state'],'FAILED')
        self.assertTrue(any('未成功' in w for w in report['warnings']))

    def test_missing_corrupt_and_malformed_artifacts_do_not_change_training_state(self):
        job=self.completed();path,_,_=self.executions.artifact(self.pid,self.a,job['id'],'results.csv');path.write_bytes(b'wrong')
        with self.assertRaises(ProjectError) as error:self.reports.get(self.pid,self.a,job['id'])
        self.assertEqual(error.exception.code,'ARTIFACT_DAMAGED');path.unlink()
        with self.assertRaises(ProjectError):self.reports.get(self.pid,self.a,job['id'])
        job2=self.completed(raw=b'epoch,metrics/recall(B)\n1,NaN')
        with self.assertRaises(ProjectError) as error:self.reports.get(self.pid,self.a,job2['id'])
        self.assertEqual(error.exception.code,'TRAINING_METRICS_INVALID')
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'SUCCEEDED')
        self.assertEqual(self.training.get(self.pid,self.a,job2['id'])['state'],'SUCCEEDED')

    def test_http_scope_and_report_identity(self):
        job=self.completed();server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        origin=f'http://127.0.0.1:{server.server_port}';base=f'/api/projects/{self.pid}/tasks/{self.a}/training/{job["id"]}/report'
        try:
            with urlopen(origin+base) as r:self.assertEqual(json.load(r)['job_id'],job['id'])
            with self.assertRaises(HTTPError) as error:urlopen(origin+base.replace(self.a,self.b))
            self.assertEqual(error.exception.code,404)
            with urlopen(origin+'/training-report.js') as r:self.assertIn(b'createTrainingReport',r.read())
            with urlopen(origin+'/training-chart.mjs') as r:self.assertIn(b'chartGeometry',r.read())
        finally:server.shutdown();server.server_close();thread.join()
