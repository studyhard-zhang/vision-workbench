from unittest.mock import patch
import test_executions as fixtures
from test_versions import VersionFixture
from app.training import TrainingRepository
from app.executions import ExecutionRepository
from app.models import ModelRepository


class ModelAutomationTests(VersionFixture):
    job=fixtures.ExecutionTests.job
    dispatch=fixtures.ExecutionTests.dispatch
    request=fixtures.ExecutionTests.request
    receipt=fixtures.ExecutionTests.receipt
    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions);self.executions=ExecutionRepository(self.store,self.training);self.models=ModelRepository(self.store,self.executions);self.executions.on_success=self.models.automatic
    def finish(self):
        job=self.job();self.dispatch(job);raw=self.receipt(job);return job,raw,self.executions.receive(self.pid,self.a,job['id'],raw)
    def test_success_auto_creates_model_once_without_evaluation_or_compute(self):
        job,raw,result=self.finish();self.assertEqual(result['job']['model_version']['number'],1);self.assertEqual(result['model']['training_run_id'],job['id'])
        again=self.executions.receive(self.pid,self.a,job['id'],raw);self.assertEqual(again['model']['id'],result['model']['id']);self.assertEqual(self.finish()[2]['model']['number'],2)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM model_events').fetchone()[0],2);self.assertEqual(db.execute('SELECT count(*) FROM evaluation_jobs').fetchone()[0],0)
    def test_publication_error_preserves_receipt_and_retry_repairs_same_model(self):
        with patch.object(self.models,'register',side_effect=OSError('disk unavailable')):job,raw,result=self.finish()
        self.assertEqual(result['job']['state'],'SUCCEEDED');self.assertIn('model_error',result);repaired=self.executions.receive(self.pid,self.a,job['id'],raw);self.assertEqual(repaired['model']['number'],1)
    def test_startup_backfill_is_idempotent_and_skips_trash(self):
        self.executions.on_success=None;first,_,_=self.finish();second,_,_=self.finish();job=self.training.get(self.pid,self.a,second['id']);self.training.change_visibility(self.pid,self.a,[{'id':job['id'],'confirmation_name':job['name'],'visibility_revision':job['visibility_revision']}])
        self.models.sync_successful();self.models.sync_successful();self.assertEqual([m['training_run_id'] for m in self.models.list(self.pid,self.a)],[first['id']])
