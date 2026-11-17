import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import test_executions as fixtures
from test_versions import VersionFixture
from app.training import TrainingRepository,unavailable_target
from app.executions import ExecutionRepository
from app.remote import RemoteRepository
from app.storage import ProjectError,timestamp


class AbandonmentTests(VersionFixture):
    job=fixtures.ExecutionTests.job
    request=fixtures.ExecutionTests.request
    receipt=fixtures.ExecutionTests.receipt

    def setUp(self):
        super().setUp()
        self.vid=self.publish()['version']['id']
        self.training=TrainingRepository(self.store,self.versions)
        self.executions=ExecutionRepository(self.store,self.training)
        self.remote=RemoteRepository(self.store,self.training,self.executions)
        self.original=self.job()
        self.executions.dispatch(self.pid,self.a,self.original['id'])
        with self.store.connect() as db:
            db.execute("INSERT INTO training_remote_sessions(job_id,endpoint,phase,created_at,updated_at) VALUES(?,?,'ERROR',?,?)",(self.original['id'],'offline.example.invalid:22',timestamp(),timestamp()))
        self.values=dict(confirmation_name=self.original['name'],visibility_revision=0,instance_unavailable=True)

    def abandon(self):return self.remote.abandon(self.pid,self.a,self.original['id'],self.values)

    def test_offline_release_is_atomic_idempotent_retains_evidence_and_rejects_late_result(self):
        receipt=self.receipt(self.original)
        with patch.object(self.remote,'transport',side_effect=AssertionError('must stay offline')):
            with ThreadPoolExecutor(2) as pool:results=list(pool.map(lambda _:self.abandon(),range(2)))
            self.assertEqual(sorted(r['reused'] for r in results),[False,True])
            job=self.training.get(self.pid,self.a,self.original['id'])
            self.assertEqual(job['snapshot'],self.original['snapshot'])
            self.assertEqual(job['dispatch']['state'],'FAILED')
            self.assertIsNotNone(job['deleted_at'])
            self.assertFalse(job['remote']['progress']['local_abandonment']['remote_stop_confirmed'])
            self.remote.reconcile({'job_id':job['id']})
            self.remote.update(job['id'],'ERROR',{'stale':True})
            self.assertEqual(self.training.get(self.pid,self.a,job['id'])['remote']['phase'],'COMPLETED')
        with self.assertRaises(ProjectError):self.executions.receive(self.pid,self.a,job['id'],receipt)
        self.assertTrue(self.executions.package(self.pid,self.a,job['id'])[0].is_file())
        from app.training_reports import TrainingReportRepository
        self.assertEqual(TrainingReportRepository(self.executions).get(self.pid,self.a,job['id'])['status'],'NOT_AVAILABLE')
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM training_dispatches WHERE state='AWAITING_RESULT'").fetchone()[0],0)
            self.assertTrue(unavailable_target(db,job['snapshot']['target']))
            self.assertFalse(unavailable_target(db,{**job['snapshot']['target'],'verified_gpu_uuid':'GPU-other-instance'}))
        self.training.change_visibility(self.pid,self.a,[dict(id=job['id'],confirmation_name=job['name'],visibility_revision=job['visibility_revision'])],True)
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'FAILED')
        with self.assertRaises(ProjectError) as error:self.job()
        self.assertEqual(error.exception.code,'COMPUTE_INSTANCE_UNAVAILABLE')

    def test_missing_confirmation_stale_revision_and_running_session_are_rejected(self):
        for values in ({**self.values,'instance_unavailable':False},{**self.values,'visibility_revision':1}):
            with self.assertRaises(ProjectError):self.remote.abandon(self.pid,self.a,self.original['id'],values)
        with self.store.connect() as db:db.execute("UPDATE training_remote_sessions SET phase='RUNNING'")
        with self.assertRaises(ProjectError):self.abandon()
        self.assertIsNone(self.training.get(self.pid,self.a,self.original['id'])['deleted_at'])

    def test_failure_during_deletion_rolls_back_release(self):
        with self.store.connect() as db:db.execute("CREATE TRIGGER fail_visibility BEFORE INSERT ON training_record_visibility BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(Exception):self.abandon()
        job=self.training.get(self.pid,self.a,self.original['id'])
        self.assertEqual(job['dispatch']['state'],'AWAITING_RESULT')
        self.assertEqual(job['remote']['phase'],'ERROR')
        self.assertIsNone(job['deleted_at'])
