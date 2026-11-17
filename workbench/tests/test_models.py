"""CPU-only registry contracts in temporary stores. Fixture bytes are never real models."""
import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import test_executions as fixtures
from test_versions import VersionFixture
from app.storage import Storage, ProjectError
from app.models import ModelRepository
from app.training import TrainingRepository
from app.executions import ExecutionRepository
from app.server import create_server


class ModelTests(VersionFixture):
    job=fixtures.ExecutionTests.job
    dispatch=fixtures.ExecutionTests.dispatch
    request=fixtures.ExecutionTests.request
    receipt=fixtures.ExecutionTests.receipt

    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id']
        self.training=TrainingRepository(self.store,self.versions)
        self.executions=ExecutionRepository(self.store,self.training)
        self.models=ModelRepository(self.store,self.executions)

    def success(self):
        job=self.job();self.dispatch(job)
        self.executions.receive(self.pid,self.a,job['id'],self.receipt(job))
        return job

    def register(self,job):return self.models.register(self.pid,self.a,job['id'])

    def test_sequential_versions_idempotence_restart_and_preserved_download(self):
        first_job=self.success();first=self.register(first_job);model=first['model']
        self.assertFalse(first['reused']);self.assertEqual(model['number'],1)
        self.assertEqual(self.register(first_job),{'model':model,'reused':True})
        second=self.register(self.success())['model'];self.assertEqual(second['number'],2)
        self.assertEqual([v['number'] for v in self.models.list(self.pid,self.a)],[2,1])
        restored=ModelRepository(Storage(self.path),self.executions)
        self.assertEqual(restored.get(self.pid,self.a,model['id']),model)
        self.assertTrue(model['available']);self.assertEqual(model['evaluation_status'],'NOT_EVALUATED')
        self.assertEqual(model['manifest']['classes'],self.detail(self.publish())['manifest']['classes'])
        self.assertEqual(model['manifest']['training_snapshot'],first_job['snapshot'])
        self.assertEqual(model['manifest']['selection_rule']['runtime'],'版本未记录')
        for name,spec in model['manifest']['artifacts'].items():
            path,filename,_=restored.artifact(self.pid,self.a,model['id'],name)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),spec['sha256']);self.assertIn('M001',filename)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM model_events').fetchone()[0],2)
            event=json.loads(db.execute('SELECT payload_json FROM model_events WHERE model_version_id=?',(model['id'],)).fetchone()[0])
            self.assertEqual(event['evaluation_policy'],'DEFERRED_UNTIL_EVALUATION_NODE')
            self.assertFalse(db.execute('PRAGMA foreign_key_check').fetchall())

    def test_concurrent_same_and_different_runs_allocate_once(self):
        a,b=self.success(),self.success()
        other=ModelRepository(self.store,self.executions)
        with ThreadPoolExecutor(3) as pool:
            results=list(pool.map(lambda args:args[0].register(self.pid,self.a,args[1]['id']),[(self.models,a),(other,a),(self.models,b)]))
        self.assertEqual(sum(not r['reused'] for r in results),2)
        self.assertEqual({r['model']['number'] for r in results},{1,2})
        self.assertEqual(len(list(self.models.root.glob('*/best.pt'))),2)

    def test_incomplete_failed_cancelled_and_interrupted_rejected(self):
        a=self.job()
        with self.assertRaises(ProjectError):self.register(a)
        self.training.cancel(self.pid,self.a,a['id'])
        with self.assertRaises(ProjectError):self.register(a)
        for status in ('failed','interrupted'):
            job=self.job();self.dispatch(job)
            with self.assertRaises(ProjectError):self.register(job)
            self.executions.receive(self.pid,self.a,job['id'],self.receipt(job,status=status))
            with self.assertRaises(ProjectError):self.register(job)
        self.assertEqual(self.models.list(self.pid,self.a),[])

    def test_damaged_source_and_registered_files_cannot_be_used(self):
        job=self.success();path,_,_=self.executions.artifact(self.pid,self.a,job['id'],'best.pt');original=path.read_bytes();path.write_bytes(b'corrupt')
        with self.assertRaises(ProjectError):self.register(job)
        self.assertEqual(self.models.list(self.pid,self.a),[]);path.write_bytes(original)
        model=self.register(job)['model'];copy=self.models.owned(model['id'],'last.pt');copy.write_bytes(b'corrupt')
        broken=self.models.get(self.pid,self.a,model['id']);self.assertFalse(broken['available'])
        with self.assertRaises(ProjectError):self.models.artifact(self.pid,self.a,model['id'],'best.pt')
        # Idempotence preserves damaged history rather than silently replacing it.
        again=self.register(job);self.assertTrue(again['reused']);self.assertFalse(again['model']['available'])
        self.assertEqual(path.read_bytes(),original)

    def test_write_or_database_failure_rolls_back_model_and_event(self):
        job=self.success()
        with patch('app.models.Path.write_bytes',side_effect=OSError(28,'full')),self.assertRaises(ProjectError):self.register(job)
        with self.store.connect() as db:
            db.execute("CREATE TRIGGER inject_event_failure BEFORE INSERT ON model_events BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.register(job)
        self.assertEqual(self.models.list(self.pid,self.a),[])
        self.assertEqual(list(self.models.staging.iterdir()),[])
        self.assertEqual(list(self.models.root.glob('*/best.pt')),[])
        with self.store.connect() as db:db.execute('DROP TRIGGER inject_event_failure')
        self.assertEqual(self.register(job)['model']['number'],1)

    def test_immutable_history_and_reference_protection(self):
        model=self.register(self.success())['model']
        for query in ('DELETE FROM model_versions','UPDATE model_versions SET number=99','DELETE FROM model_events',
                      'DELETE FROM training_jobs','DELETE FROM task_data_versions','DELETE FROM vision_tasks'):
            with self.subTest(query=query),self.assertRaises(sqlite3.IntegrityError),self.store.connect() as db:db.execute(query)
        with self.assertRaises(ProjectError):self.models.get(self.pid,self.b,model['id'])
        with self.assertRaises(ProjectError):self.models.artifact(self.pid,self.a,model['id'],'../../platform.db')
        with self.assertRaises(ProjectError):self.models.register(self.pid,self.b,model['training_run_id'])

    def test_v10_migration_is_backed_up_and_rolls_back(self):
        job=self.success()
        with self.store.connect() as db:
            __import__('schema_utils').drop_evaluations(db);db.execute('DROP TABLE training_record_visibility');db.execute('DROP TABLE model_events');db.execute('DROP TABLE model_versions');db.execute('DROP INDEX training_task_identity')
            db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=10')
            before=[tuple(r) for r in db.execute('SELECT * FROM training_jobs')]
        class Broken(Storage):
            @staticmethod
            def create_model_tables(db):Storage.create_model_tables(db);raise RuntimeError('injected')
        with self.assertRaises(RuntimeError):Broken(self.path)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],10)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='model_versions'").fetchone())
        restored=Storage(self.path)
        with restored.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16)
            self.assertEqual([tuple(r) for r in db.execute('SELECT * FROM training_jobs')],before)
        self.assertTrue(list((self.path.parent/'backups').glob('schema-v10-*.db')))
        self.assertEqual(self.register(job)['model']['number'],1)

    def test_http_registration_download_identity_and_errors(self):
        job=self.success();server=create_server(0,self.path)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        origin=f'http://127.0.0.1:{server.server_port}';base=f'/api/projects/{self.pid}/tasks/{self.a}'
        def fetch(path,body=None):return urlopen(Request(origin+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=15)
        try:
            endpoint=base+'/training/'+job['id']+'/register-model'
            with fetch(endpoint,{}) as r:self.assertEqual(r.status,200);model=json.load(r)['model']
            with fetch(endpoint,{}) as r:self.assertEqual(r.status,200);self.assertTrue(json.load(r)['reused'])
            with fetch(base+'/models') as r:self.assertEqual(json.load(r)['total'],1)
            with fetch(base+'/models/'+model['id']+'/artifacts/best.pt') as r:
                self.assertIn('M001',r.headers['Content-Disposition']);self.assertEqual(hashlib.sha256(r.read()).hexdigest(),model['manifest']['artifacts']['best.pt']['sha256'])
            with fetch(base+'/models/'+model['id']+'/artifacts/manifest.json') as r:self.assertNotIn('return_token',r.read().decode())
            with self.assertRaises(HTTPError) as e:fetch(endpoint,{'path':'arbitrary'})
            self.assertEqual(e.exception.code,422)
            with self.assertRaises(HTTPError) as e:fetch(base.replace(self.a,self.b)+'/models/'+model['id'])
            self.assertEqual(e.exception.code,404)
            self.models.owned(model['id'],'best.pt').unlink()
            with self.assertRaises(HTTPError) as e:fetch(base+'/models/'+model['id']+'/artifacts/best.pt')
            self.assertEqual(e.exception.code,409)
        finally:server.shutdown();server.server_close();thread.join()
