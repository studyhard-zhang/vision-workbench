import concurrent.futures
import hashlib
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from test_versions import VersionFixture
from app.storage import Storage,ProjectError
from app.training import TrainingRepository,TARGET
from app.versions import encoded
from app.server import create_server


class TrainingTests(VersionFixture):
    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions)

    def body(self,**changes):
        return {'name':'首次训练','version_id':self.vid,'target_id':TARGET['id'],'epochs':1,'idempotency_key':uuid.uuid4().hex,**changes}

    def create(self,body=None):return self.training.create(self.pid,self.a,body or self.body())

    def test_submit_is_frozen_persistent_and_does_not_start_compute(self):
        before=self.versions.read(self.pid,self.a,self.vid)
        job=self.create()['job'];snapshot=job['snapshot']
        self.assertEqual(job['state'],'QUEUED');self.assertEqual(job['queue_position'],1)
        self.assertEqual(snapshot['image_count'],len(before['manifest']['images']))
        self.assertEqual(snapshot['target']['gpu_selector'],'0');self.assertFalse(snapshot['target']['executor_connected'])
        self.assertEqual(hashlib.sha256(encoded(snapshot)).hexdigest(),job['snapshot_sha256'])
        self.repo.change_class(self.pid,self.a,{'name':'工作区改类','revision':1},self.cids[0])
        restored=TrainingRepository(Storage(self.path)).get(self.pid,self.a,job['id'])
        self.assertEqual(job,restored);self.assertEqual(before,self.versions.read(self.pid,self.a,self.vid))
        self.assertFalse((self.path.parent/'runs').exists())

    def test_same_submission_reused_and_changed_content_conflicts(self):
        body=self.body();first=self.create(body);again=self.create(body)
        self.assertFalse(first['reused']);self.assertTrue(again['reused']);self.assertEqual(first['job'],again['job'])
        for change in ({'epochs':2},{'name':'其他训练'}):
            with self.assertRaises(ProjectError) as e:self.create({**body,**change})
            self.assertEqual(e.exception.code,'IDEMPOTENCY_CONFLICT')
        self.assertEqual(len(self.training.list(self.pid,self.a)),1)

    def test_concurrent_duplicate_requests_commit_once(self):
        body=self.body();barrier=threading.Barrier(2)
        # The shared archive semaphore deliberately rejects parallel packaging. Use independent repositories,
        # as on a restarted server, to exercise transaction-level idempotency after both checks complete.
        second=TrainingRepository(self.store)
        def checked(original):
            def call(*args):
                result=original(*args);barrier.wait(timeout=10);return result
            return call
        with patch.object(self.versions,'readiness',checked(self.versions.readiness)),patch.object(second.versions,'readiness',checked(second.versions.readiness)):
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(repo.create,self.pid,self.a,body) for repo in (self.training,second)]
                results=[f.result(timeout=20) for f in futures]
        self.assertEqual({r['job']['id'] for r in results},{results[0]['job']['id']})
        self.assertEqual(sum(not r['reused'] for r in results),1)

    def test_fifo_cancellation_is_idempotent_and_recreation_is_new(self):
        first=self.create()['job'];second=self.create()['job']
        self.assertEqual(second['queue_position'],2)
        done=self.training.cancel(self.pid,self.a,first['id'])
        self.assertEqual(done['state'],'CANCELLED');self.assertIsNone(done['queue_position'])
        self.assertEqual(done,self.training.cancel(self.pid,self.a,first['id']))
        self.assertEqual(self.training.get(self.pid,self.a,second['id'])['queue_position'],1)
        third=self.create()['job'];self.assertNotEqual(third['id'],first['id']);self.assertEqual(third['queue_position'],2)

    def test_scope_rejects_other_task_version_or_job(self):
        job=self.create()['job']
        for fn in (lambda:self.training.get(self.pid,self.b,job['id']),lambda:self.training.cancel(self.pid,self.b,job['id']),
                   lambda:self.training.create(self.pid,self.b,self.body())):
            with self.assertRaises(ProjectError) as e:fn()
            self.assertEqual(e.exception.status,404)

    def test_request_validation_no_arbitrary_paths_or_execution_flags(self):
        for change in ({'epochs':True},{'epochs':0},{'epochs':301},{'epochs':1.5},{'target_id':'self-hosted'},
                       {'idempotency_key':'bad'},{'name':'\n'},{'name':'x'*61},{'weights':'/tmp/a.pt'},{'device':'cpu'},
                       {'parameters':{'epochs':1}},{'version_id':'../../a'}):
            with self.subTest(change=change),self.assertRaises(ProjectError) as e:self.create(self.body(**change))
            self.assertEqual(e.exception.status,422)
        self.assertEqual(self.training.list(self.pid,self.a),[])

    def test_damaged_or_failed_data_check_never_creates_request(self):
        with patch.object(self.versions,'readiness',return_value={'passed':False}):
            with self.assertRaises(ProjectError):self.create()
        manifest=self.versions.read(self.pid,self.a,self.vid)['manifest']
        path=self.versions.root/self.vid/manifest['images'][0]['label_path'];path.write_text('damaged',encoding='utf-8')
        with self.assertRaises(ProjectError):self.create()
        self.assertEqual(self.training.list(self.pid,self.a),[])

    def test_mismatched_check_cannot_be_committed(self):
        report=self.versions.readiness(self.pid,self.a,self.vid);report['manifest_sha256']='0'*64
        with patch.object(self.versions,'readiness',return_value=report),self.assertRaises(ProjectError) as e:self.create()
        self.assertEqual(e.exception.code,'TRAINING_DATA_CHANGED');self.assertEqual(self.training.list(self.pid,self.a),[])

    def test_database_protects_snapshot_and_terminal_status(self):
        job=self.create()['job']
        for assignment in ("snapshot_json='{}'","state='RUNNING'","name='rewritten'","task_id='other'"):
            with self.assertRaises(sqlite3.IntegrityError),self.store.connect() as db:db.execute('UPDATE training_jobs SET '+assignment+' WHERE id=?',(job['id'],))
        with self.assertRaises(sqlite3.IntegrityError),self.store.connect() as db:db.execute('DELETE FROM training_jobs WHERE id=?',(job['id'],))
        self.training.cancel(self.pid,self.a,job['id'])
        with self.assertRaises(sqlite3.IntegrityError),self.store.connect() as db:db.execute("UPDATE training_jobs SET state='QUEUED' WHERE id=?",(job['id'],))

    def test_v8_migration_backup_and_rollback_preserve_existing_rows(self):
        with self.store.connect() as db:
            __import__('schema_utils').drop_evaluations(db);db.execute('DROP TABLE training_record_visibility');db.execute('DROP TABLE model_events');db.execute('DROP TABLE model_versions');db.execute('DROP TABLE training_remote_sessions');db.execute('DROP TABLE training_dispatches');db.execute('DROP TABLE training_jobs');db.execute('DROP INDEX version_task_identity');db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=8')
            before=[tuple(r) for r in db.execute('SELECT * FROM task_data_versions')]
        class BrokenStorage(Storage):
            @staticmethod
            def create_training_tables(db):Storage.create_training_tables(db);raise RuntimeError('injected')
        with self.assertRaises(RuntimeError):BrokenStorage(self.path)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],8)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='training_jobs'").fetchone())
        restored=Storage(self.path)
        with restored.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16)
            self.assertEqual([tuple(r) for r in db.execute('SELECT * FROM task_data_versions')],before)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(),[])
        self.assertTrue(list((self.path.parent/'backups').glob('schema-v8-*.db')))

    def test_http_contract_snapshot_download_and_cross_origin_rejection(self):
        server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        origin=f'http://127.0.0.1:{server.server_port}';base=f'/api/projects/{self.pid}/tasks/{self.a}/training'
        def fetch(path,body=None,extra=None):
            return urlopen(Request(origin+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json',**(extra or {})}),timeout=15)
        try:
            body=self.body()
            with fetch(base,body) as response:self.assertEqual(response.status,201);job=json.load(response)['job']
            with fetch(base,body) as response:self.assertEqual(response.status,200);self.assertTrue(json.load(response)['reused'])
            with fetch(base+'/'+job['id']+'/snapshot') as response:
                self.assertIn('attachment',response.headers['Content-Disposition']);self.assertEqual(json.load(response)['snapshot'],job['snapshot'])
            with self.assertRaises(HTTPError) as e:fetch(base,body,{'Origin':'https://other.example'})
            self.assertEqual(e.exception.code,403)
            with fetch(base+'/'+job['id']+'/cancel',{}) as response:self.assertEqual(json.load(response)['job']['state'],'CANCELLED')
            with fetch('/api/training-options') as response:self.assertFalse(json.load(response)['executor_connected'])
        finally:server.shutdown();server.server_close();thread.join()
