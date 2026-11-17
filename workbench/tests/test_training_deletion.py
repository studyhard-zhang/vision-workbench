import hashlib,json,sqlite3,threading,uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from test_versions import VersionFixture
import test_executions as contract
from app.storage import Storage,ProjectError
from app.training import TrainingRepository
from app.executions import ExecutionRepository
from app.models import ModelRepository
from app.server import create_server


class TrainingDeletionTests(VersionFixture):
    job=contract.ExecutionTests.job
    receipt=contract.ExecutionTests.receipt
    request=contract.ExecutionTests.request

    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions);self.executions=ExecutionRepository(self.store,self.training);self.models=ModelRepository(self.store,self.executions)

    def selection(self,job):return {'id':job['id'],'confirmation_name':job['name'],'visibility_revision':job['visibility_revision']}
    def remove(self,job):return self.training.change_visibility(self.pid,self.a,[self.selection(job)])
    def success(self):
        job=self.job();self.executions.dispatch(self.pid,self.a,job['id']);self.executions.receive(self.pid,self.a,job['id'],self.receipt(job));return self.training.get(self.pid,self.a,job['id'])

    def test_delete_waiting_atomically_cancels_and_restore_never_requeues(self):
        job=self.job();self.assertEqual(self.remove(job)['changed'],1);self.assertEqual(self.training.list(self.pid,self.a),[])
        trashed=self.training.list(self.pid,self.a,'trash')[0];self.assertEqual(trashed['state'],'CANCELLED')
        self.assertEqual(trashed['snapshot'],job['snapshot']);self.assertEqual(self.remove(job)['changed'],0)
        self.training.change_visibility(self.pid,self.a,[self.selection(trashed)],True)
        restored=self.training.list(self.pid,self.a)[0];self.assertEqual(restored['state'],'CANCELLED')
        with self.assertRaises(ProjectError) as e:self.remove(job)
        self.assertEqual(e.exception.code,'TRAINING_VISIBILITY_CHANGED')

    def test_batch_rejects_active_and_cross_task_without_partial_deletion(self):
        ended=self.job();self.training.cancel(self.pid,self.a,ended['id']);active=self.job();self.executions.dispatch(self.pid,self.a,active['id'])
        with self.assertRaises(ProjectError) as e:self.training.change_visibility(self.pid,self.a,[self.selection(ended),self.selection(active)])
        self.assertEqual(e.exception.code,'TRAINING_ACTIVE');self.assertEqual(len(self.training.list(self.pid,self.a)),2)
        with self.assertRaises(ProjectError):self.training.change_visibility(self.pid,self.b,[self.selection(ended)])
        self.assertEqual(self.training.list(self.pid,self.a,'trash'),[])

    def test_concurrent_dispatch_and_deletion_never_hide_an_active_run(self):
        job=self.job();barrier=threading.Barrier(2)
        def perform(operation):
            barrier.wait()
            try:operation();return 'ok'
            except ProjectError as exc:return exc.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(perform,lambda:self.remove(job)),pool.submit(perform,lambda:self.executions.dispatch(self.pid,self.a,job['id']))]
            results=[f.result() for f in futures]
        current=self.training.get(self.pid,self.a,job['id']);self.assertEqual(results.count('ok'),1)
        if current['deleted_at']:self.assertEqual(current['state'],'CANCELLED');self.assertIsNone(current['dispatch'])
        else:self.assertEqual(current['state'],'DISPATCHED');self.assertEqual(current['dispatch']['state'],'AWAITING_RESULT')

    def test_model_source_files_evidence_and_registration_survive_list_deletion(self):
        job=self.success();model=self.models.register(self.pid,self.a,job['id'])['model'];path=self.executions.owned(job['id'],'returned','best.pt');sha=hashlib.sha256(path.read_bytes()).hexdigest()
        self.remove(job)
        self.assertEqual(self.models.get(self.pid,self.a,model['id'])['training_run_id'],job['id'])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),sha);self.assertTrue(self.models.register(self.pid,self.a,job['id'])['reused'])
        with self.store.connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):db.execute('DELETE FROM training_jobs WHERE id=?',(job['id'],))

    def test_unregistered_trashed_run_must_be_restored_before_model_registration(self):
        job=self.success();self.remove(job)
        with self.assertRaises(ProjectError) as e:self.models.register(self.pid,self.a,job['id'])
        self.assertEqual(e.exception.code,'TRAINING_IN_TRASH')

    def test_validation_capability_not_removed_with_training_list_entry(self):
        from app.training import TARGET
        import uuid
        body={'name':'configuration evidence','version_id':self.vid,'target_id':TARGET['id'],'epochs':1,'idempotency_key':uuid.uuid4().hex,'configuration':{'profile':'geometry640'}}
        job=self.training.create(self.pid,self.a,body,allow_validation=True)['job'];self.executions.dispatch(self.pid,self.a,job['id'])
        self.executions.receive(self.pid,self.a,job['id'],self.receipt(job,change=lambda r:r['report'].update(geometry_check_passed=True)))
        self.assertTrue(next(p for p in self.training.options()['configuration_profiles'] if p['id']=='geometry640')['enabled']);self.remove(job)
        self.assertTrue(next(p for p in self.training.options()['configuration_profiles'] if p['id']=='geometry640')['enabled'])

    def test_schema11_upgrade_backup_rollback_and_idempotent_open(self):
        job=self.job()
        with self.store.connect() as db:
            __import__('schema_utils').drop_evaluations(db);db.execute('DROP TABLE training_record_visibility');db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=11')
        class Broken(Storage):
            @staticmethod
            def create_training_visibility(db):Storage.create_training_visibility(db);raise RuntimeError('injected failure')
        with self.assertRaises(RuntimeError):Broken(self.path)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],11)
        restored=Storage(self.path);again=Storage(self.path)
        with again.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16);self.assertEqual(db.execute('SELECT id FROM training_jobs').fetchone()[0],job['id']);self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(),[])
        self.assertTrue(list((self.path.parent/'backups').glob('schema-v11-*.db')))

    def test_http_delete_restore_batch_and_unknown_fields(self):
        job=self.job();server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        origin=f'http://127.0.0.1:{server.server_port}';base=f'/api/projects/{self.pid}/tasks/{self.a}/training'
        def call(path,body=None,method=None,headers=None):
            request=Request(origin+path,data=None if body is None else json.dumps(body).encode(),method=method,headers={'Content-Type':'application/json',**(headers or {})})
            try:response=urlopen(request,timeout=15)
            except HTTPError as e:response=e
            with response:return response.status,json.load(response)
        try:
            selection=self.selection(job);body={k:v for k,v in selection.items() if k!='id'}
            self.assertEqual(call(base+'/'+job['id'],body,'DELETE',{'Origin':'https://other.invalid'})[0],403)
            self.assertEqual(call(base+'/'+job['id'],body,'DELETE')[0],200)
            self.assertEqual(call(base)[1]['total'],0);trashed=call(base+'?view=trash')[1]['jobs'][0]
            self.assertEqual(call(base+'/'+job['id']+'/restore',{k:v for k,v in self.selection(trashed).items() if k!='id'},'POST')[0],200)
            self.assertEqual(call(base)[1]['jobs'][0]['state'],'CANCELLED')
            self.assertEqual(call(base+'/trash',{'records':[self.selection(call(base)[1]['jobs'][0])]},'POST')[0],200)
            self.assertEqual(call(base+'?view=all')[0],422)
            self.assertEqual(call(base+'/trash',{'records':[],'force':True},'POST')[0],422)
            # Real upper-bound request, including JSON-escaped supplementary Unicode names.
            selections=[]
            with self.store.connect() as db:
                source=dict(db.execute('SELECT * FROM training_jobs WHERE id=?',(job['id'],)).fetchone());source.pop('sequence')
                for _ in range(200):
                    source.update(id=uuid.uuid4().hex,idempotency_key=uuid.uuid4().hex,name='😀'*60,state='CANCELLED')
                    db.execute('INSERT INTO training_jobs ('+','.join(source)+') VALUES ('+','.join('?' for _ in source)+')',tuple(source.values()))
                    selections.append({'id':source['id'],'confirmation_name':source['name'],'visibility_revision':0})
            self.assertGreater(len(json.dumps({'records':selections}).encode()),32768)
            status,data=call(base+'/trash',{'records':selections},'POST');self.assertEqual(status,200);self.assertEqual(data['changed'],200)
        finally:server.shutdown();server.server_close();thread.join(3)
