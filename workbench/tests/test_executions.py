import contextlib
import hashlib
import io
import json
import sqlite3
import sys
import threading
import unittest
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from test_versions import VersionFixture
from app.storage import Storage,ProjectError
from app.training import TrainingRepository,TARGET
from app.executions import ExecutionRepository
from app.versions import encoded
from app.server import create_server
from tools import run_training_job as runner


class ExecutionTests(VersionFixture):
    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions);self.executions=ExecutionRepository(self.store,self.training)

    def job(self):
        return self.training.create(self.pid,self.a,{'name':'CPU fixture — never GPU execution','version_id':self.vid,'target_id':TARGET['id'],'epochs':1,'idempotency_key':uuid.uuid4().hex})['job']

    def dispatch(self,job):return self.executions.dispatch(self.pid,self.a,job['id'])['job']

    def request(self,job):
        with self.store.connect() as db:return json.loads(db.execute('SELECT request_json FROM training_dispatches WHERE job_id=?',(job['id'],)).fetchone()[0])

    def receipt(self,job,status='completed',change=None,extra=None):
        r=self.request(job);params=r['snapshot']['requested_parameters']
        report={'status':status,'training_executed':status=='completed','forward_passed':True,'requested_parameters':params,
                'weights_sha256':TARGET['weights']['sha256'],'epochs_completed':1,'batches_completed':7,
                'environment':{'ready':True,'cuda_visible_devices':'0','gpu':{'logical_index':0},'nvidia_smi':TARGET['verified_gpu_uuid']},
                'reload_passed':{'best.pt':True,'last.pt':True},'completion_reason':'completed_epochs'}
        report.update(actual_parameters=params,actual_optimizer='CPU-test-fixture',lock_check_passed=True)
        files={'console.log':b'CPU contract fixture; no training took place.','args.yaml':b'fixture: true','results.csv':b'epoch,loss\n1,1.0\n',
               'best.pt':b'CPU-test placeholder; not a model','last.pt':b'CPU-test placeholder; not a model'}
        receipt={k:r[k] for k in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256','return_token')}
        receipt.update(format_version=1,report=report,artifacts={k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in files.items()})
        if change:change(receipt)
        if extra:files.update(extra)
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('receipt.json',encoded(receipt))
            for name,data in files.items():archive.writestr(name,data)
        return output.getvalue()

    def unpack(self,job):
        path,_,_=self.executions.package(self.pid,self.a,job['id']);folder=self.path.parent/'remote';folder.mkdir()
        with zipfile.ZipFile(path) as archive:archive.extractall(folder)
        return folder

    def test_interruption_preserves_durable_completed_progress_without_guessing_partial_epoch(self):
        folder=self.path.parent/'stopped-run';folder.mkdir()
        (folder/'progress.json').write_text(json.dumps({'stage':'training','epochs_completed':2,'batches_completed':26}),encoding='utf-8')
        counters=runner.interrupted_progress(folder,10)
        self.assertEqual((counters['epochs_completed'],counters['batches_completed']),(2,26));self.assertTrue(counters['progress_is_partial'])
        (folder/'progress.json').write_text(json.dumps({'epochs_completed':True,'batches_completed':999}),encoding='utf-8')
        self.assertEqual(runner.interrupted_progress(folder,10)['progress_source'],'unknown')

    def test_package_is_frozen_bound_and_repeat_dispatch_reuses_it(self):
        job=self.job();result=self.dispatch(job);self.assertEqual(result['state'],'DISPATCHED')
        path,_,_=self.executions.package(self.pid,self.a,job['id']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest,result['dispatch']['package_sha256'])
        self.assertNotIn('return_token',json.dumps(result))
        self.assertTrue(self.executions.dispatch(self.pid,self.a,job['id'])['reused'])
        folder=self.unpack(job);request,report=runner.check_input(folder)
        self.assertTrue(report['passed']);self.assertEqual(request['snapshot_sha256'],job['snapshot_sha256']);self.assertEqual(report['version_id'],self.vid)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)

    def test_fifo_and_persistent_reservation_block_later_job_and_local_cancel(self):
        a,b=self.job(),self.job()
        with self.assertRaises(ProjectError) as error:self.dispatch(b)
        self.assertEqual(error.exception.code,'QUEUE_ORDER')
        self.dispatch(a)
        restored=ExecutionRepository(Storage(self.path),TrainingRepository(self.store))
        with self.assertRaises(ProjectError) as error:restored.dispatch(self.pid,self.a,b['id'])
        self.assertEqual(error.exception.code,'TARGET_RESERVED')
        with self.assertRaises(ProjectError):self.training.cancel(self.pid,self.a,a['id'])
        self.assertEqual(self.training.get(self.pid,self.a,a['id'])['state'],'DISPATCHED')

    def test_concurrent_dispatch_of_same_job_keeps_one_package_and_reservation(self):
        job=self.job();other=ExecutionRepository(self.store,TrainingRepository(self.store))
        with ThreadPoolExecutor(2) as pool:
            results=list(pool.map(lambda repo:repo.dispatch(self.pid,self.a,job['id']),[self.executions,other]))
        self.assertEqual(sum(not r['reused'] for r in results),1)
        self.assertEqual({r['job']['dispatch']['id'] for r in results},{results[0]['job']['dispatch']['id']})

    def test_result_contract_commits_files_releases_target_and_is_idempotent(self):
        job=self.job();self.dispatch(job);raw=self.receipt(job)
        result=self.executions.receive(self.pid,self.a,job['id'],raw)
        self.assertEqual(result['job']['state'],'SUCCEEDED');self.assertFalse(result['reused'])
        self.assertNotIn('return_token',json.dumps(result))
        self.assertTrue(self.executions.receive(self.pid,self.a,job['id'],raw)['reused'])
        for name in ('best.pt','last.pt','console.log'):
            path,_,_=self.executions.artifact(self.pid,self.a,job['id'],name);self.assertTrue(path.is_file())
        self.assertEqual(self.dispatch(self.job())['state'],'DISPATCHED')
        with self.assertRaises(ProjectError) as error:self.executions.receive(self.pid,self.a,job['id'],self.receipt(job,status='failed'))
        self.assertEqual(error.exception.code,'RESULT_ALREADY_IMPORTED')

    def test_wrong_identity_token_config_gpu_or_missing_weight_keeps_reservation(self):
        job=self.job();self.dispatch(job)
        changes=[lambda r:r.update(job_id='0'*32),lambda r:r.update(return_token='wrong'),
                 lambda r:r.update(snapshot_sha256='0'*64),lambda r:r['report'].update(requested_parameters={'epochs':300}),
                 lambda r:r['report']['environment'].update(nvidia_smi='other GPU'),
                 lambda r:r['report']['reload_passed'].update({'best.pt':False}),
                 lambda r:r['report'].update(training_executed=False),lambda r:r['artifacts']['last.pt'].update(sha256='0'*64)]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ProjectError) as e:self.executions.receive(self.pid,self.a,job['id'],self.receipt(job,change=change))
            self.assertEqual(e.exception.code,'INVALID_TRAINING_RESULT')
        self.assertEqual(self.training.get(self.pid,self.a,job['id'])['state'],'DISPATCHED')
        self.assertFalse(self.executions.owned(job['id'],'returned').exists())

    def test_zip_traversal_unlisted_file_duplicates_and_invalid_zip_rejected(self):
        job=self.job();self.dispatch(job)
        for raw in (b'not zip',self.receipt(job,extra={'../escape':b'x'}),self.receipt(job,extra={'hidden.exe':b'x'})):
            with self.assertRaises(ProjectError):self.executions.receive(self.pid,self.a,job['id'],raw)
        raw=self.receipt(job);out=io.BytesIO(raw)
        with zipfile.ZipFile(out,'a') as z:z.writestr('console.log',b'duplicate')
        with self.assertRaises(ProjectError):self.executions.receive(self.pid,self.a,job['id'],out.getvalue())

    def test_failure_and_interruption_are_terminal_without_formal_model(self):
        for status,state in [('failed','FAILED'),('interrupted','INTERRUPTED')]:
            job=self.job();self.dispatch(job);raw=self.receipt(job,status=status)
            result=self.executions.receive(self.pid,self.a,job['id'],raw)
            self.assertEqual(result['job']['state'],state)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM model_versions').fetchone()[0],0)

    def test_tampered_package_or_artifact_cannot_be_downloaded(self):
        job=self.job();self.dispatch(job);package,_,_=self.executions.package(self.pid,self.a,job['id']);package.write_bytes(b'bad')
        with self.assertRaises(ProjectError):self.executions.package(self.pid,self.a,job['id'])
        self.executions.receive(self.pid,self.a,job['id'],self.receipt(job))
        file,_,_=self.executions.artifact(self.pid,self.a,job['id'],'last.pt');file.write_bytes(b'bad')
        with self.assertRaises(ProjectError):self.executions.artifact(self.pid,self.a,job['id'],'last.pt')

    def test_prepare_only_never_imports_torch_and_tampered_input_cannot_pass(self):
        job=self.job();self.dispatch(job);folder=self.unpack(job);original=__import__
        def guard(name,*args,**kwargs):
            if name.startswith(('torch','ultralytics')):raise AssertionError('GPU import on control side')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',guard),patch.object(sys,'argv',['run_training_job.py','--directory',str(folder),'--prepare-only']),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(runner.main(),0)
        self.assertFalse((folder/'started.json').exists());self.assertFalse((folder/'return-result.zip').exists())
        (folder/'dataset.zip').write_bytes(b'changed')
        with self.assertRaises(ValueError):runner.check_input(folder)

    def test_failed_executor_generates_receipt_and_never_runs_twice(self):
        job=self.job();self.dispatch(job);folder=self.unpack(job)
        with (patch.object(runner,'device_lock',return_value=contextlib.nullcontext()),patch.object(runner,'verify_lock'),patch.object(runner,'registry_path',return_value=folder/'registry.json'),patch.object(runner,'train',side_effect=RuntimeError('CPU injected failure')) as train,
             patch.object(sys,'argv',['run_training_job.py','--directory',str(folder)]),contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(runner.main(),2);self.assertEqual(runner.main(),0);self.assertEqual(train.call_count,1)
        result=self.executions.receive(self.pid,self.a,job['id'],(folder/'return-result.zip').read_bytes())
        self.assertEqual(result['job']['state'],'FAILED')

    def test_recovery_has_no_training_and_produces_interrupted_result(self):
        job=self.job();self.dispatch(job);folder=self.unpack(job)
        with (patch.object(runner,'device_lock',return_value=contextlib.nullcontext()),patch.object(runner,'verify_lock'),patch.object(runner,'registry_path',return_value=folder/'registry.json'),patch.object(runner,'train',side_effect=AssertionError('Must not train')),
             patch.object(sys,'argv',['run_training_job.py','--directory',str(folder),'--recover-interrupted']),contextlib.redirect_stdout(io.StringIO())):self.assertEqual(runner.main(),2)
        result=self.executions.receive(self.pid,self.a,job['id'],(folder/'return-result.zip').read_bytes())
        self.assertEqual(result['job']['state'],'INTERRUPTED')

    def test_scope_is_checked_for_dispatch_import_and_download(self):
        job=self.job();self.dispatch(job)
        for method,args in [(self.executions.dispatch,()),(self.executions.package,()),(self.executions.receive,(b'zip',)),(self.executions.artifact,('console.log',))]:
            with self.assertRaises(ProjectError) as e:method(self.pid,self.b,job['id'],*args)
            self.assertEqual(e.exception.status,404)

    def test_lock_verification_requires_the_other_process_to_be_blocked(self):
        from types import SimpleNamespace
        with patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=3)),contextlib.redirect_stdout(io.StringIO()):runner.verify_lock(self.path.parent)
        with patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=0)),self.assertRaises(RuntimeError):runner.verify_lock(self.path.parent)

    def test_copying_package_directory_does_not_restart_registered_dispatch(self):
        job=self.job();self.dispatch(job);folder=self.unpack(job);registry=folder/'registry.json'
        registry.write_text(json.dumps({'directory':'/previous/run'}),encoding='utf-8')
        with (patch.object(runner,'device_lock',return_value=contextlib.nullcontext()),patch.object(runner,'registry_path',return_value=registry),
              patch.object(runner,'train',side_effect=AssertionError('Must not replay')),
              patch.object(sys,'argv',['run_training_job.py','--directory',str(folder)]),contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(runner.main(),2)
        self.assertFalse((folder/'started.json').exists())

    def test_http_binary_upload_download_and_oversize_are_checked(self):
        job=self.job();server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        origin=f'http://127.0.0.1:{server.server_port}';base=f'/api/projects/{self.pid}/tasks/{self.a}/training/{job["id"]}'
        def fetch(path,raw=None,mime='application/json',headers=None):return urlopen(Request(origin+path,data=raw,headers={'Content-Type':mime,**(headers or {})}),timeout=15)
        try:
            with fetch(base+'/dispatch',b'{}') as response:self.assertEqual(response.status,201)
            with fetch(base+'/package') as response:self.assertTrue(response.read().startswith(b'PK'));self.assertIn('attachment',response.headers['Content-Disposition'])
            with self.assertRaises(HTTPError) as e:fetch(base+'/result',b'{}')
            self.assertEqual(e.exception.code,415)
            with self.assertRaises(HTTPError) as e:fetch(base+'/result',b'','application/zip',{'Content-Length':str(128*1024**2+1)})
            self.assertEqual(e.exception.code,413)
            raw=self.receipt(job)
            with fetch(base+'/result',raw,'application/zip') as response:self.assertEqual(json.load(response)['job']['state'],'SUCCEEDED')
            with fetch(base+'/artifacts/last.pt') as response:self.assertIn(b'CPU-test',response.read())
        finally:server.shutdown();server.server_close();thread.join()

    def test_v9_upgrade_rollback_and_preservation(self):
        job=self.job()
        with self.store.connect() as db:
            __import__('schema_utils').drop_evaluations(db);db.execute('DROP TABLE training_record_visibility');db.execute('DROP TABLE model_events');db.execute('DROP TABLE model_versions');db.execute('DROP TABLE training_remote_sessions');db.execute('DROP TABLE training_dispatches');db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=9')
            before=[tuple(r) for r in db.execute('SELECT * FROM training_jobs')]
        class Broken(Storage):
            def upgrade_training_v10(self,db):super().upgrade_training_v10(db);raise RuntimeError('injected')
        with self.assertRaises(RuntimeError):Broken(self.path)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],9)
            self.assertEqual([tuple(r) for r in db.execute('SELECT * FROM training_jobs')],before)
        restored=Storage(self.path)
        with restored.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16)
            self.assertEqual([tuple(r) for r in db.execute('SELECT * FROM training_jobs')],before)
            self.assertFalse(db.execute('PRAGMA foreign_key_check').fetchall())
        self.assertTrue(list((self.path.parent/'backups').glob('schema-v9-*.db')))
