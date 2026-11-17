"""Independent evaluation contracts. CPU fixtures never import a GPU runtime."""
import copy
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
from urllib.error import HTTPError
from urllib.request import Request,urlopen

import test_executions as fixtures
from test_versions import VersionFixture
from app.storage import Storage,ProjectError
from app.training import TrainingRepository,TARGET
from app.executions import ExecutionRepository
from app.models import ModelRepository
from app.evaluations import EvaluationRepository,EvaluationExecution,METRICS
from app.remote import RemoteRepository
from app.versions import encoded
from app.server import create_server
from tools import run_evaluation_job as runner,ssh_control
from schema_utils import drop_evaluations


class EvaluationTests(VersionFixture):
    job=fixtures.ExecutionTests.job
    dispatch=fixtures.ExecutionTests.dispatch
    request=fixtures.ExecutionTests.request
    receipt=fixtures.ExecutionTests.receipt

    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id']
        self.training=TrainingRepository(self.store,self.versions);self.executions=ExecutionRepository(self.store,self.training)
        self.models=ModelRepository(self.store,self.executions)
        job=self.job();self.dispatch(job);self.executions.receive(self.pid,self.a,job['id'],self.receipt(job))
        self.model=self.models.register(self.pid,self.a,job['id'])['model']
        self.evaluations=EvaluationRepository(self.store,self.models);self.eexec=EvaluationExecution(self.store,self.evaluations)

    def create(self,**values):
        return self.evaluations.create(self.pid,self.a,{'model_id':self.model['id'],'subset':'val','idempotency_key':uuid.uuid4().hex,**values})['job']

    def erequest(self,job):
        with self.store.connect() as db:return json.loads(db.execute('SELECT request_json FROM evaluation_dispatches WHERE job_id=?',(job['id'],)).fetchone()[0])

    def detail_report(self,job):
        snap=job['snapshot'];data=self.versions.read(self.pid,self.a,self.vid)['manifest'];images=[i for i in data['images'] if i['subset']==snap['subset']]
        counts=[sum(b['class_id']==c['id'] for i in images for b in i['boxes']) for c in snap['classes']]
        return {'format_version':1,'runtime':'8.3.241','snapshot':snap,'sample_count':len(images),'metrics':dict.fromkeys(METRICS,0.5),
                'per_class':[{'index':i,'name':c['name'],'instances':counts[i],'metrics':dict.fromkeys(METRICS,0.5 if counts[i] else None)} for i,c in enumerate(snap['classes'])],
                'images':[{'id':i['id'],'width':i['width'],'height':i['height'],'predictions':[]} for i in images],
                'metric_semantics':{'source':'CPU TEST ONLY; not real model scores'},'missing':['CPU fixture: no runtime plots'],
                'timing':{'scope':'CPU fixture'},'confusion_matrix':[[0]*(len(counts)+1) for _ in range(len(counts)+1)]}

    def eresult(self,job,status='completed',detail=None,extra_files=None):
        req=self.erequest(job);report={'status':status,'training_executed':False,'evaluation_executed':status=='completed','lock_check_passed':True,'weights_sha256':req['weights_sha256'],
            'environment':{'ready':True,'nvidia_smi':TARGET['verified_gpu_uuid'],'cuda_visible_devices':'0'},'error':'','actual_parameters':job['snapshot']['requested_parameters']}
        files={'console.log':b'CPU fixture only'}
        files.update(extra_files or {})
        if status=='completed':files['report.json']=encoded(detail or self.detail_report(job))
        receipt={k:req[k] for k in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256','return_token')}
        receipt.update(format_version=1,kind='evaluation',report=report,artifacts={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()})
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr('receipt.json',encoded(receipt))
            for n,b in files.items():z.writestr(n,b)
        return stream.getvalue()

    def test_version_bound_idempotence_and_history_retention(self):
        key=uuid.uuid4().hex;values={'model_id':self.model['id'],'subset':'val','idempotency_key':key}
        first=self.evaluations.create(self.pid,self.a,values);again=self.evaluations.create(self.pid,self.a,values)
        self.assertEqual(first['job'],again['job']);self.assertTrue(again['reused'])
        snap=first['job']['snapshot'];self.assertEqual(snap['model_version_id'],self.model['id']);self.assertEqual(snap['data_version_id'],self.vid)
        self.assertEqual(snap['weights_sha256'],self.model['manifest']['artifacts']['best.pt']['sha256'])
        with self.assertRaises(ProjectError):self.evaluations.create(self.pid,self.a,{**values,'conf':0.25})
        self.evaluations.cancel(self.pid,self.a,first['job']['id'])
        self.assertEqual(self.evaluations.cancel(self.pid,self.a,first['job']['id'])['state'],'CANCELLED')
        for sql in ('DELETE FROM evaluation_jobs','UPDATE evaluation_jobs SET model_id=id','UPDATE evaluation_jobs SET state=\'QUEUED\''):
            with self.assertRaises(sqlite3.IntegrityError),self.store.connect() as db:db.execute(sql)

    def test_reject_empty_test_train_wrong_task_invalid_threshold_and_damaged_model(self):
        for values in ({'subset':'test'},{'subset':'train'},{'conf':True},{'conf':float('inf')},{'iou':0},{'unexpected':1}):
            with self.subTest(values=values),self.assertRaises(ProjectError):self.create(**values)
        with self.assertRaises(ProjectError):self.evaluations.options(self.pid,self.b,self.model['id'])
        self.models.owned(self.model['id'],'best.pt').write_bytes(b'damaged')
        with self.assertRaises(ProjectError):self.create()

    def test_package_is_cpu_preparable_and_pins_selected_weights(self):
        job=self.create();self.eexec.dispatch(self.pid,self.a,job['id']);path,_,_=self.eexec.package(self.pid,self.a,job['id'])
        directory=self.path.parent/'eval-runner';directory.mkdir()
        with zipfile.ZipFile(path) as z:z.extractall(directory)
        result=subprocess.run([sys.executable,str(directory/'run_training_job.py'),'--directory',str(directory),'--prepare-only'],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('"evaluation_executed": false',result.stdout)
        self.assertEqual(hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest(),self.model['manifest']['artifacts']['best.pt']['sha256'])
        self.assertFalse((directory/'started.json').exists());self.assertFalse((directory/'work').exists())
        (directory/'model.pt').write_bytes(b'wrong')
        with self.assertRaises(ValueError):runner.check_input(directory)

    def test_shared_reservation_both_directions_and_fifo(self):
        evaluation=self.create();training=self.job()
        with self.assertRaises(ProjectError) as e:self.dispatch(training)
        self.assertEqual(e.exception.code,'QUEUE_ORDER')
        self.eexec.dispatch(self.pid,self.a,evaluation['id'])
        with self.assertRaises(ProjectError) as e:self.dispatch(training)
        self.assertEqual(e.exception.code,'TARGET_RESERVED')
        self.eexec.receive(self.pid,self.a,evaluation['id'],self.eresult(evaluation,'interrupted'))
        self.dispatch(training);later=self.create()
        with self.assertRaises(ProjectError) as e:self.eexec.dispatch(self.pid,self.a,later['id'])
        self.assertEqual(e.exception.code,'TARGET_RESERVED')
        with self.store.connect() as db:
            request=encoded({}).decode()
            with self.assertRaises(sqlite3.IntegrityError):db.execute("INSERT INTO evaluation_dispatches VALUES(?,?,?,'AWAITING_RESULT',?,?,NULL,NULL,?,?)",(later['id'],uuid.uuid4().hex,TARGET['id'],request,'x','a','a'))

    def test_receive_idempotence_report_values_and_model_status_without_manifest_rewrite(self):
        before=self.models.owned(self.model['id'],'manifest.json').read_bytes();job=self.create();self.eexec.dispatch(self.pid,self.a,job['id']);raw=self.eresult(job)
        result=self.eexec.receive(self.pid,self.a,job['id'],raw);self.assertEqual(result['job']['state'],'SUCCEEDED')
        self.assertTrue(self.eexec.receive(self.pid,self.a,job['id'],raw)['reused'])
        report=self.eexec.report(self.pid,self.a,job['id']);self.assertEqual(report['metrics'],dict.fromkeys(METRICS,0.5));self.assertTrue(report['images'][0]['truth'])
        self.assertEqual(self.models.get(self.pid,self.a,self.model['id'])['evaluation_status'],'EVALUATED')
        self.assertEqual(self.models.owned(self.model['id'],'manifest.json').read_bytes(),before)
        with self.assertRaises(ProjectError):self.eexec.receive(self.pid,self.a,job['id'],self.eresult(job,'failed'))
        self.eexec.owned(job['id'],'returned','report.json').write_bytes(b'{}')
        with self.assertRaises(ProjectError):self.eexec.report(self.pid,self.a,job['id'])

    def test_untrusted_result_rejected_without_releasing_target(self):
        job=self.create();self.eexec.dispatch(self.pid,self.a,job['id']);good=self.detail_report(job)
        bads=[]
        for field,value in [('sample_count',999),('runtime','other')]:
            d=copy.deepcopy(good);d[field]=value;bads.append(d)
        d=copy.deepcopy(good);d['images']=[];bads.append(d)
        d=copy.deepcopy(good);d['metrics']['precision']=1.1;bads.append(d)
        d=copy.deepcopy(good);d['per_class'][0]['instances']+=1;bads.append(d)
        d=copy.deepcopy(good);d['images'][0]['predictions']=[{'class_index':0,'confidence':0.5,'xyxy':[0,0,10000,10000]}];bads.append(d)
        for d in bads:
            with self.assertRaises(ProjectError):self.eexec.receive(self.pid,self.a,job['id'],self.eresult(job,detail=d))
        self.assertEqual(self.evaluations.get(self.pid,self.a,job['id'])['dispatch']['state'],'AWAITING_RESULT')
        self.assertFalse(self.eexec.owned(job['id'],'returned').exists())

    def test_failed_evaluation_retry_never_creates_training_or_overwrites_failure(self):
        job=self.create();self.eexec.dispatch(self.pid,self.a,job['id']);self.eexec.receive(self.pid,self.a,job['id'],self.eresult(job,'failed'))
        retry=self.create();self.assertNotEqual(retry['id'],job['id']);self.assertEqual(self.evaluations.get(self.pid,self.a,job['id'])['state'],'FAILED')
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM training_jobs').fetchone()[0],1)
        self.assertEqual(self.models.get(self.pid,self.a,self.model['id'])['evaluation_status'],'NOT_EVALUATED')

    def test_remote_uses_separate_tables_and_requires_process_and_lock_release(self):
        job=self.create();remote=RemoteRepository(self.store,self.evaluations,self.eexec,kind='evaluation')
        fake=MagicMock();fake.endpoint='root@cpu-fixture:1';fake.call.return_value={'nvidia_smi':TARGET['verified_gpu_uuid']}
        remote.transport=lambda:fake;remote.start_worker=lambda:None;remote.submit(self.pid,self.a,job['id'])
        with self.store.connect() as db:
            row=dict(db.execute('SELECT s.*,t.project_id,j.task_id,d.package_sha256 FROM evaluation_remote_sessions s JOIN evaluation_jobs j ON j.id=s.job_id JOIN vision_tasks t ON t.id=j.task_id JOIN evaluation_dispatches d ON d.job_id=j.id').fetchone())
        raw=self.eresult(job);state={'directory_exists':True,'result_ready':True,'alive':True,'launch':{'pid':123},'progress':{},'resources':{'checked':True,'free':False},'result_sha256':hashlib.sha256(raw).hexdigest()}
        fake.call.return_value=state;remote.reconcile(row)
        self.assertEqual(self.evaluations.get(self.pid,self.a,job['id'])['remote']['phase'],'RELEASING');fake.copy.assert_not_called()
        with self.assertRaises(ProjectError):self.eexec.receive(self.pid,self.a,job['id'],raw)
        fake.call.return_value={**state,'alive':False,'resources':{'checked':True,'free':True}}
        fake.copy.side_effect=lambda path,*args:Path(path).write_bytes(raw)
        remote.reconcile(row);self.assertEqual(self.evaluations.get(self.pid,self.a,job['id'])['state'],'SUCCEEDED')

    def test_schema12_backup_rollback_preserves_original_rows(self):
        with self.store.connect() as db:
            drop_evaluations(db);db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=12');before=[tuple(r) for r in db.execute('SELECT * FROM model_versions')]
        class Broken(Storage):
            @staticmethod
            def create_evaluation_tables(db):Storage.create_evaluation_tables(db);raise RuntimeError('injected')
        with self.assertRaises(RuntimeError):Broken(self.path)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],12)
        restored=Storage(self.path)
        with restored.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16)
            self.assertEqual([tuple(r) for r in db.execute('SELECT * FROM model_versions')],before);self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(),[])
        self.assertTrue(list((self.path.parent/'backups').glob('schema-v12-*.db')))

    def test_http_create_options_cross_scope_cancel_and_static(self):
        server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();origin=f'http://127.0.0.1:{server.server_port}'
        base=f'/api/projects/{self.pid}/tasks/{self.a}/evaluations'
        def fetch(path,body=None):return urlopen(Request(origin+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=15)
        try:
            with fetch(base+'/options?model_id='+self.model['id']) as r:self.assertEqual(json.load(r)['subsets'][1]['count'],0)
            with fetch(base,{'model_id':self.model['id'],'subset':'val','idempotency_key':uuid.uuid4().hex}) as r:job=json.load(r)['job'];self.assertEqual(r.status,201)
            with fetch(base) as r:self.assertEqual(json.load(r)['total'],1)
            with fetch(base+'/'+job['id']+'/cancel',{}) as r:self.assertEqual(json.load(r)['job']['state'],'CANCELLED')
            with self.assertRaises(HTTPError) as error:fetch(base.replace(self.a,self.b)+'/'+job['id'])
            self.assertEqual(error.exception.code,404)
            with fetch('/evaluations.js') as r:self.assertEqual(r.status,200)
            with fetch('/api/health') as r:self.assertEqual(json.load(r)['schema'],16)
        finally:server.shutdown();server.server_close();thread.join()

    def test_http_all_runtime_plot_names_include_f1_and_match_saved_sha(self):
        from PIL import Image
        from app.evaluations import PLOTS
        image=io.BytesIO();Image.new('RGB',(2,2),'white').save(image,format='PNG');png=image.getvalue()
        job=self.create();self.eexec.dispatch(self.pid,self.a,job['id'])
        self.eexec.receive(self.pid,self.a,job['id'],self.eresult(job,extra_files={name:png for name in PLOTS}))
        server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}/api/projects/{self.pid}/tasks/{self.a}/evaluations/{job["id"]}'
        try:
            for name in PLOTS:
                with self.subTest(name=name),urlopen(base+'/artifacts/'+name,timeout=10) as response:
                    self.assertEqual(response.status,200);self.assertEqual(response.headers['Content-Type'],'image/png');self.assertEqual(hashlib.sha256(response.read()).hexdigest(),hashlib.sha256(png).hexdigest())
            with urlopen(base+'/report',timeout=10) as response:self.assertEqual(json.load(response)['sample_count'],job['snapshot']['sample_count'])
        finally:server.shutdown();server.server_close();thread.join()

    def test_ssh_install_evaluation_package_does_not_start_a_process(self):
        job=self.create();self.eexec.dispatch(self.pid,self.a,job['id']);path,_,_=self.eexec.package(self.pid,self.a,job['id']);root=self.path.parent/'remote-eval';directory=root/job['id'];directory.mkdir(parents=True)
        (directory/'execution.zip.upload').write_bytes(path.read_bytes())
        with patch.object(ssh_control,'ROOT',root),patch.object(ssh_control.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=TARGET['verified_gpu_uuid'])),patch.object(ssh_control.subprocess,'Popen') as launch:
            state=ssh_control.handle({'action':'install','job_id':job['id'],'expected_gpu_uuid':TARGET['verified_gpu_uuid'],'package_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        self.assertFalse(state['started']);launch.assert_not_called();self.assertTrue((directory/'model.pt').is_file())

    def test_initial_evaluation_is_queued_once_without_dispatch_or_gpu(self):
        first=self.evaluations.ensure_initial(self.pid,self.a,self.model)
        again=self.evaluations.ensure_initial(self.pid,self.a,self.model)
        self.assertEqual(first['job'],again['job']);self.assertTrue(again['reused'])
        self.assertIsNone(first['job']['dispatch']);self.assertIsNone(first['job']['remote'])
        self.assertEqual(self.evaluations.list(self.pid,self.a)[0]['state'],'QUEUED')

    def test_concurrent_creation_with_same_key_is_idempotent(self):
        values={'model_id':self.model['id'],'subset':'val','idempotency_key':uuid.uuid4().hex}
        with ThreadPoolExecutor(3) as pool:results=list(pool.map(lambda _:self.evaluations.create(self.pid,self.a,values),range(3)))
        self.assertEqual(len({r['job']['id'] for r in results}),1);self.assertEqual(sum(not r['reused'] for r in results),1)

    def test_result_io_failure_and_database_failure_retain_reservation_and_files(self):
        job=self.create();self.eexec.dispatch(self.pid,self.a,job['id']);raw=self.eresult(job)
        with patch('app.executions.Path.write_bytes',side_effect=OSError(28,'CPU injected full disk')),self.assertRaises(ProjectError):self.eexec.receive(self.pid,self.a,job['id'],raw)
        with self.store.connect() as db:db.execute("CREATE TRIGGER fail_eval_commit BEFORE UPDATE ON evaluation_dispatches BEGIN SELECT RAISE(ABORT,'CPU injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.eexec.receive(self.pid,self.a,job['id'],raw)
        self.assertEqual(self.evaluations.get(self.pid,self.a,job['id'])['dispatch']['state'],'AWAITING_RESULT');self.assertFalse(self.eexec.owned(job['id'],'returned').exists())
        with self.store.connect() as db:db.execute('DROP TRIGGER fail_eval_commit')
        self.assertEqual(self.eexec.receive(self.pid,self.a,job['id'],raw)['job']['state'],'SUCCEEDED')

    def test_runner_cpu_failure_finalization_and_recovery_never_train(self):
        from contextlib import nullcontext
        job=self.create();self.eexec.dispatch(self.pid,self.a,job['id']);path,_,_=self.eexec.package(self.pid,self.a,job['id']);directory=self.path.parent/'cpu-runner-recovery';directory.mkdir()
        with zipfile.ZipFile(path) as z:z.extractall(directory)
        registry=directory/'ledger.json'
        with patch.object(sys,'argv',['run_evaluation_job.py','--directory',str(directory)]),patch.object(runner.support,'device_lock',return_value=nullcontext()),patch.object(runner.support,'verify_lock'),patch.object(runner.support,'registry_path',return_value=registry),patch.object(runner,'evaluate',side_effect=RuntimeError('CPU injected CUDA out of memory')),patch.object(runner,'pack_result',side_effect=OSError(28,'CPU injected full disk')):
            self.assertEqual(runner.main(),2)
        self.assertFalse((directory/'return-result.zip').exists());self.assertEqual(json.loads((directory/'progress.json').read_text(encoding='utf-8'))['diagnostic']['code'],'STORAGE_FULL')
        with patch.object(sys,'argv',['run_evaluation_job.py','--directory',str(directory),'--recover-interrupted']),patch.object(runner.support,'device_lock',return_value=nullcontext()),patch.object(runner.support,'verify_lock'),patch.object(runner.support,'registry_path',return_value=registry),patch.object(runner,'evaluate') as execute:
            self.assertEqual(runner.main(),2);execute.assert_not_called()
        self.assertEqual(self.eexec.receive(self.pid,self.a,job['id'],(directory/'return-result.zip').read_bytes())['job']['state'],'INTERRUPTED')

    def test_current_model_choice_does_not_change_to_latest_registered_version(self):
        source=self.job();self.dispatch(source);self.executions.receive(self.pid,self.a,source['id'],self.receipt(source));old=self.create();new=self.models.register(self.pid,self.a,source['id'])['model']
        self.assertNotEqual(new['id'],old['model_id']);self.assertEqual(self.evaluations.get(self.pid,self.a,old['id'])['model_id'],self.model['id'])
        self.eexec.dispatch(self.pid,self.a,old['id']);self.assertEqual(self.erequest(old)['weights_sha256'],self.model['manifest']['artifacts']['best.pt']['sha256'])
