"""CPU-only inference contracts; fixture predictions never count as a real GPU run."""
import copy,hashlib,io,json,sqlite3,subprocess,sys,threading,types,uuid,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from urllib.parse import quote
from PIL import Image

from test_versions import VersionFixture
import test_executions as fixtures
from test_images import picture
from schema_utils import drop_inferences
from app.storage import Storage,ProjectError
from app.training import TrainingRepository,TARGET
from app.executions import ExecutionRepository
from app.models import ModelRepository
from app.evaluations import EvaluationRepository,EvaluationExecution
from app.inferences import InferenceInputRepository,InferenceRepository,InferenceExecution
from app.remote import RemoteRepository
from app.versions import encoded
from app.server import create_server
from tools import run_inference_job as runner


class InferenceFixture(VersionFixture):
    job=fixtures.ExecutionTests.job
    dispatch=fixtures.ExecutionTests.dispatch
    request=fixtures.ExecutionTests.request
    receipt=fixtures.ExecutionTests.receipt

    def setUp(self):
        super().setUp();self.vid=self.publish()['version']['id'];self.training=TrainingRepository(self.store,self.versions);self.executions=ExecutionRepository(self.store,self.training)
        source=self.job();self.dispatch(source);self.executions.receive(self.pid,self.a,source['id'],self.receipt(source))
        self.models=ModelRepository(self.store,self.executions);self.model=self.models.register(self.pid,self.a,source['id'])['model']
        self.infer_inputs=InferenceInputRepository(self.store,self.tasks);self.inferences=InferenceRepository(self.store,self.models,self.infer_inputs);self.iexec=InferenceExecution(self.store,self.inferences)
        self.object=self.infer_inputs.upload(self.pid,self.a,'object.png',picture(size=(80,40)))['input']
        self.empty=self.infer_inputs.upload(self.pid,self.a,'empty.jpg',picture('JPEG',color='black',size=(80,40)))['input']
        self.bad=self.infer_inputs.upload(self.pid,self.a,'broken.jpg',b'CPU deliberately corrupt JPEG') ['input']

    def create(self,**values):
        return self.inferences.create(self.pid,self.a,{'model_id':self.model['id'],'input_ids':[self.object['id'],self.empty['id'],self.bad['id']],'idempotency_key':uuid.uuid4().hex,**values})['job']

    def irequest(self,job):
        with self.store.connect() as db:return json.loads(db.execute('SELECT request_json FROM inference_dispatches WHERE job_id=?',(job['id'],)).fetchone()[0])

    def report(self,job):
        snap=job['snapshot'];rows=[]
        for image in snap['inputs']:
            invalid=image['status']=='INVALID'
            predictions=[] if invalid or image['filename'].startswith('empty') else [{'class_index':0,'class_name':snap['classes'][0]['name'],'confidence':0.9,'xyxy':[1,2,18,19]}]
            rows.append({'id':image['id'],'filename':image['filename'],'status':'FAILED' if invalid else 'SUCCEEDED','width':image.get('width'),'height':image.get('height'),
                         'error':image.get('error'),'error_code':image.get('error_code'),'predictions':predictions})
        return {'format_version':1,'runtime':'8.3.241','coordinate_system':'exif-oriented-pixels','complete':True,'snapshot':snap,'images':rows,'timing':{'scope':'CPU fixture only; never actual GPU speed'}}

    def iresult(self,job,status='completed',detail=None):
        req=self.irequest(job);files={'console.log':b'CPU fixture, no GPU'}
        if detail is not False:files['report.json']=encoded(detail or self.report(job))
        report={'status':status,'training_executed':False,'evaluation_executed':False,'inference_executed':status=='completed','lock_check_passed':True,'weights_sha256':req['weights_sha256'],
                'actual_parameters':job['snapshot']['requested_parameters'],'environment':{'ready':True,'nvidia_smi':TARGET['verified_gpu_uuid'],'cuda_visible_devices':'0'}}
        receipt={k:req[k] for k in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256','return_token')}
        receipt.update(kind='inference',format_version=1,report=report,artifacts={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()})
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as z:
            z.writestr('receipt.json',encoded(receipt))
            for n,b in files.items():z.writestr(n,b)
        return output.getvalue()


class InferenceTests(InferenceFixture):
    def test_inputs_are_isolated_from_datasets_labels_and_deduplicate_by_content(self):
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM images').fetchone()[0],9)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM datasets').fetchone()[0],1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM inference_inputs').fetchone()[0],2)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM inference_upload_failures').fetchone()[0],1)
        result=self.infer_inputs.upload(self.pid,self.a,'duplicate.png',picture(size=(80,40)))
        self.assertTrue(result['reused']);self.assertEqual(result['input']['id'],self.object['id'])
        with self.assertRaises(ProjectError):self.infer_inputs.get_input(self.pid,self.b,self.object['id'])
        with self.assertRaises(ProjectError):self.infer_inputs.upload(self.pid,self.a,'../escape.png',b'x')

    def test_exif_canonical_coordinates_preserve_original_and_dimensions(self):
        raw=picture('JPEG',orientation=6,size=(80,40));record=self.infer_inputs.upload(self.pid,self.a,'orientation.jpg',raw)['input']
        self.assertEqual((record['source_width'],record['source_height']),(80,40));self.assertEqual((record['width'],record['height']),(40,80))
        original,_,_=self.infer_inputs.file(self.pid,self.a,record['id'],'original');self.assertEqual(original.read_bytes(),raw)
        view,_,_=self.infer_inputs.file(self.pid,self.a,record['id'],'view')
        with Image.open(view) as image:self.assertEqual(image.size,(40,80));self.assertNotIn(274,image.getexif())

    def test_transparent_input_uses_same_white_rgb_pixels_in_preview_and_runner(self):
        raw=picture('PNG',color=(255,0,0,0),mode='RGBA');record=self.infer_inputs.upload(self.pid,self.a,'transparent.png',raw)['input']
        view,_,_=self.infer_inputs.file(self.pid,self.a,record['id'],'view')
        with Image.open(view) as image:self.assertEqual(image.mode,'RGB');self.assertEqual(image.getpixel((0,0)),(255,255,255))
        original,_,_=self.infer_inputs.file(self.pid,self.a,record['id'],'original');self.assertEqual(original.read_bytes(),raw)

    def test_failed_run_can_be_recreated_without_overwriting_results_or_training(self):
        job=self.create();self.iexec.dispatch(self.pid,self.a,job['id']);self.iexec.receive(self.pid,self.a,job['id'],self.iresult(job,status='failed',detail=False))
        next_job=self.create();self.assertNotEqual(next_job['id'],job['id']);self.assertEqual(self.inferences.get(self.pid,self.a,job['id'])['state'],'FAILED')
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM training_jobs').fetchone()[0],1)

    def test_corrupt_input_is_failure_and_empty_image_is_ready(self):
        self.assertEqual(self.bad['status'],'INVALID');self.assertEqual(self.empty['status'],'READY')
        self.assertTrue(self.bad['error']);self.assertEqual(self.infer_inputs.upload(self.pid,self.a,'broken.jpg',b'CPU deliberately corrupt JPEG')['input'],self.bad)
        with self.assertRaises(ProjectError):self.create(input_ids=[self.bad['id']])

    def test_creation_pins_inputs_model_and_is_idempotent(self):
        values={'model_id':self.model['id'],'input_ids':[self.object['id'],self.empty['id'],self.bad['id']],'idempotency_key':uuid.uuid4().hex}
        first=self.inferences.create(self.pid,self.a,values);again=self.inferences.create(self.pid,self.a,values)
        self.assertEqual(first['job'],again['job']);self.assertTrue(again['reused'])
        snap=first['job']['snapshot'];self.assertEqual(snap['valid_count'],2);self.assertEqual(snap['sample_count'],3);self.assertEqual(snap['model_version_id'],self.model['id'])
        self.assertEqual(snap['weights_sha256'],self.model['manifest']['artifacts']['best.pt']['sha256'])
        with self.assertRaises(ProjectError):self.inferences.create(self.pid,self.a,{**values,'conf':0.3})
        with self.assertRaises(ProjectError):self.create(input_ids=[self.object['id']]*2)
        with self.assertRaises(ProjectError):self.create(input_ids=['f'*32])
        for values in ({'conf':True},{'conf':float('nan')},{'iou':0},{'extra':1}):
            with self.assertRaises(ProjectError):self.create(**values)

    def test_cancel_retains_inputs_and_terminal_history(self):
        job=self.create();self.inferences.cancel(self.pid,self.a,job['id']);self.assertEqual(self.inferences.cancel(self.pid,self.a,job['id'])['state'],'CANCELLED')
        for query in ('DELETE FROM inference_jobs','DELETE FROM inference_inputs','DELETE FROM inference_job_inputs','UPDATE inference_jobs SET model_id=id','UPDATE inference_upload_failures SET error=\'changed\''):
            with self.assertRaises(sqlite3.IntegrityError),self.store.connect() as db:db.execute(query)
        self.assertTrue(self.infer_inputs.file(self.pid,self.a,self.object['id'],'original')[0].exists())

    def test_package_cpu_prepare_only_and_tamper_rejection(self):
        job=self.create();self.iexec.dispatch(self.pid,self.a,job['id']);path,_,_=self.iexec.package(self.pid,self.a,job['id']);directory=self.path.parent/'infer-prepare';directory.mkdir()
        with zipfile.ZipFile(path) as z:z.extractall(directory)
        result=subprocess.run([sys.executable,str(directory/'run_training_job.py'),'--directory',str(directory),'--prepare-only'],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('"training_executed": false',result.stdout);self.assertFalse((directory/'started.json').exists())
        request=runner.check_input(directory);self.assertEqual(request['kind'],'inference')
        (directory/'model.pt').write_bytes(b'wrong')
        with self.assertRaises(ValueError):runner.check_input(directory)

    def test_cpu_adapter_reports_object_empty_and_bad_without_training(self):
        self.assert_cpu_adapter('8.3.241')

    def test_quantize_runtime_without_half_preserves_fp32_contract(self):
        self.assert_cpu_adapter('8.4.203')

    def assert_cpu_adapter(self,runtime):
        job=self.create();self.iexec.dispatch(self.pid,self.a,job['id']);path,_,_=self.iexec.package(self.pid,self.a,job['id']);directory=self.path.parent/'cpu-infer';directory.mkdir()
        with zipfile.ZipFile(path) as z:z.extractall(directory)
        request=runner.check_input(directory);request['snapshot']['target']['runtime']['ultralytics']=runtime
        snap=job['snapshot'];sources={i['id']:i for i in snap['inputs']}
        class Tensor:
            def __init__(self,value):self.value=value
            def detach(self):return self
            def cpu(self):return self
            def tolist(self):return self.value
        class Model:
            task='detect';names={i:c['name'] for i,c in enumerate(snap['classes'])}
            predictor=SimpleNamespace(device=SimpleNamespace(type='cuda',index=0),args=SimpleNamespace(**snap['requested_parameters']))
            def predict(self,source,**kwargs):
                image=sources[Path(source).stem];empty=image['id']==self_empty
                return [SimpleNamespace(orig_shape=(image['height'],image['width']),boxes=SimpleNamespace(xyxy=Tensor([] if empty else [[1,2,18,19]]),cls=Tensor([] if empty else [0]),conf=Tensor([] if empty else [0.9])),speed={'preprocess':0.1,'inference':0.2,'postprocess':0.1})]
        if runtime=='8.4.203':
            del Model.predictor.args.half
            Model.predictor.args.quantize=None
        self_empty=self.empty['id'];module=types.ModuleType('ultralytics');module.__version__=runtime;module.YOLO=lambda path:Model()
        report={'status':'failed','training_executed':False,'evaluation_executed':False,'lock_check_passed':True,'images_completed':0}
        env={'ready':True,'nvidia_smi':TARGET['verified_gpu_uuid'],'python':TARGET['runtime']['python'],'packages':{**TARGET['runtime'],'ultralytics':runtime}}
        with patch.dict(sys.modules,{'ultralytics':module}),patch('tools.compute_probe.checkpoint_dependencies',return_value={'scan_supported':True,'missing':[]}),patch('tools.compute_probe.inspect_environment',return_value=env):runner.infer(directory,request,report)
        data=json.loads((directory/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'],'completed');self.assertEqual([i['status'] for i in data['images']],['SUCCEEDED','SUCCEEDED','FAILED']);self.assertEqual([len(i['predictions']) for i in data['images']],[1,0,0]);self.assertFalse(report['training_executed']);self.assertFalse(report['evaluation_executed'])
        self.iexec.validate_report(data,request,True)

    def test_success_receive_retains_empty_result_and_old_evaluation_and_model(self):
        manifest=self.models.owned(self.model['id'],'manifest.json').read_bytes();job=self.create();self.iexec.dispatch(self.pid,self.a,job['id']);raw=self.iresult(job)
        self.assertEqual(self.iexec.receive(self.pid,self.a,job['id'],raw)['job']['state'],'SUCCEEDED');self.assertTrue(self.iexec.receive(self.pid,self.a,job['id'],raw)['reused'])
        report=self.iexec.report(self.pid,self.a,job['id']);self.assertEqual(report['images'][1]['status'],'SUCCEEDED');self.assertEqual(report['images'][1]['predictions'],[]);self.assertEqual(report['images'][2]['status'],'FAILED')
        self.assertEqual(self.models.owned(self.model['id'],'manifest.json').read_bytes(),manifest);self.assertEqual(self.models.get(self.pid,self.a,self.model['id'])['evaluation_status'],'NOT_EVALUATED')
        with self.assertRaises(ProjectError):self.iexec.receive(self.pid,self.a,job['id'],self.iresult(job,status='failed'))

    def test_untrusted_missing_wrong_class_and_out_of_range_results_retain_reservation(self):
        job=self.create();self.iexec.dispatch(self.pid,self.a,job['id']);good=self.report(job);bad=[]
        data=copy.deepcopy(good);data['images']=[];bad.append(data)
        data=copy.deepcopy(good);data['images'][0]['predictions'][0]['xyxy']=[-1,0,10,20];bad.append(data)
        data=copy.deepcopy(good);data['images'][0]['predictions'][0]['class_name']='wrong';bad.append(data)
        data=copy.deepcopy(good);data['images'][2]['status']='SUCCEEDED';bad.append(data)
        data=copy.deepcopy(good);data['images'][1]['status']='NOT_PROCESSED';data['images'][1]['error']='pending';bad.append(data)
        for data in bad:
            with self.assertRaises(ProjectError):self.iexec.receive(self.pid,self.a,job['id'],self.iresult(job,detail=data))
        self.assertEqual(self.inferences.get(self.pid,self.a,job['id'])['dispatch']['state'],'AWAITING_RESULT')

    def test_filtered_json_and_png_are_readonly_and_use_oriented_dimensions(self):
        job=self.create();self.iexec.dispatch(self.pid,self.a,job['id']);self.iexec.receive(self.pid,self.a,job['id'],self.iresult(job));path=self.iexec.owned(job['id'],'returned','report.json');before=path.read_bytes()
        filtered=self.iexec.export(self.pid,self.a,job['id'],0.99);self.assertEqual(filtered['images'][0]['predictions'],[])
        png=self.iexec.overlay(self.pid,self.a,job['id'],self.object['id'],0.25)
        with Image.open(io.BytesIO(png)) as image:self.assertEqual(image.size,(80,40))
        self.assertEqual(path.read_bytes(),before)
        with self.assertRaises(ProjectError):self.iexec.export(self.pid,self.a,job['id'],0)
        with self.assertRaises(ProjectError):self.iexec.overlay(self.pid,self.a,job['id'],self.bad['id'],0.25)

    def test_three_way_queue_and_database_reservation_guards(self):
        evaluations=EvaluationRepository(self.store,self.models);eexec=EvaluationExecution(self.store,evaluations)
        eval_job=evaluations.ensure_initial(self.pid,self.a,self.model)['job'];infer=self.create()
        with self.assertRaises(ProjectError) as error:self.iexec.dispatch(self.pid,self.a,infer['id'])
        self.assertEqual(error.exception.code,'QUEUE_ORDER');eexec.dispatch(self.pid,self.a,eval_job['id'])
        with self.assertRaises(ProjectError) as error:self.iexec.dispatch(self.pid,self.a,infer['id'])
        self.assertEqual(error.exception.code,'TARGET_RESERVED')
        with self.store.connect() as db:
            db.execute("CREATE TRIGGER block_eval_terminal BEFORE UPDATE ON evaluation_jobs BEGIN SELECT RAISE(ABORT,'test'); END")
            with self.assertRaises(sqlite3.IntegrityError):db.execute("INSERT INTO inference_dispatches VALUES(?,?,?,'AWAITING_RESULT',?,?,NULL,NULL,?,?)",(infer['id'],uuid.uuid4().hex,TARGET['id'],'{}','x','a','a'))

    def test_active_inference_blocks_new_training_and_evaluation(self):
        job=self.create();self.iexec.dispatch(self.pid,self.a,job['id']);later=self.job()
        with self.assertRaises(ProjectError) as error:self.dispatch(later)
        self.assertEqual(error.exception.code,'TARGET_RESERVED');self.training.cancel(self.pid,self.a,later['id'])
        er=EvaluationRepository(self.store,self.models);ee=EvaluationExecution(self.store,er);ej=er.ensure_initial(self.pid,self.a,self.model)['job']
        with self.assertRaises(ProjectError) as error:ee.dispatch(self.pid,self.a,ej['id'])
        self.assertEqual(error.exception.code,'TARGET_RESERVED')

    def test_remote_result_needs_process_and_both_locks_released(self):
        job=self.create();remote=RemoteRepository(self.store,self.inferences,self.iexec,kind='inference');fake=MagicMock();fake.endpoint='root@cpu-fixture:1';fake.call.return_value={'nvidia_smi':TARGET['verified_gpu_uuid']};remote.transport=lambda:fake;remote.start_worker=lambda:None;remote.submit(self.pid,self.a,job['id'])
        with self.store.connect() as db:row=dict(db.execute('SELECT s.*,t.project_id,j.task_id,d.package_sha256 FROM inference_remote_sessions s JOIN inference_jobs j ON j.id=s.job_id JOIN vision_tasks t ON t.id=j.task_id JOIN inference_dispatches d ON d.job_id=j.id').fetchone())
        raw=self.iresult(job);state={'directory_exists':True,'result_ready':True,'alive':True,'launch':{'pid':123},'progress':{},'resources':{'checked':True,'free':False},'result_sha256':hashlib.sha256(raw).hexdigest()};fake.call.return_value=state;remote.reconcile(row);fake.copy.assert_not_called()
        with self.assertRaises(ProjectError):self.iexec.receive(self.pid,self.a,job['id'],raw)
        fake.call.return_value={**state,'alive':False,'resources':{'checked':True,'free':True}};fake.copy.side_effect=lambda path,*args:Path(path).write_bytes(raw);remote.reconcile(row)
        self.assertEqual(self.inferences.get(self.pid,self.a,job['id'])['state'],'SUCCEEDED')

    def test_schema13_backup_and_rollback_preserve_training_model_and_evaluation(self):
        with self.store.connect() as db:
            # Remove test inputs before reconstructing the genuine schema13 baseline.
            drop_inferences(db);db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=13');before=[tuple(r) for r in db.execute('SELECT * FROM model_versions')]
        class Broken(Storage):
            @staticmethod
            def create_inference_tables(db):Storage.create_inference_tables(db);raise RuntimeError('CPU injected')
        with self.assertRaises(RuntimeError):Broken(self.path)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],13)
        restored=Storage(self.path)
        with restored.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16);self.assertEqual([tuple(r) for r in db.execute('SELECT * FROM model_versions')],before);self.assertFalse(db.execute('PRAGMA foreign_key_check').fetchall())
        self.assertTrue(list((self.path.parent/'backups').glob('schema-v13-*.db')))

    def test_result_write_failure_keeps_reservation_and_retry_receives_original(self):
        job=self.create();self.iexec.dispatch(self.pid,self.a,job['id']);raw=self.iresult(job)
        with patch('app.executions.Path.write_bytes',side_effect=OSError(28,'CPU full disk')),self.assertRaises(ProjectError):self.iexec.receive(self.pid,self.a,job['id'],raw)
        self.assertEqual(self.inferences.get(self.pid,self.a,job['id'])['dispatch']['state'],'AWAITING_RESULT')
        self.assertFalse(self.iexec.owned(job['id'],'returned').exists());self.assertEqual(self.iexec.receive(self.pid,self.a,job['id'],raw)['job']['state'],'SUCCEEDED')

    def test_http_upload_result_json_png_and_cross_task(self):
        job=self.create();self.iexec.dispatch(self.pid,self.a,job['id']);self.iexec.receive(self.pid,self.a,job['id'],self.iresult(job))
        server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}/api/projects/{self.pid}/tasks/{self.a}'
        def fetch(path,body=None):return urlopen(Request(base+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=15)
        try:
            with fetch('/inference-inputs') as r:self.assertEqual(json.load(r)['total'],3)
            with urlopen(Request(base+'/inference-inputs',data=b'CPU invalid file',headers={'X-File-Name':quote('bad-new.png'),'Content-Type':'application/octet-stream'}),timeout=15) as r:self.assertEqual(json.load(r)['input']['status'],'INVALID')
            with fetch('/inferences') as r:self.assertEqual(json.load(r)['total'],1)
            with fetch('/inferences/'+job['id']+'/export?conf=0.99') as r:self.assertEqual(json.load(r)['images'][0]['predictions'],[])
            with fetch('/inferences/'+job['id']+'/images/'+self.object['id']+'/overlay?conf=0.25') as r:
                self.assertEqual(r.headers['Content-Type'],'image/png')
                with Image.open(io.BytesIO(r.read())) as image:self.assertEqual(image.size,(80,40))
            with self.assertRaises(HTTPError) as err:urlopen(base.replace(self.a,self.b)+'/inferences/'+job['id'],timeout=10)
            self.assertEqual(err.exception.code,404)
        finally:server.shutdown();server.server_close();thread.join()
