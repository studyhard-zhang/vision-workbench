"""Independent inference inputs and immutable runs. No controller-side model loading."""
import hashlib
import hmac
import io
import json
import math
import re
import secrets
from .images import MAX_FILE_BYTES
import shutil
import stat
import tempfile
import uuid
import zipfile
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont
from .model_parameters import model_imgsz
from .storage import ProjectError,timestamp,validate_project
from .images import ImageRepository,DECODE_SLOTS
from .versions import encoded,sha_file
from .executions import ExecutionRepository,MAX_RESULT_BYTES

MAX_INPUTS=50
MAX_INPUT_BYTES=128*1024**2


class InferenceInputRepository(ImageRepository):
    def __init__(self,storage,tasks):
        super().__init__(storage);self.tasks=tasks
        self.root=(storage.path.parent/'inference-inputs').resolve();self.root.mkdir(exist_ok=True)
        (self.root/'.staging').mkdir(exist_ok=True)

    def check_scope(self,pid,tid,db=None):return self.tasks.get(pid,tid,db)

    def canonical_image(self,oriented,fmt):
        if fmt=='PNG' and ('A' in oriented.getbands() or 'transparency' in oriented.info):
            rgba=oriented.convert('RGBA');result=Image.new('RGB',rgba.size,'white')
            try:result.paste(rgba,mask=rgba.getchannel('A'))
            finally:rgba.close()
            return result
        return oriented.convert('RGB')

    def execute(self,db,sql,parameters=()):
        return db.execute(sql.replace('images','inference_inputs').replace('dataset_id','task_id'),parameters)

    def describe(self,row):
        suffix='jpg' if row['format']=='JPEG' else 'png'
        path=self.owned_path(row['task_id'],row['id'],'display.'+suffix)
        return {k:row[k] for k in ('id','task_id','filename','original_sha256','display_sha256','byte_size','width','height','source_width','source_height','format','created_at')} | {
            'status':'READY','display_bytes':path.stat().st_size if path.is_file() else 0,'file_path':f'inputs/{row["id"]}.{suffix}',
            'coordinate_system':'exif-oriented-pixels','pixel_processing':'EXIF校正方向、RGB标准图；透明区域合成白底。'}

    def get_input(self,pid,tid,iid,db=None):
        if db is None:
            with self.storage.connect() as connection:return self.get_input(pid,tid,iid,connection)
        self.check_scope(pid,tid,db)
        failed=db.execute('SELECT * FROM inference_upload_failures WHERE task_id=? AND id=?',(tid,iid)).fetchone()
        if failed:return dict(failed)|{'status':'INVALID','coordinate_system':'not_decoded'}
        return self.describe(super().get(pid,tid,iid,db))

    def list_inputs(self,pid,tid):
        with self.storage.connect() as db:
            self.check_scope(pid,tid,db)
            rows=[self.describe(dict(r)) for r in db.execute('SELECT * FROM inference_inputs WHERE task_id=? ORDER BY created_at DESC LIMIT 100',(tid,))]
            rows.extend(dict(r)|{'status':'INVALID','coordinate_system':'not_decoded'} for r in db.execute('SELECT * FROM inference_upload_failures WHERE task_id=? ORDER BY created_at DESC LIMIT 100',(tid,)))
        return sorted(rows,key=lambda r:r['created_at'],reverse=True)[:100]

    def upload(self,pid,tid,name,raw):
        self.check_scope(pid,tid)
        try:
            result=super().import_image(pid,tid,name,raw)
            return {'input':self.describe(result['image']),'reused':result['status']=='duplicate'}
        except ProjectError as exc:
            if exc.status not in (413,422) or exc.code=='INVALID_FILENAME':raise
            # Decode failures are durable per-image results, not fake successful images.
            digest=hashlib.sha256(raw).hexdigest();iid=uuid.uuid4().hex
            with self.storage.connect() as db:
                db.execute('BEGIN IMMEDIATE');self.check_scope(pid,tid,db)
                old=db.execute('SELECT id FROM inference_upload_failures WHERE task_id=? AND filename=? AND original_sha256=?',(tid,name,digest)).fetchone()
                if old:return {'input':self.get_input(pid,tid,old['id'],db),'reused':True}
                db.execute('INSERT INTO inference_upload_failures VALUES(?,?,?,?,?,?,?,?)',(iid,tid,name,digest,len(raw),exc.code,exc.message,timestamp()))
            return {'input':self.get_input(pid,tid,iid),'reused':False}

    def from_dataset(self,pid,tid,dataset_id,image_id):
        self.check_scope(pid,tid)
        if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{32}',v) for v in (dataset_id,image_id)):
            raise ProjectError(422,'INVALID_SOURCE_IMAGE','请选择当前项目数据集中的图片。')
        source=ImageRepository(self.storage)
        path,_,image=source.file(pid,dataset_id,image_id,'original')
        if path.stat().st_size>MAX_FILE_BYTES:
            raise ProjectError(409,'SOURCE_IMAGE_DAMAGED','数据集原图缺失、过大或校验不符，请检查源图片；尚未创建推理任务。')
        raw=path.read_bytes()
        if len(raw)>MAX_FILE_BYTES or hashlib.sha256(raw).hexdigest()!=image['original_sha256']:
            raise ProjectError(409,'SOURCE_IMAGE_DAMAGED','数据集原图校验不符，尚未创建推理任务。')
        from .roi import bounds
        with self.storage.connect() as db:area=bounds(db,image)
        name=image['filename']
        if area:
            cropped_path,_,_=source.file(pid,dataset_id,image_id,'view');raw=cropped_path.read_bytes();name=Path(name).stem+'-ROI.png'
        result=self.upload(pid,tid,name,raw)
        return result|{'source':{'dataset_id':dataset_id,'image_id':image_id,'original_unchanged':True,'roi':area}}


class InferenceRepository:
    def __init__(self,storage,models,inputs):
        self.storage=storage;self.models=models;self.training=models.training;self.versions=models.versions;self.inputs=inputs

    def target(self):return self.training.target()

    def row(self,db,pid,tid,jid):
        self.versions.tasks.get(pid,tid,db)
        row=db.execute('SELECT * FROM inference_jobs WHERE task_id=? AND id=?',(tid,jid)).fetchone()
        if row is None:raise ProjectError(404,'INFERENCE_NOT_FOUND','推理记录不存在或不属于当前任务。')
        return row

    def public(self,db,row):
        value={k:row[k] for k in ('id','sequence','task_id','model_id','version_id','name','state','snapshot_sha256','created_at','updated_at')}
        value['snapshot']=json.loads(row['snapshot_json'])
        d=db.execute('SELECT * FROM inference_dispatches WHERE job_id=?',(row['id'],)).fetchone()
        value['dispatch']=None if d is None else {k:d[k] for k in ('id','state','package_sha256','created_at','updated_at')}
        if d is not None:value['dispatch']['result']=json.loads(d['result_json']) if d['result_json'] else None
        remote=db.execute('SELECT phase,cancel_requested,progress_json,error,updated_at FROM inference_remote_sessions WHERE job_id=?',(row['id'],)).fetchone()
        value['remote']=None if remote is None else {k:remote[k] for k in ('phase','cancel_requested','error','updated_at')}
        if remote is not None:value['remote']['progress']=json.loads(remote['progress_json'])
        return value

    def get(self,pid,tid,jid):
        with self.storage.connect() as db:return self.public(db,self.row(db,pid,tid,jid))

    def list(self,pid,tid):
        with self.storage.connect() as db:
            self.versions.tasks.get(pid,tid,db)
            return [self.public(db,r) for r in db.execute('SELECT * FROM inference_jobs WHERE task_id=? ORDER BY sequence DESC',(tid,))]

    def options(self,pid,tid,mid):
        model=self.models.get(pid,tid,mid)
        return {'model':model,'target':self.target(),'max_inputs':MAX_INPUTS,
                'defaults':{'imgsz':model_imgsz(model),'batch':1,'conf':0.001,'iou':0.7,'max_det':300},
                'coordinate_system':'exif-oriented-pixels','inputs':self.inputs.list_inputs(pid,tid),'backends':self.backends()}

    def backends(self):
        from .compute_instances import active
        prepared=bool((active(self.storage.path.parent) or {}).get('profile',{}).get('acceleration'))
        return [{'id':'pytorch','name':'原生推理','enabled':True},{'id':'tensorrt','name':'TensorRT加速（FP16）','enabled':prepared}]

    def create(self,pid,tid,values):
        allowed={'model_id','name','input_ids','conf','iou','idempotency_key','backend','deployment_profile'}
        if not isinstance(values,dict) or set(values)-allowed:raise ProjectError(422,'INVALID_INFERENCE','推理请求字段无效。')
        backend=values.get('backend','pytorch')
        if backend not in ('pytorch','tensorrt'):raise ProjectError(422,'INVALID_BACKEND','请选择原生或TensorRT推理。')
        deployment_profile=values.get('deployment_profile','cuda12')
        if deployment_profile not in ('cuda11','cuda12'):raise ProjectError(422,'INVALID_DEPLOYMENT_PROFILE','请选择CUDA11或CUDA12产线配置。')
        key=values.get('idempotency_key');ids=values.get('input_ids')
        if not isinstance(key,str) or not re.fullmatch('[0-9a-f]{32}',key):raise ProjectError(422,'INVALID_IDEMPOTENCY_KEY','请重新打开推理窗口再提交。')
        if not isinstance(ids,list) or not 1<=len(ids)<=MAX_INPUTS or any(not isinstance(i,str) or not re.fullmatch('[0-9a-f]{32}',i) for i in ids) or len(set(ids))!=len(ids):raise ProjectError(422,'INVALID_INFERENCE_INPUTS','请选择1至50张独立推理输入，不能重复。')
        mid=values.get('model_id')
        if not isinstance(mid,str) or not re.fullmatch('[0-9a-f]{32}',mid):raise ProjectError(422,'INVALID_MODEL_ID','请选择具体模型版本。')
        try:digest=hashlib.sha256(encoded({k:v for k,v in values.items() if k!='idempotency_key'})).hexdigest()
        except (ValueError,TypeError) as exc:raise ProjectError(422,'INVALID_INFERENCE','配置含无效数值。') from exc
        with self.storage.connect() as db:
            self.versions.tasks.get(pid,tid,db)
            old=db.execute('SELECT * FROM inference_jobs WHERE task_id=? AND idempotency_key=?',(tid,key)).fetchone()
            if old:
                if old['request_sha256']!=digest:raise ProjectError(409,'IDEMPOTENCY_CONFLICT','此提交标识已用于其他配置。')
                return {'job':self.public(db,old),'reused':True}
            inputs=[self.inputs.get_input(pid,tid,i,db) for i in ids]
        ready=[i for i in inputs if i['status']=='READY']
        if not ready:raise ProjectError(422,'NO_VALID_INFERENCE_INPUT','没有可推理的图片，请先补充有效JPG/PNG；坏图原因已保留。')
        if sum(i['display_bytes'] for i in ready)>MAX_INPUT_BYTES:raise ProjectError(413,'INFERENCE_BATCH_SIZE','当前批次校正后图片合计最多128MiB。')
        for image in ready:
            path,_,_=self.inputs.file(pid,tid,image['id'],'view')
            if sha_file(path)!=image['display_sha256']:raise ProjectError(409,'INFERENCE_INPUT_DAMAGED','推理输入缺失或校验不符。')
        model=self.models.get(pid,tid,mid)
        if not model['available']:raise ProjectError(409,'MODEL_DAMAGED','模型文件完整性不通过，不能推理。')
        params={'imgsz':model_imgsz(model),'batch':1,'workers':0,'half':False,'device':0,'augment':False,'rect':False,'conf':values.get('conf',0.001),'iou':values.get('iou',0.7),'max_det':300}
        for k in ('conf','iou'):
            if type(params[k]) not in (int,float) or not math.isfinite(params[k]) or not 0.001<=params[k]<=0.95:raise ProjectError(422,'INVALID_INFERENCE_THRESHOLD','最低置信度和NMS IoU需在0.001至0.95之间。')
        name,_,_=validate_project({'name':values.get('name') or f'图片推理 v{model["number"]}'})
        manifest={'format_version':1,'kind':'inference_inputs','task_id':tid,'images':inputs}
        snapshot={'kind':'inference','model_version_id':mid,'model_number':model['number'],'model_manifest_sha256':model['manifest_sha256'],
                  'weights_sha256':model['manifest']['artifacts']['best.pt']['sha256'],'artifact':'best.pt',
                  'data_version_id':model['task_data_version_id'],'source_data_version_number':model['manifest']['data_version_number'],
                  'manifest_sha256':hashlib.sha256(encoded(manifest)).hexdigest(),'inputs':inputs,'sample_count':len(inputs),'valid_count':len(ready),
                  'classes':model['manifest']['classes'],'coordinate_system':'exif-oriented-pixels','requested_parameters':params,'target':self.target()}
        if backend=='tensorrt':
            from .compute_instances import active
            acceleration=(active(self.storage.path.parent) or {}).get('profile',{}).get('acceleration')
            if not acceleration:raise ProjectError(409,'TENSORRT_NOT_READY','当前设备尚未准备TensorRT加速环境；可先使用原生推理。')
            snapshot['backend']='tensorrt';snapshot['acceleration']={'runtime':acceleration['runtime'],'deployment_profile':deployment_profile}
            params['half']=True
        raw=encoded(snapshot);jid=uuid.uuid4().hex;now=timestamp()
        if model['manifest']['training_snapshot'].get('task_type')=='segment':snapshot['task_type']='segment';raw=encoded(snapshot)
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT * FROM inference_jobs WHERE task_id=? AND idempotency_key=?',(tid,key)).fetchone()
            if old:
                if old['request_sha256']!=digest:raise ProjectError(409,'IDEMPOTENCY_CONFLICT','提交标识冲突。')
                return {'job':self.public(db,old),'reused':True}
            db.execute("INSERT INTO inference_jobs(id,task_id,version_id,model_id,idempotency_key,request_sha256,name,state,snapshot_json,snapshot_sha256,created_at,updated_at) VALUES(?,?,?,?,?,?,?,'QUEUED',?,?,?,?)",(jid,tid,model['task_data_version_id'],mid,key,digest,name,raw.decode(),hashlib.sha256(raw).hexdigest(),now,now))
            for pos,image in enumerate(inputs):db.execute('INSERT INTO inference_job_inputs VALUES(?,?,?,?)',(jid,pos,image['id'] if image['status']=='READY' else None,image['id'] if image['status']=='INVALID' else None))
        return {'job':self.get(pid,tid,jid),'reused':False}

    def cancel(self,pid,tid,jid):
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE');row=self.row(db,pid,tid,jid)
            if row['state']=='QUEUED':db.execute("UPDATE inference_jobs SET state='CANCELLED',updated_at=? WHERE id=?",(timestamp(),jid))
            elif row['state']!='CANCELLED':raise ProjectError(409,'INFERENCE_NOT_WAITING','已派发的推理请中止实例运行。')
        return self.get(pid,tid,jid)


class InferenceExecution(ExecutionRepository):
    kind='inference'
    def __init__(self,storage,inferences):
        super().__init__(storage,inferences)
        self.root=(storage.path.parent/'inferences').resolve();self.root.mkdir(exist_ok=True)
        self.staging=self.root/'.staging';self.staging.mkdir(exist_ok=True)

    def dispatch(self,pid,tid,jid):
        if not self.slot.acquire(blocking=False):raise ProjectError(409,'EXECUTION_BUSY','正在准备或接收推理材料。')
        destination=self.owned(jid);published=False;committed=False
        try:
            job=self.training.get(pid,tid,jid)
            if job['dispatch']:return {'job':job,'reused':True}
            if job['state']!='QUEUED':raise ProjectError(409,'INFERENCE_NOT_WAITING','推理已结束，请按配置新建。')
            snap=job['snapshot'];target=self.training.target()
            if (self.storage.path.parent/'ssh-target-change.json').is_file() or snap['target']!=target:raise ProjectError(409,'SSH_RUN_TARGET_MISMATCH','计算目标已变化，请保留历史并按配置新建。')
            model=self.training.models.get(pid,tid,job['model_id'])
            if not model['available'] or model['manifest_sha256']!=snap['model_manifest_sha256']:raise ProjectError(409,'MODEL_DAMAGED','所选模型完整性不通过。')
            self.versions.space(sum(i.get('display_bytes',0) for i in snap['inputs'])*2+MAX_RESULT_BYTES+8*1024**2)
            with tempfile.TemporaryDirectory(dir=self.staging) as temp:
                stage=Path(temp)
                manifest={'format_version':1,'kind':'inference_inputs','task_id':tid,'images':snap['inputs']}
                if hashlib.sha256(encoded(manifest)).hexdigest()!=snap['manifest_sha256']:raise ProjectError(409,'INFERENCE_SNAPSHOT_DAMAGED','推理输入清单SHA不符。')
                with zipfile.ZipFile(stage/'dataset.zip','w',compression=zipfile.ZIP_STORED) as z:
                    z.writestr('manifest.json',encoded(manifest))
                    for image in snap['inputs']:
                        if image['status']!='READY':continue
                        path,_,_=self.training.inputs.file(pid,tid,image['id'],'view')
                        if sha_file(path)!=image['display_sha256'] or path.stat().st_size!=image['display_bytes']:raise ProjectError(409,'INFERENCE_INPUT_DAMAGED','输入图片缺失或变化，未派发。')
                        z.write(path,image['file_path'])
                weights,_,_=self.training.models.artifact(pid,tid,job['model_id'],'best.pt');tools=Path(__file__).resolve().parents[1]/'tools'
                scripts={name:(tools/source).read_bytes().replace(b'\r\n',b'\n') for name,source in [('run_training_job.py','run_inference_job.py'),('training_support.py','run_training_job.py'),('inference_support.py','run_evaluation_job.py'),('compute_probe.py','compute_probe.py'),('verify_dataset.py','verify_dataset.py')]}
                if snap.get('backend')=='tensorrt':
                    for name in ('tensorrt_support.py','deployment_build.py','deployment_predict_trt.py'):scripts[name]=(tools/name).read_bytes().replace(b'\r\n',b'\n')
                request={'format_version':1,'kind':'inference','job_id':jid,'dispatch_id':uuid.uuid4().hex,'return_token':secrets.token_hex(32),
                         'snapshot':snap,'snapshot_sha256':job['snapshot_sha256'],'expected_gpu_uuid':target['verified_gpu_uuid'],
                         'weights_sha256':snap['weights_sha256'],'dataset_sha256':sha_file(stage/'dataset.zip'),'scripts':{n:hashlib.sha256(b).hexdigest() for n,b in scripts.items()}}
                with zipfile.ZipFile(stage/'execution.zip','w',compression=zipfile.ZIP_STORED) as z:
                    z.write(stage/'dataset.zip','dataset.zip');z.write(weights,'model.pt');z.writestr('request.json',encoded(request));z.writestr('README.txt','图片推理，不训练或评估：python -u run_training_job.py\n')
                    for n,b in scripts.items():z.writestr(n,b)
                (stage/'dataset.zip').unlink();sha=sha_file(stage/'execution.zip');now=timestamp()
                with self.storage.connect() as db:
                    db.execute('BEGIN IMMEDIATE');row=self.training.row(db,pid,tid,jid)
                    if row['state']!='QUEUED':raise ProjectError(409,'INFERENCE_STATE_CHANGED','推理状态已变化。')
                    for kind in ('training','evaluation','inference'):
                        if db.execute(f"SELECT 1 FROM {kind}_dispatches WHERE target_id=? AND state='AWAITING_RESULT'",(target['id'],)).fetchone():raise ProjectError(409,'TARGET_RESERVED','该目标已有训练、评估或推理占用。')
                        if db.execute(f"SELECT 1 FROM {kind}_jobs WHERE state='QUEUED' AND created_at<? AND json_extract(snapshot_json,'$.target.id')=?",(row['created_at'],target['id'])).fetchone():raise ProjectError(409,'QUEUE_ORDER','请先处理更早的计算任务。')
                    if destination.exists():raise ProjectError(409,'EXECUTION_FILES_EXIST','已有未提交目录，未覆盖材料。')
                    stage.rename(destination);published=True
                    db.execute("INSERT INTO inference_dispatches VALUES(?,?,?,'AWAITING_RESULT',?,?,NULL,NULL,?,?)",(jid,request['dispatch_id'],target['id'],encoded(request).decode(),sha,now,now))
                    db.execute("UPDATE inference_jobs SET state='DISPATCHED',updated_at=? WHERE id=?",(now,jid))
                committed=True
            return {'job':self.training.get(pid,tid,jid),'reused':False}
        except OSError as exc:raise ProjectError(507 if exc.errno==28 else 503,'INFERENCE_IO_FAILED','推理材料未完成，请检查空间和权限。') from exc
        finally:
            if published and not committed and destination.exists():shutil.rmtree(destination)
            self.slot.release()

    def validate_result(self,raw,request):
        try:
            if not 0<len(raw)<=MAX_RESULT_BYTES:raise ValueError('结果包体积无效。')
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                items=z.infolist();names=[m.filename for m in items]
                if len(names)!=len(set(names)) or not set(names)<=({'receipt.json','report.json','console.log','deployment.zip'} if request['snapshot'].get('backend')=='tensorrt' else {'receipt.json','report.json','console.log'}) or 'receipt.json' not in names:raise ValueError('结果文件清单无效。')
                if sum(m.file_size for m in items)>MAX_RESULT_BYTES or any(m.is_dir() or m.flag_bits&1 or stat.S_ISLNK(m.external_attr>>16) for m in items) or z.getinfo('receipt.json').file_size>1024**2:raise ValueError('结果成员体积或类型无效。')
                receipt=json.loads(z.read('receipt.json'))
                for key in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256'):
                    if receipt.get(key)!=request[key]:raise ValueError('推理回传身份不符：'+key)
                if receipt.get('kind')!='inference' or receipt.get('format_version')!=1 or not isinstance(receipt.get('return_token'),str) or not hmac.compare_digest(receipt['return_token'],request['return_token']):raise ValueError('回传标识不符。')
                files={n:z.read(n) for n in names if n!='receipt.json'}
                if set(receipt['artifacts'])!=set(files):raise ValueError('产物声明不一致。')
                for n,b in files.items():
                    if receipt['artifacts'][n]!={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}:raise ValueError('文件SHA不符。')
                report=receipt['report'];status=report.get('status')
                if status not in ('completed','failed','interrupted') or report.get('training_executed') is not False or report.get('evaluation_executed') is not False or report.get('lock_check_passed') is not True:raise ValueError('推理执行或锁证据无效。')
                if not isinstance(report.get('error',''),str) or len(report.get('error',''))>4000:raise ValueError('错误说明无效。')
                if 'report.json' in files:self.validate_report(json.loads(files['report.json']),request,status=='completed')
                if status=='completed':
                    if report.get('inference_executed') is not True or 'report.json' not in files or report.get('actual_parameters')!=request['snapshot']['requested_parameters'] or report.get('weights_sha256')!=request['weights_sha256']:raise ValueError('缺少实际推理或参数/权重不符。')
                    if request['snapshot'].get('backend')=='tensorrt':
                        if 'deployment.zip' not in files:raise ValueError('缺少产线模型包。')
                        self.validate_deployment(files['deployment.zip'],request,json.loads(files['report.json']))
                    env=report.get('environment',{})
                    if env.get('ready') is not True or request['expected_gpu_uuid'] not in env.get('nvidia_smi','') or env.get('cuda_visible_devices')!='0':raise ValueError('指定GPU环境证据不符。')
                receipt.pop('return_token');encoded(receipt)
                return receipt,files
        except (ValueError,KeyError,TypeError,AttributeError,zipfile.BadZipFile,RuntimeError,NotImplementedError) as exc:raise ProjectError(422,'INVALID_INFERENCE_RESULT','未接收推理结果：'+str(exc)) from exc

    def validate_deployment(self,raw,request,data):
        snap=request['snapshot'];a=data['acceleration'];binding=a['binding']
        if a['backend']!='tensorrt' or binding['weights_sha256']!=snap['weights_sha256'] or binding['gpu_uuid']!=snap['target']['verified_gpu_uuid'] or binding['runtime']!=snap['acceleration']['runtime'] or binding['imgsz']!=snap['requested_parameters']['imgsz'] or binding['batch']!=1 or binding['precision']!='FP16':raise ValueError('TensorRT来源与当前快照不符。')
        for backend in ('pytorch','tensorrt'):
            b=a[backend]
            if b['iterations']!=20 or b['warmup']!=5 or any(type(b[k]) not in (int,float) or not math.isfinite(b[k]) or b[k]<0 for k in ('processing_median_ms','processing_p95_ms','inference_median_ms')):raise ValueError('加速计时记录无效。')
        if not re.fullmatch('[a-f0-9]{64}',a['engine_sha256']) or not re.fullmatch('[a-f0-9]{64}',a['onnx_sha256']):raise ValueError('加速产物SHA无效。')
        allowed={'model.onnx','metadata.json','deployment.json','benchmark.json','classes.json','build_engine.py','predict.py','README.md','checksums.json'}
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            items=z.infolist()
            if len(items)!=len(allowed) or set(z.namelist())!=allowed or sum(i.file_size for i in items)>240*1024**2 or any(i.flag_bits&1 or i.is_dir() or stat.S_ISLNK(i.external_attr>>16) for i in items):raise ValueError('产线包文件清单或体积无效。')
            for name in allowed-{'model.onnx'}:
                if z.getinfo(name).file_size>1024**2:raise ValueError('产线包元数据过大。')
            sums=json.loads(z.read('checksums.json'))['files']
            if set(sums)!=allowed-{'checksums.json'}:raise ValueError('产线包校验清单无效。')
            for name,spec in sums.items():
                body=z.read(name)
                if spec!={'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}:raise ValueError('产线包SHA不符。')
            if sums['model.onnx']['sha256']!=a['onnx_sha256'] or json.loads(z.read('classes.json'))!=snap['classes'] or json.loads(z.read('benchmark.json'))!=a:raise ValueError('产线包来源不符。')
            metadata=json.loads(z.read('metadata.json'))
            if metadata.get('task')!=snap.get('task_type','detect') or metadata.get('imgsz')!=[binding['imgsz'],binding['imgsz']] or metadata.get('batch')!=1 or {int(k):v for k,v in metadata['names'].items()}!={c['index']:c['name'] for c in snap['classes']}:raise ValueError('产线模型元数据与快照不符。')
            for name,source in [('build_engine.py','deployment_build.py'),('predict.py','deployment_predict_trt.py')]:
                if sums[name]['sha256']!=request['scripts'][source]:raise ValueError('产线脚本与派发版本不符。')
            config=json.loads(z.read('deployment.json'))
            if config['cuda_profile']!=snap['acceleration']['deployment_profile'] or config['source_model_id']!=snap['model_version_id'] or config['source_weights_sha256']!=snap['weights_sha256'] or config['source_onnx_sha256']!=a['onnx_sha256'] or config['imgsz']!=snap['requested_parameters']['imgsz'] or config['conf']!=snap['requested_parameters']['conf'] or config['iou']!=snap['requested_parameters']['iou'] or config['task_type']!=snap.get('task_type','detect') or config['batch']!=1 or config['precision']!='FP16':raise ValueError('产线配置不符。')

    def validate_report(self,data,request,complete):
        snap=request['snapshot'];rows=data['images'];expected={i['id']:i for i in snap['inputs']}
        if data['snapshot']!=snap or data['runtime']!=snap['target']['runtime']['ultralytics'] or data['coordinate_system']!='exif-oriented-pixels' or type(data['complete']) is not bool or (complete and data['complete'] is not True):raise ValueError('报告来源、坐标口径或完成标记不符。')
        if not isinstance(rows,list) or len(rows)!=len(expected) or {i['id'] for i in rows}!=set(expected):raise ValueError('逐图报告不完整或重复。')
        success=0
        for row in rows:
            source=expected[row['id']]
            if row['filename']!=source['filename'] or row['status'] not in ('SUCCEEDED','FAILED','NOT_PROCESSED') or not isinstance(row['predictions'],list):raise ValueError('逐图来源或状态无效。')
            if complete and row['status']=='NOT_PROCESSED':raise ValueError('完成报告仍有未处理图片。')
            if row['status']!='SUCCEEDED':
                if row['predictions'] or not isinstance(row.get('error'),str) or not 1<=len(row['error'])<=1000:raise ValueError('失败原因缺失或混入预测。')
                if source['status']=='INVALID' and (row['status']!='FAILED' or row.get('error_code')!=source['error_code']):raise ValueError('坏图失败原因不符。')
                continue
            success+=1
            if source['status']!='READY' or row['width']!=source['width'] or row['height']!=source['height'] or len(row['predictions'])>snap['requested_parameters']['max_det']:raise ValueError('图片尺寸或预测数无效。')
            for box in row['predictions']:
                idx=box['class_index'];conf=box['confidence'];xy=box['xyxy']
                if type(idx) is not int or not 0<=idx<len(snap['classes']) or box['class_name']!=snap['classes'][idx]['name'] or type(conf) not in (int,float) or not math.isfinite(conf) or not snap['requested_parameters']['conf']-1e-7<=conf<=1:raise ValueError('预测类别/置信度无效。')
                if not isinstance(xy,list) or len(xy)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in xy) or not 0<=xy[0]<=xy[2]<=source['width'] or not 0<=xy[1]<=xy[3]<=source['height']:raise ValueError('预测坐标超出校正方向原图。')
                if snap.get('task_type')=='segment':
                    from .prediction_masks import validate_segments
                    validate_segments(box,source['width'],source['height'])
        if complete and not success:raise ValueError('没有成功推理的图片，不能登记完成。')
        encoded(data)

    def artifact(self,pid,tid,jid,name):
        job=self.training.get(pid,tid,jid);result=job['dispatch']['result'] if job['dispatch'] else None
        if name not in ('report.json','console.log','deployment.zip') or not result or name not in result['artifacts']:raise ProjectError(404,'ARTIFACT_NOT_FOUND','推理材料尚不存在。')
        path=self.owned(jid,'returned',name);spec=result['artifacts'][name]
        if not path.is_file() or path.stat().st_size!=spec['bytes'] or sha_file(path)!=spec['sha256']:raise ProjectError(409,'ARTIFACT_DAMAGED','推理材料缺失或SHA不符。')
        return path,f'infer-{jid[:8]}-{name}','application/zip' if name.endswith('.zip') else 'application/json' if name.endswith('.json') else 'text/plain; charset=utf-8'

    def report(self,pid,tid,jid):
        path,_,_=self.artifact(pid,tid,jid,'report.json')
        return json.loads(path.read_text(encoding='utf-8'))

    @staticmethod
    def display_confidence(data,value):
        try:confidence=float(value)
        except (ValueError,TypeError) as exc:raise ProjectError(422,'INVALID_DISPLAY_THRESHOLD','显示置信度无效。') from exc
        if not math.isfinite(confidence) or not data['snapshot']['requested_parameters']['conf']<=confidence<=1:raise ProjectError(422,'INVALID_DISPLAY_THRESHOLD','显示置信度不能低于本次保存的最低值，且不能超过1。')
        return confidence

    def export(self,pid,tid,jid,value):
        data=self.report(pid,tid,jid);conf=self.display_confidence(data,value)
        return {**data,'display_confidence':conf,'export_scope':'筛选已保存预测，不改变原任务、参数或评估报告。',
                'images':[{**i,'predictions':[p for p in i['predictions'] if p['confidence']>=conf]} for i in data['images']]}

    def overlay(self,pid,tid,jid,iid,value):
        data=self.export(pid,tid,jid,value);row=next((i for i in data['images'] if i['id']==iid),None)
        if not row or row['status']!='SUCCEEDED':raise ProjectError(409,'INFERENCE_IMAGE_UNAVAILABLE','该图片没有成功推理结果，不能下载叠图。')
        if not DECODE_SLOTS.acquire(blocking=False):raise ProjectError(429,'IMAGE_BUSY','图片处理繁忙，请稍后重试。')
        try:
            path,_,record=self.training.inputs.file(pid,tid,iid,'view')
            if sha_file(path)!=record['display_sha256']:raise ProjectError(409,'INFERENCE_INPUT_DAMAGED','输入图片校验不符，未生成叠图。')
            with Image.open(path) as source:
                image=source.convert('RGB')
                try:
                    draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=max(14,min(image.size)//55))
                    for box in row['predictions']:
                        if 'segments' in box:
                            mask=Image.new('L',image.size,0);md=ImageDraw.Draw(mask)
                            try:
                                for ring in box['segments']:md.polygon([tuple(p) for p in ring['points']],fill=0 if ring['hole'] else 85)
                                tint=Image.new('RGB',image.size,'#ee741c');image.paste(tint,(0,0),mask);tint.close()
                            finally:mask.close()
                            draw=ImageDraw.Draw(image)
                            for ring in box['segments']:draw.line([tuple(p) for p in ring['points']]+[tuple(ring['points'][0])],fill='#ee741c',width=max(2,min(image.size)//400))
                        else:
                            draw.rectangle(box['xyxy'],outline='#ee741c',width=max(2,min(image.size)//400))
                        draw.text((box['xyxy'][0]+2,max(0,box['xyxy'][1]-22)),f"#{box['class_index']} {box['confidence']:.3f}",fill='#ee741c',font=font)
                    result=io.BytesIO();image.save(result,format='PNG');return result.getvalue()
                finally:image.close()
        finally:DECODE_SLOTS.release()
