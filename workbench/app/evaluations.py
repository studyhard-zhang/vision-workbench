"""Version-bound independent evaluation. Controller handles JSON and bytes, never torch."""
import hashlib
import hmac
import io
import json
import math
import re
import secrets
import shutil
import stat
import tempfile
import uuid
import zipfile
from pathlib import Path

from .model_parameters import model_imgsz
from .storage import ProjectError,timestamp,validate_project
from .versions import encoded,sha_file
from .executions import ExecutionRepository,MAX_RESULT_BYTES

METRICS=('precision','recall','map50','map50_95')
PLOTS={'BoxPR_curve.png','BoxF1_curve.png','BoxP_curve.png','BoxR_curve.png','MaskPR_curve.png','MaskF1_curve.png','MaskP_curve.png','MaskR_curve.png','confusion_matrix.png','confusion_matrix_normalized.png'}
ARTIFACTS={'report.json','console.log',*PLOTS}


class EvaluationRepository:
    def __init__(self,storage,models):
        self.storage=storage;self.models=models;self.training=models.training;self.versions=models.versions

    def target(self):return self.training.target()

    def row(self,db,project_id,task_id,job_id):
        self.versions.tasks.get(project_id,task_id,db)
        row=db.execute('SELECT * FROM evaluation_jobs WHERE task_id=? AND id=?',(task_id,job_id)).fetchone()
        if row is None:raise ProjectError(404,'EVALUATION_NOT_FOUND','评估记录不存在或不属于当前任务。')
        return row

    def public(self,db,row):
        value={k:row[k] for k in ('id','sequence','task_id','model_id','version_id','name','state','created_at','updated_at','snapshot_sha256')}
        value['snapshot']=json.loads(row['snapshot_json'])
        d=db.execute('SELECT * FROM evaluation_dispatches WHERE job_id=?',(row['id'],)).fetchone()
        value['dispatch']=None if d is None else {k:d[k] for k in ('id','state','package_sha256','created_at','updated_at')}
        if d is not None:value['dispatch']['result']=json.loads(d['result_json']) if d['result_json'] else None
        r=db.execute('SELECT phase,cancel_requested,progress_json,error,updated_at FROM evaluation_remote_sessions WHERE job_id=?',(row['id'],)).fetchone()
        value['remote']=None if r is None else {k:r[k] for k in ('phase','cancel_requested','error','updated_at')}
        if r is not None:value['remote']['progress']=json.loads(r['progress_json'])
        return value

    def get(self,pid,tid,jid):
        with self.storage.connect() as db:return self.public(db,self.row(db,pid,tid,jid))

    def list(self,pid,tid):
        with self.storage.connect() as db:
            self.versions.tasks.get(pid,tid,db)
            return [self.public(db,r) for r in db.execute('SELECT * FROM evaluation_jobs WHERE task_id=? ORDER BY sequence DESC',(tid,))]

    def options(self,pid,tid,model_id):
        model=self.models.get(pid,tid,model_id)
        data=self.versions.read(pid,tid,model['task_data_version_id'])['manifest']
        return {'model':model,'subsets':[{'id':s,'count':sum(i['subset']==s for i in data['images'])} for s in ('val','test')],
                'target':self.target(),'defaults':{'imgsz':model_imgsz(model),'batch':1,'conf':0.001,'iou':0.7,'max_det':300},
                'notice':'独立于训练重新执行验证/测试；验证集已用于训练选模，不代表未见过的独立测试集。保存只排队，SSH运行才使用GPU。'}

    def create(self,pid,tid,values):
        allowed={'model_id','subset','name','idempotency_key','conf','iou'}
        if not isinstance(values,dict) or set(values)-allowed:raise ProjectError(422,'INVALID_EVALUATION','评估请求字段无效。')
        key=values.get('idempotency_key','')
        if not isinstance(key,str) or not re.fullmatch('[0-9a-f]{32}',key):raise ProjectError(422,'INVALID_IDEMPOTENCY_KEY','请重新打开评估窗口再提交。')
        if not isinstance(values.get('model_id'),str) or not re.fullmatch('[0-9a-f]{32}',values['model_id']):raise ProjectError(422,'INVALID_MODEL_ID','请选择具体模型版本。')
        try:digest=hashlib.sha256(encoded({k:v for k,v in values.items() if k!='idempotency_key'})).hexdigest()
        except (ValueError,TypeError) as exc:raise ProjectError(422,'INVALID_EVALUATION','评估配置含无效数值。') from exc
        with self.storage.connect() as db:
            self.versions.tasks.get(pid,tid,db)
            old=db.execute('SELECT * FROM evaluation_jobs WHERE task_id=? AND idempotency_key=?',(tid,key)).fetchone()
            if old:
                if old['request_sha256']!=digest:raise ProjectError(409,'IDEMPOTENCY_CONFLICT','此提交标识已用于其他评估配置。')
                return {'job':self.public(db,old),'reused':True}
        model=self.models.get(pid,tid,values.get('model_id',''))
        if not model['available']:raise ProjectError(409,'MODEL_DAMAGED','模型完整性不通过，不能提交评估。')
        version=self.versions.read(pid,tid,model['task_data_version_id']);data=version['manifest']
        if not version['integrity']['ok']:raise ProjectError(409,'VERSION_DAMAGED','冻结版本完整性不通过。')
        subset=values.get('subset','val');images=[i for i in data['images'] if i['subset']==subset]
        if subset not in ('val','test') or not images:raise ProjectError(422,'EVALUATION_EMPTY_SUBSET','所选验证/测试集没有图片；不能用训练集代替。')
        if len(images)>2000:raise ProjectError(422,'EVALUATION_CAPACITY','当前评估对照最多2000张，请保留版本并分阶段扩容。')
        params={'imgsz':model_imgsz(model),'batch':1,'workers':0,'half':False,'device':0,'conf':values.get('conf',0.001),'iou':values.get('iou',0.7),'max_det':300,'augment':False}
        for k in ('conf','iou'):
            if type(params[k]) not in (float,int) or not math.isfinite(params[k]) or not 0.001<=params[k]<=0.95:raise ProjectError(422,'INVALID_EVALUATION_THRESHOLD','置信度和NMS IoU需在0.001至0.95之间。')
        name,_,_=validate_project({'name':values.get('name') or f'评估 v{model["number"]} · {subset}'})
        snapshot={'kind':'evaluation','model_version_id':model['id'],'model_number':model['number'],'model_manifest_sha256':model['manifest_sha256'],
                  'artifact':'best.pt','weights_sha256':model['manifest']['artifacts']['best.pt']['sha256'],
                  'data_version_id':model['task_data_version_id'],'data_version_number':data['number'],'manifest_sha256':hashlib.sha256(encoded(data)).hexdigest(),
                  'subset':subset,'sample_count':len(images),'classes':data['classes'],'requested_parameters':params,'target':self.target()}
        raw=encoded(snapshot);jid=uuid.uuid4().hex;now=timestamp()
        if model['manifest']['training_snapshot'].get('task_type')=='segment':snapshot['task_type']='segment';raw=encoded(snapshot)
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT * FROM evaluation_jobs WHERE task_id=? AND idempotency_key=?',(tid,key)).fetchone()
            if old:
                if old['request_sha256']!=digest:raise ProjectError(409,'IDEMPOTENCY_CONFLICT','提交内容冲突。')
                return {'job':self.public(db,old),'reused':True}
            db.execute('''INSERT INTO evaluation_jobs(id,task_id,version_id,model_id,idempotency_key,request_sha256,name,state,snapshot_json,snapshot_sha256,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,'QUEUED',?,?,?,?)''',(jid,tid,model['task_data_version_id'],model['id'],key,digest,name,raw.decode(),hashlib.sha256(raw).hexdigest(),now,now))
        return {'job':self.get(pid,tid,jid),'reused':False}

    def cancel(self,pid,tid,jid):
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE');row=self.row(db,pid,tid,jid)
            if row['state']=='QUEUED':db.execute("UPDATE evaluation_jobs SET state='CANCELLED',updated_at=? WHERE id=?",(timestamp(),jid))
            elif row['state']!='CANCELLED':raise ProjectError(409,'EVALUATION_NOT_WAITING','已派发的评估请使用中止实例运行。')
        return self.get(pid,tid,jid)

    def ensure_initial(self,pid,tid,model):
        # An explicit model registration queues one evaluation; it never starts a GPU.
        return self.create(pid,tid,{'model_id':model['id'],'subset':'val','name':f'首次验证 v{model["number"]}','idempotency_key':model['id']})


class EvaluationExecution(ExecutionRepository):
    kind='evaluation'
    def __init__(self,storage,evaluations):
        super().__init__(storage,evaluations)
        self.root=(storage.path.parent/'evaluations').resolve();self.root.mkdir(exist_ok=True)
        self.staging=self.root/'.staging';self.staging.mkdir(exist_ok=True)

    def dispatch(self,pid,tid,jid):
        if not self.slot.acquire(blocking=False):raise ProjectError(409,'EXECUTION_BUSY','正在准备或接收评估材料。')
        destination=self.owned(jid);published=False;committed=False
        try:
            job=self.training.get(pid,tid,jid)
            if job['dispatch']:return {'job':job,'reused':True}
            if job['state']!='QUEUED':raise ProjectError(409,'EVALUATION_NOT_WAITING','评估已结束，请按配置新建。')
            snap=job['snapshot'];target=self.training.target()
            if (self.storage.path.parent/'ssh-target-change.json').is_file() or snap['target']!=target:raise ProjectError(409,'SSH_RUN_TARGET_MISMATCH','计算目标已变化，请保留记录并按配置新建。')
            model=self.training.models.get(pid,tid,job['model_id'])
            if not model['available'] or model['manifest_sha256']!=snap['model_manifest_sha256']:raise ProjectError(409,'MODEL_DAMAGED','指定模型缺失或校验不符。')
            with tempfile.TemporaryDirectory(dir=self.staging) as temp:
                stage=Path(temp)
                with self.versions.archive(pid,tid,job['version_id']) as (source,size,_):
                    self.versions.space(size*2+MAX_RESULT_BYTES)
                    with (stage/'dataset.zip').open('wb') as stream:shutil.copyfileobj(source,stream)
                from tools.verify_dataset import audit
                checked=audit(stage/'dataset.zip')
                if not checked['passed'] or checked['manifest_sha256']!=snap['manifest_sha256']:raise ProjectError(409,'VERSION_DAMAGED','执行前冻结版本核对不通过。')
                weights,_,_=self.training.models.artifact(pid,tid,job['model_id'],'best.pt')
                tools=Path(__file__).resolve().parents[1]/'tools'
                scripts={name:(tools/source).read_bytes().replace(b'\r\n',b'\n') for name,source in
                         [('run_training_job.py','run_evaluation_job.py'),('training_support.py','run_training_job.py'),('compute_probe.py','compute_probe.py'),('verify_dataset.py','verify_dataset.py')]}
                request={'format_version':1,'kind':'evaluation','job_id':jid,'dispatch_id':uuid.uuid4().hex,'return_token':secrets.token_hex(32),
                         'snapshot':snap,'snapshot_sha256':job['snapshot_sha256'],'expected_gpu_uuid':target['verified_gpu_uuid'],
                         'dataset_sha256':sha_file(stage/'dataset.zip'),'weights_sha256':snap['weights_sha256'],
                         'scripts':{n:hashlib.sha256(b).hexdigest() for n,b in scripts.items()}}
                with zipfile.ZipFile(stage/'execution.zip','w',compression=zipfile.ZIP_STORED) as z:
                    z.write(stage/'dataset.zip','dataset.zip');z.write(weights,'model.pt');z.writestr('request.json',encoded(request))
                    z.writestr('README.txt','独立评估，不训练：python -u run_training_job.py；结果为return-result.zip。\n')
                    for name,raw in scripts.items():z.writestr(name,raw)
                (stage/'dataset.zip').unlink();package_sha=sha_file(stage/'execution.zip');now=timestamp()
                with self.storage.connect() as db:
                    db.execute('BEGIN IMMEDIATE');row=self.training.row(db,pid,tid,jid)
                    if row['state']!='QUEUED':raise ProjectError(409,'EVALUATION_STATE_CHANGED','评估状态已变化。')
                    for kind in ('training','evaluation','inference'):
                        if db.execute(f"SELECT 1 FROM {kind}_dispatches WHERE state='AWAITING_RESULT' AND target_id=?",(target['id'],)).fetchone():raise ProjectError(409,'TARGET_RESERVED','该目标已有训练或评估占用，请先完成原任务。')
                        if db.execute(f"SELECT 1 FROM {kind}_jobs WHERE state='QUEUED' AND created_at<? AND json_extract(snapshot_json,'$.target.id')=?",(row['created_at'],target['id'])).fetchone():raise ProjectError(409,'QUEUE_ORDER','请先处理更早提交的训练或评估。')
                    if destination.exists():raise ProjectError(409,'EXECUTION_FILES_EXIST','已有未提交目录，不覆盖材料。')
                    stage.rename(destination);published=True
                    db.execute("INSERT INTO evaluation_dispatches VALUES(?,?,?,'AWAITING_RESULT',?,?,NULL,NULL,?,?)",(jid,request['dispatch_id'],target['id'],encoded(request).decode(),package_sha,now,now))
                    db.execute("UPDATE evaluation_jobs SET state='DISPATCHED',updated_at=? WHERE id=?",(now,jid))
                committed=True
            return {'job':self.training.get(pid,tid,jid),'reused':False}
        except OSError as exc:raise ProjectError(507 if exc.errno==28 else 503,'EVALUATION_IO_FAILED','评估材料未准备完成，请检查空间与权限。') from exc
        finally:
            if published and not committed and destination.exists():shutil.rmtree(destination)
            self.slot.release()

    def validate_result(self,raw,request):
        try:
            if not 0<len(raw)<=MAX_RESULT_BYTES:raise ValueError('结果包体积无效。')
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                items=z.infolist();names=[m.filename for m in items]
                if len(names)!=len(set(names)) or not set(names)<={'receipt.json',*ARTIFACTS} or 'receipt.json' not in names:raise ValueError('结果文件清单无效。')
                if sum(m.file_size for m in items)>MAX_RESULT_BYTES or any(m.is_dir() or m.flag_bits&1 or stat.S_ISLNK(m.external_attr>>16) for m in items):raise ValueError('结果成员体积或类型无效。')
                if z.getinfo('receipt.json').file_size>1024**2:raise ValueError('清单过大。')
                receipt=json.loads(z.read('receipt.json'))
                for k in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256'):
                    if receipt.get(k)!=request[k]:raise ValueError('评估身份不符：'+k)
                if receipt.get('kind')!='evaluation' or receipt.get('format_version')!=1 or not isinstance(receipt.get('return_token'),str) or not hmac.compare_digest(receipt['return_token'],request['return_token']):raise ValueError('评估回传标识不符。')
                files={n:z.read(n) for n in names if n!='receipt.json'}
                if set(receipt['artifacts'])!=set(files):raise ValueError('文件声明不一致。')
                for n,b in files.items():
                    if receipt['artifacts'][n]!={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}:raise ValueError('文件校验不符：'+n)
                report=receipt['report']
                if report.get('status') not in ('completed','failed','interrupted') or report.get('training_executed') is not False or report.get('lock_check_passed') is not True:raise ValueError('评估状态或锁证据无效。')
                if not isinstance(report.get('error',''),str) or len(report.get('error',''))>4000:raise ValueError('错误说明无效。')
                if report['status']=='completed':
                    if report.get('evaluation_executed') is not True or 'report.json' not in files:raise ValueError('缺少实际评估报告。')
                    if report.get('actual_parameters')!=request['snapshot']['requested_parameters']:raise ValueError('实际评估参数不符。')
                    detail=json.loads(files['report.json']);self.validate_report(detail,request)
                    env=report.get('environment',{})
                    if not env.get('ready') or request['expected_gpu_uuid'] not in env.get('nvidia_smi','') or env.get('cuda_visible_devices')!='0':raise ValueError('指定GPU环境证据不符。')
                    if report.get('weights_sha256')!=request['weights_sha256']:raise ValueError('模型权重校验不符。')
                receipt.pop('return_token');encoded(receipt)
                return receipt,files
        except (ValueError,KeyError,TypeError,AttributeError,zipfile.BadZipFile,RuntimeError,NotImplementedError) as exc:raise ProjectError(422,'INVALID_EVALUATION_RESULT','未接收评估：'+str(exc)) from exc

    def validate_report(self,detail,request):
        snap=request['snapshot']
        if detail['snapshot']!=snap or detail['sample_count']!=snap['sample_count'] or detail['runtime']!=snap['target']['runtime']['ultralytics']:raise ValueError('报告来源或运行库不符。')
        with self.storage.connect() as db:
            v=db.execute('SELECT manifest_json FROM task_data_versions WHERE id=?',(snap['data_version_id'],)).fetchone()
        data=json.loads(v[0]);expected={i['id']:i for i in data['images'] if i['subset']==snap['subset']}
        rows=detail['images']
        if not isinstance(rows,list) or len(rows)!=len(expected) or {i['id'] for i in rows}!=set(expected):raise ValueError('逐图结果不完整或有重复。')
        counts=[sum(b['class_id']==c['id'] for i in expected.values() for b in i['boxes']) for c in snap['classes']]
        def score(value):return value is None or type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
        if set(detail['metrics'])!=set(METRICS) or not all(score(v) for v in detail['metrics'].values()):raise ValueError('总指标无效。')
        if not any(counts) and any(v is not None for v in detail['metrics'].values()):raise ValueError('无真值时指标应不可用。')
        if len(detail['per_class'])!=len(counts):raise ValueError('分类别报告不完整。')
        for idx,row in enumerate(detail['per_class']):
            if row['index']!=idx or row['instances']!=counts[idx] or set(row['metrics'])!=set(METRICS) or not all(score(v) for v in row['metrics'].values()):raise ValueError('分类别样本或指标不符。')
            if not counts[idx] and any(v is not None for v in row['metrics'].values()):raise ValueError('无样本类别不能填零分。')
        for row in rows:
            image=expected[row['id']]
            if row['width']!=image['width'] or row['height']!=image['height'] or len(row['predictions'])>snap['requested_parameters']['max_det']:raise ValueError('图片尺寸或预测数不符。')
            for b in row['predictions']:
                if type(b['class_index']) is not int or not 0<=b['class_index']<len(counts) or not score(b['confidence']) or b['confidence'] is None:raise ValueError('预测类别或分数无效。')
                box=b['xyxy']
                if not isinstance(box,list) or len(box)!=4 or any(type(n) not in (int,float) or not math.isfinite(n) for n in box) or not 0<=box[0]<=box[2]<=image['width'] or not 0<=box[1]<=box[3]<=image['height']:raise ValueError('预测坐标超出原图。')
                if snap.get('task_type')=='segment':
                    from .prediction_masks import validate_segments
                    validate_segments(b,image['width'],image['height'])
        if snap.get('task_type')=='segment' and detail.get('metric_scope')!='mask':raise ValueError('分割评估必须返回掩膜指标，不能用框指标代替。')
        matrix=detail.get('confusion_matrix')
        if matrix is not None and (len(matrix)!=len(counts)+1 or any(len(row)!=len(counts)+1 or any(type(n) not in (int,float) or not math.isfinite(n) or n<0 or int(n)!=n for n in row) for row in matrix)):raise ValueError('混淆矩阵无效。')
        encoded(detail)

    def artifact(self,pid,tid,jid,name):
        job=self.training.get(pid,tid,jid);result=job['dispatch']['result'] if job['dispatch'] else None
        if name not in ARTIFACTS or not result or name not in result['artifacts']:raise ProjectError(404,'ARTIFACT_NOT_FOUND','评估材料不存在或运行库未生成。')
        path=self.owned(jid,'returned',name);spec=result['artifacts'][name]
        if not path.is_file() or path.stat().st_size!=spec['bytes'] or sha_file(path)!=spec['sha256']:raise ProjectError(409,'ARTIFACT_DAMAGED','评估材料缺失或校验不符。')
        return path,f'eval-{jid[:8]}-{name}','image/png' if name.endswith('.png') else 'application/json' if name.endswith('.json') else 'text/plain; charset=utf-8'

    def report(self,pid,tid,jid):
        path,_,_=self.artifact(pid,tid,jid,'report.json')
        value=json.loads(path.read_text(encoding='utf-8'))
        data=self.versions.read(pid,tid,value['snapshot']['data_version_id'])['manifest']
        truth={i['id']:i for i in data['images'] if i['subset']==value['snapshot']['subset']}
        return {**value,'images':[{**i,'truth':truth[i['id']]['boxes'],'filename':truth[i['id']]['filename']} for i in value['images']]}
