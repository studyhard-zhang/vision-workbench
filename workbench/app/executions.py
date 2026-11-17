"""Manual execution packages and bounded result ingestion. No local torch/pickle loading."""
import hashlib
import hmac
import io
import json
import re
import secrets
import shutil
import stat
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

from .images import comparable_path
from .storage import ProjectError,timestamp
from .versions import encoded,sha_file

MAX_RESULT_BYTES=128*1024**2  # Bounded best+last artifacts for the medium model.
ARTIFACTS={'console.log','args.yaml','results.csv','best.pt','last.pt'}


class ExecutionRepository:
    def execute(self,db,sql,parameters=()):
        if getattr(self,'kind','training')!='training':sql=sql.replace('training_',self.kind+'_')
        return db.execute(sql,parameters)
    def __init__(self,storage,training):
        self.storage=storage;self.training=training;self.versions=training.versions
        self.root=(storage.path.parent/'training').resolve();self.root.mkdir(exist_ok=True)
        self.staging=self.root/'.staging';self.staging.mkdir(exist_ok=True)
        self.slot=threading.BoundedSemaphore(1)

    def owned(self,*parts):
        path=self.root.joinpath(*parts).resolve();key,root=comparable_path(path),comparable_path(self.root)
        if not key.is_relative_to(root) or key==root:raise ProjectError(400,'INVALID_PATH','训练文件路径无效。')
        return path

    def dispatch(self,project_id,task_id,job_id):
        if not self.slot.acquire(blocking=False):raise ProjectError(409,'EXECUTION_BUSY','正在准备执行包或接收结果，请稍后重试。')
        destination=self.owned(job_id);published=False;committed=False
        try:
            job=self.training.get(project_id,task_id,job_id)
            if job['dispatch']:return {'job':job,'reused':True}
            if job['state']!='QUEUED':raise ProjectError(409,'TRAINING_NOT_QUEUED','只有待执行记录可生成执行包。')
            snapshot=job['snapshot'];target=snapshot['target'];dispatch_id=uuid.uuid4().hex
            if (self.storage.path.parent/'ssh-target-change.json').is_file():raise ProjectError(409,'SSH_TARGET_CHANGE_PENDING','计算目标切换待核对，未生成可运行的新执行包。')
            current=self.training.target()
            if target['id']!=current['id'] or target.get('verified_gpu_uuid')!=current['verified_gpu_uuid']:raise ProjectError(409,'SSH_RUN_TARGET_MISMATCH','此记录属于其他计算目标或缺少GPU身份，请保留历史并按配置新建。')
            with tempfile.TemporaryDirectory(dir=self.staging) as temporary:
                stage=Path(temporary)
                with self.versions.archive(project_id,task_id,job['version_id']) as (source,size,_):
                    self.versions.space(size*2+MAX_RESULT_BYTES)
                    with (stage/'dataset.zip').open('wb') as stream:shutil.copyfileobj(source,stream)
                from tools.verify_dataset import audit
                if snapshot.get('initial_model'):
                    from .models import ModelRepository
                    models=ModelRepository(self.storage,self)
                    initial=snapshot['initial_model']
                    source_model=models.get(project_id,task_id,initial['id'])
                    if source_model['manifest_sha256']!=initial['manifest_sha256']:raise ProjectError(409,'MODEL_DAMAGED','来源模型清单已变化。')
                    weights,_,_=models.artifact(project_id,task_id,initial['id'],'best.pt')
                    if sha_file(weights)!=initial['weights_sha256']:raise ProjectError(409,'MODEL_DAMAGED','来源模型权重校验不符。')
                    self.versions.space(weights.stat().st_size*2)
                    shutil.copyfile(weights,stage/'initial.pt')
                report=audit(stage/'dataset.zip')
                if not report['passed'] or report['manifest_sha256']!=snapshot['manifest_sha256']:raise ProjectError(409,'TRAINING_DATA_CHANGED','执行前数据核对不通过，未发出执行包。')
                scripts={name:(Path(__file__).resolve().parents[1]/'tools'/name).read_bytes().replace(b'\r\n',b'\n') for name in ('run_training_job.py','compute_probe.py','verify_dataset.py','training_config.py','polygon_geometry.py')}
                request={'format_version':1,'job_id':job_id,'dispatch_id':dispatch_id,'return_token':secrets.token_hex(32),
                         'snapshot':snapshot,'snapshot_sha256':job['snapshot_sha256'],
                         'expected_gpu_uuid':target['verified_gpu_uuid'],
                         'dataset_sha256':sha_file(stage/'dataset.zip'),'scripts':{k:hashlib.sha256(v).hexdigest() for k,v in scripts.items()}}
                readme=('在本实例数据盘解压到独立目录后执行：python -u run_training_job.py\n'
                        '自动写console.log并在终端显示进度，结束后回传return-result.zip。重复命令不会重复训练。\n'
                        '若曾被强杀且没有结果：确认原进程已结束后执行python -u run_training_job.py --recover-interrupted。\n'
                        '不要删除started.json或设备锁文件以绕开保护；无需安装库或下载权重。\n').encode('utf-8')
                with zipfile.ZipFile(stage/'execution.zip','w',compression=zipfile.ZIP_STORED) as z:
                    z.write(stage/'dataset.zip','dataset.zip');z.writestr('request.json',encoded(request));z.writestr('README.txt',readme)
                    for name,raw in scripts.items():z.writestr(name,raw)
                    if snapshot.get('initial_model'):z.write(stage/'initial.pt','initial.pt')
                package_sha=sha_file(stage/'execution.zip');(stage/'dataset.zip').unlink()
                with self.storage.connect() as db:
                    self.execute(db,'BEGIN IMMEDIATE');row=self.training.row(db,project_id,task_id,job_id)
                    old=self.execute(db,'SELECT 1 FROM training_dispatches WHERE job_id=?',(job_id,)).fetchone()
                    if old:return {'job':self.training.public(db,row),'reused':True}
                    if row['state']!='QUEUED':raise ProjectError(409,'TRAINING_NOT_QUEUED','记录状态已变化，请刷新。')
                    from .training import unavailable_target
                    if unavailable_target(db,target):raise ProjectError(409,'COMPUTE_INSTANCE_UNAVAILABLE','此设备已标记不可用，请选择其他已验证实例。')
                    active=self.execute(db,"SELECT job_id FROM training_dispatches WHERE state='AWAITING_RESULT' AND target_id=?",(target['id'],)).fetchone()
                    if active:raise ProjectError(409,'TARGET_RESERVED','该计算目标已有未回传结果的执行包，请先完成或在实例确认中断后回传结果。')
                    if self.execute(db,"SELECT 1 FROM evaluation_dispatches WHERE state='AWAITING_RESULT' AND target_id=?",(target['id'],)).fetchone():raise ProjectError(409,'TARGET_RESERVED','计算目标正由独立评估占用，请先完成或回收原评估。')
                    if self.execute(db,"SELECT 1 FROM evaluation_jobs WHERE state='QUEUED' AND created_at<? AND json_extract(snapshot_json,'$.target.id')=?",(job['created_at'],target['id'])).fetchone():raise ProjectError(409,'QUEUE_ORDER','请先处理更早提交的独立评估。')
                    if self.execute(db,"SELECT 1 FROM inference_dispatches WHERE state='AWAITING_RESULT' AND target_id=?",(target['id'],)).fetchone():raise ProjectError(409,'TARGET_RESERVED','计算目标正由图片推理占用，请先完成或回收原任务。')
                    if self.execute(db,"SELECT 1 FROM inference_jobs WHERE state='QUEUED' AND created_at<? AND json_extract(snapshot_json,'$.target.id')=?",(job['created_at'],target['id'])).fetchone():raise ProjectError(409,'QUEUE_ORDER','请先处理更早提交的图片推理。')
                    first=self.execute(db,"SELECT id FROM training_jobs WHERE state='QUEUED' AND json_extract(snapshot_json,'$.target.id')=? ORDER BY sequence LIMIT 1",(target['id'],)).fetchone()
                    if not first or first['id']!=job_id:raise ProjectError(409,'QUEUE_ORDER','请先处理排在前面的待执行记录。')
                    if destination.exists():raise ProjectError(409,'EXECUTION_FILES_EXIST','已有未提交的训练目录，请保留并联系检查，不覆盖文件。')
                    shutil.move(str(stage),str(destination));published=True;now=timestamp()
                    self.execute(db,"INSERT INTO training_dispatches VALUES(?,?,?,'AWAITING_RESULT',?,?,NULL,NULL,?,?)",(job_id,dispatch_id,target['id'],encoded(request).decode(),package_sha,now,now))
                    self.execute(db,"UPDATE training_jobs SET state='DISPATCHED',updated_at=? WHERE id=?",(now,job_id))
                committed=True
            return {'job':self.training.get(project_id,task_id,job_id),'reused':False}
        except OSError as exc:raise ProjectError(507 if exc.errno==28 else 503,'EXECUTION_IO_FAILED','训练文件操作未完成，请检查空间和文件权限。') from exc
        finally:
            if published and not committed and destination.exists():shutil.rmtree(self.owned(job_id))
            self.slot.release()

    def package(self,project_id,task_id,job_id):
        job=self.training.get(project_id,task_id,job_id)
        if not job['dispatch']:raise ProjectError(409,'EXECUTION_NOT_PREPARED','请先生成执行包。')
        path=self.owned(job_id,'execution.zip')
        if not path.is_file() or sha_file(path)!=job['dispatch']['package_sha256']:raise ProjectError(409,'EXECUTION_DAMAGED','执行包缺失或校验不符，请保留数据并检查存储。')
        return path,f'train-{job_id}.zip','application/zip'

    def validate_result(self,raw,request):
        try:
            if not 0<len(raw)<=MAX_RESULT_BYTES:raise ValueError('结果包最多128MiB。')
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                members=z.infolist();names=[m.filename for m in members]
                if len(names)!=len(set(names)) or not set(names)<={'receipt.json',*ARTIFACTS} or 'receipt.json' not in names:raise ValueError('结果包成员名称或重复路径无效。')
                if sum(m.file_size for m in members)>256*1024**2 or any(m.is_dir() or m.flag_bits&1 or stat.S_ISLNK(m.external_attr>>16) for m in members):raise ValueError('结果包体积或文件类型无效。')
                if z.getinfo('receipt.json').file_size>1024**2:raise ValueError('结果清单过大。')
                receipt=json.loads(z.read('receipt.json'))
                if not isinstance(receipt,dict) or receipt.get('format_version')!=1:raise ValueError('结果清单版本无效。')
                for key in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256'):
                    if receipt.get(key)!=request[key]:raise ValueError('结果不属于该执行包：'+key)
                token=receipt.get('return_token')
                if not isinstance(token,str) or not hmac.compare_digest(token,request['return_token']):raise ValueError('回传标识不符。')
                report=receipt.get('report');artifacts=receipt.get('artifacts')
                if not isinstance(report,dict) or not isinstance(artifacts,dict) or set(artifacts)!=set(names)-{'receipt.json'}:raise ValueError('缺少报告或文件清单。')
                if report.get('status') not in ('completed','failed','interrupted'):raise ValueError('执行结果状态无效。')
                files={}
                for name,spec in artifacts.items():
                    data=z.read(name)
                    if not isinstance(spec,dict) or spec.get('bytes')!=len(data) or spec.get('sha256')!=hashlib.sha256(data).hexdigest():raise ValueError('产物校验不符：'+name)
                    files[name]=data
                if not isinstance(report.get('error',''),str) or len(report.get('error',''))>4000:raise ValueError('错误说明无效。')
                diagnostic=report.get('diagnostic')
                if diagnostic is not None:
                    if not isinstance(diagnostic,dict) or set(diagnostic)!={'code','title','next_step','phase'}:raise ValueError('故障分类格式无效。')
                    if not isinstance(diagnostic['code'],str) or not re.fullmatch('[A-Z_]{3,60}',diagnostic['code']) or report.get('error_code')!=diagnostic['code']:raise ValueError('故障分类标识不一致。')
                    if any(not isinstance(diagnostic[k],str) or not 1<=len(diagnostic[k])<=limit for k,limit in (('title',120),('next_step',600),('phase',40))):raise ValueError('故障分类说明无效。')
                if report['status']=='completed':
                    count=report.get('epochs_completed');snapshot=request['snapshot']
                    if report.get('training_executed') is not True or report.get('forward_passed') is not True:raise ValueError('没有实际训练/前向通过记录。')
                    if report.get('lock_check_passed') is not True:raise ValueError('没有实例跨进程锁检查通过记录。')
                    if type(count) is not int or not 1<=count<=snapshot['requested_parameters']['epochs'] or type(report.get('batches_completed')) is not int or report['batches_completed']<1:raise ValueError('训练轮数/批次无效。')
                    if count<snapshot['requested_parameters']['epochs'] and report.get('completion_reason')!='early_stopped':raise ValueError('未完成配置轮数且没有早停说明。')
                    expected_weights=snapshot.get('initial_model',{}).get('weights_sha256',snapshot['target']['weights']['sha256'])
                    if report.get('requested_parameters')!=snapshot['requested_parameters'] or report.get('weights_sha256')!=expected_weights:raise ValueError('训练配置或原权重不符。')
                    if snapshot.get('initial_model') and report.get('initial_model')!=snapshot['initial_model']:raise ValueError('迭代训练来源证据不符。')
                    actual=report.get('actual_parameters')
                    auto=snapshot['requested_parameters']['batch']==-1
                    if not isinstance(actual,dict) or any(actual.get(k)!=snapshot['requested_parameters'][k] for k in ('epochs','imgsz','batch','workers','amp','seed') if not (auto and k=='batch')):raise ValueError('缺少一致的实际解析参数。')
                    if auto and (report.get('autobatch_estimation_passed') is not True or report.get('autobatch_memory_fraction')!=.6 or type(report.get('resolved_batch_size')) is not int or report['resolved_batch_size']<1 or actual.get('batch') not in (-1,report['resolved_batch_size'])):raise ValueError('缺少AutoBatch成功估算和实际批大小记录。')
                    derived=set()
                    if snapshot.get('configuration',{}).get('profile') in ('official26','custom26') and snapshot['requested_parameters'].get('optimizer')=='auto':
                        # The official auto optimizer resolves these fields after parsing defaults.
                        derived={'optimizer','lr0','momentum','warmup_bias_lr'}
                        if actual.get('optimizer')!=report.get('actual_optimizer'):raise ValueError('实际优化器记录不一致。')
                    if snapshot.get('configuration') and any((str(actual.get(k))!=str(v) if k=='device' else actual.get(k)!=v) for k,v in snapshot['requested_parameters'].items() if k not in derived and not (auto and k=='batch')):raise ValueError('受控方案实际解析参数与请求不一致。')
                    if snapshot.get('configuration',{}).get('optimizer')=='AdamW' and report.get('actual_optimizer')!='AdamW':raise ValueError('实际优化器与指定AdamW不符。')
                    if snapshot.get('configuration') and report.get('geometry_check_passed') is not True:raise ValueError('缺少预览与实例几何坐标核对通过记录。')
                    if snapshot.get('task_type')=='segment' and (report.get('mask_geometry_passed') is not True or report.get('actual_parameters',{}).get('task')!='segment'):raise ValueError('缺少分割顶点几何核对或实际分割训练证据。')
                    if not isinstance(report.get('actual_optimizer'),str) or not 1<=len(report['actual_optimizer'])<=60:raise ValueError('缺少实际优化器。')
                    environment=report.get('environment',{})
                    if environment.get('ready') is not True or environment.get('cuda_visible_devices')!=snapshot['target']['gpu_selector'] or environment.get('gpu',{}).get('logical_index')!=0:raise ValueError('缺少指定GPU实际环境通过记录。')
                    if request['expected_gpu_uuid'] not in environment.get('nvidia_smi',''):raise ValueError('实例GPU身份与派发目标不符。')
                    for name in ('best.pt','last.pt'):
                        if not files.get(name) or report.get('reload_passed',{}).get(name) is not True:raise ValueError('权重缺失或实例未通过重载。')
                    if not files.get('args.yaml') or not files.get('results.csv'):raise ValueError('缺少实际参数或训练指标。')
                receipt.pop('return_token',None)
                # JSON serialization also rejects NaN/Infinity in untrusted reports.
                encoded(receipt)
                return receipt,files
        except (ValueError,TypeError,KeyError,AttributeError,zipfile.BadZipFile,RuntimeError,NotImplementedError) as exc:
            raise ProjectError(422,'INVALID_TRAINING_RESULT','未接收结果：'+str(exc)) from exc

    def receive(self,project_id,task_id,job_id,raw,*,remote_resources_verified=False):
        result=self._receive(project_id,task_id,job_id,raw,remote_resources_verified=remote_resources_verified)
        callback=getattr(self,'on_success',None)
        if getattr(self,'kind','training')=='training' and result['job']['state']=='SUCCEEDED' and callback:
            publication=callback(project_id,task_id,job_id)
            result['job']=self.training.get(project_id,task_id,job_id)
            result.update(publication)
        return result

    def _receive(self,project_id,task_id,job_id,raw,*,remote_resources_verified=False):
        job=self.training.get(project_id,task_id,job_id);digest=hashlib.sha256(raw).hexdigest()
        if not job['dispatch']:raise ProjectError(409,'EXECUTION_NOT_PREPARED','此记录尚未发出执行包。')
        with self.storage.connect() as db:dispatch=self.execute(db,'SELECT * FROM training_dispatches WHERE job_id=?',(job_id,)).fetchone();request=json.loads(dispatch['request_json'])
        if dispatch['result_sha256']:
            if dispatch['result_sha256']==digest:return {'job':job,'reused':True}
            raise ProjectError(409,'RESULT_ALREADY_IMPORTED','已有已确认结果，不能用另一结果覆盖。')
        if not remote_resources_verified:
            with self.storage.connect() as db:remote=self.execute(db,'SELECT 1 FROM training_remote_sessions WHERE job_id=?',(job_id,)).fetchone()
            if remote:raise ProjectError(409,'SSH_RESOURCE_PROOF_REQUIRED','此训练由SSH管理，请恢复连接核实原进程和设备锁释放后自动接收结果；手动回传不能跳过资源核对。')
        receipt,files=self.validate_result(raw,request)
        if not self.slot.acquire(blocking=False):raise ProjectError(409,'EXECUTION_BUSY','正在准备或接收训练文件，请稍后重试。')
        published=False;committed=False;destination=self.owned(job_id,'returned')
        try:
            self.versions.space(len(raw)+sum(len(v) for v in files.values()))
            with tempfile.TemporaryDirectory(dir=self.staging) as temporary:
                stage=Path(temporary)
                (stage/'result.zip').write_bytes(raw)
                for name,data in files.items():(stage/name).write_bytes(data)
                (stage/'receipt.json').write_bytes(encoded(receipt))
                with self.storage.connect() as db:
                    self.execute(db,'BEGIN IMMEDIATE');row=self.training.row(db,project_id,task_id,job_id)
                    actual=self.execute(db,'SELECT * FROM training_dispatches WHERE job_id=?',(job_id,)).fetchone()
                    if actual['result_sha256']:
                        if actual['result_sha256']!=digest:raise ProjectError(409,'RESULT_ALREADY_IMPORTED','结果已由另一请求提交，不能覆盖。')
                        return {'job':self.training.public(db,row),'reused':True}
                    if actual['state']!='AWAITING_RESULT' or row['state'] not in ('DISPATCHED','RUNNING'):raise ProjectError(409,'RESULT_STATE_CONFLICT','记录状态已变化，请刷新。')
                    if destination.exists():raise ProjectError(409,'RESULT_FILES_EXIST','存在未提交结果目录，请保留并检查。')
                    stage.rename(destination);published=True
                    state={'completed':'SUCCEEDED','failed':'FAILED','interrupted':'INTERRUPTED'}[receipt['report']['status']];now=timestamp()
                    self.execute(db,'UPDATE training_dispatches SET state=?,result_sha256=?,result_json=?,updated_at=? WHERE job_id=?',(state,digest,encoded(receipt).decode(),now,job_id))
                    self.execute(db,'UPDATE training_jobs SET state=?,updated_at=? WHERE id=?',(state,now,job_id))
                committed=True
            return {'job':self.training.get(project_id,task_id,job_id),'reused':False}
        except OSError as exc:raise ProjectError(507 if exc.errno==28 else 503,'RESULT_IO_FAILED','结果未提交，请检查空间和文件权限后重试。') from exc
        finally:
            if published and not committed and destination.exists():shutil.rmtree(self.owned(job_id,'returned'))
            self.slot.release()

    def artifact(self,project_id,task_id,job_id,name):
        job=self.training.get(project_id,task_id,job_id)
        result=job['dispatch']['result'] if job['dispatch'] else None
        if name not in ARTIFACTS or not result or name not in result['artifacts']:raise ProjectError(404,'ARTIFACT_NOT_FOUND','训练产物不存在。')
        path=self.owned(job_id,'returned',name);spec=result['artifacts'][name]
        if not path.is_file() or path.stat().st_size!=spec['bytes'] or sha_file(path)!=spec['sha256']:raise ProjectError(409,'ARTIFACT_DAMAGED','训练产物缺失或校验不符。')
        return path,f'{job_id[:8]}-{name}','application/octet-stream'
