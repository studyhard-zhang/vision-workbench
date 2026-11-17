"""N-09 persisted requests. No executor, GPU imports or remote connection in this node."""
import hashlib
import json
import re
import unicodedata
import uuid

from .storage import ProjectError,timestamp
from .versions import encoded,VersionRepository
from tools.training_config import configuration,parameters,PROFILES,SEGMENT_CONFIG,PRECISION_CONFIG,OFFICIAL_CONFIG
from .training_modes import MODES,model_name,high_accuracy_target
from .compute_instances import active,read

TARGET={
    'id':'unconfigured-device','name':'尚未连接训练设备','kind':'self_hosted',
    'transport':'manual_terminal','gpu_selector':'0','expected_gpu':'NVIDIA GPU',
    'weights':{'id':'yolo11n-compatible','name':'YOLO11n · 兼容权重',
               'path':'/root/.vision-workbench/weights/yolo11n.pt',
               'sha256':'8b78d4bb51429574665967a314f9f0410d0e786bfab039c87fcf89165e218cd4'},
    'runtime':{'python':'3.12.3','torch':'2.3.0+cu121','torchvision':'0.18.0+cu121','ultralytics':'8.3.241'},
    'verified_run_id':None,'verified_at':None,
    'verified_gpu_uuid':'GPU-unconfigured',
    'executor_connected':False,
}
PARAMETERS={'imgsz':640,'batch':1,'workers':0,'cache':False,'amp':False,'plots':False,'seed':42,
            'save':True,'optimizer':'auto','device':0}


def unavailable_target(db,target):
    return db.execute("SELECT 1 FROM training_remote_sessions s JOIN training_jobs j ON j.id=s.job_id WHERE json_extract(s.progress_json,'$.local_abandonment.remote_stop_confirmed')=0 AND coalesce(json_extract(s.progress_json,'$.local_abandonment_resolution.resources_released'),0)!=1 AND json_extract(j.snapshot_json,'$.target.verified_gpu_uuid')=? AND json_extract(j.snapshot_json,'$.target.id')=? LIMIT 1",(target['verified_gpu_uuid'],target['id'])).fetchone() is not None


class TrainingRepository:
    def __init__(self,storage,versions=None):
        self.storage=storage;self.versions=versions or VersionRepository(storage)

    def target(self, legacy=False):
        record = active(self.storage.path.parent) if not legacy else None
        if record:
            return record['target']
        path=self.storage.path.parent/'compute-target.json'
        if not path.is_file():return json.loads(json.dumps(TARGET))
        try:
            target=json.loads(path.read_text(encoding='utf-8'))
            if not re.fullmatch('[a-z0-9-]{3,80}',target['id']) or target['kind'] not in ('autodl_instance','self_hosted') or target['gpu_selector']!='0' or not re.fullmatch('GPU-[A-Za-z0-9-]+',target['verified_gpu_uuid']):raise ValueError('invalid target')
            if target['weights']['sha256']!=TARGET['weights']['sha256'] or target['runtime']!=TARGET['runtime']:raise ValueError('runtime/weights require explicit compatibility verification')
            return target
        except (ValueError,KeyError,TypeError) as exc:raise ProjectError(503,'COMPUTE_TARGET_INVALID','计算目标定义不完整或运行库尚未核对，未提交训练。') from exc

    def mode_target(self,task_type='detect',training_mode='standard',require_verified=True):
        if training_mode=='high_accuracy':return high_accuracy_target(self,task_type,require_verified)
        if training_mode!='standard':raise ProjectError(422,'INVALID_TRAINING_MODE','请选择通用模式或高精度模式。')
        return self.segment_target(require_verified=require_verified)[0] if task_type=='segment' else self.target()

    def capabilities(self,task_type='detect',training_mode='standard'):
        passed={key:[] for key in PROFILES};optimizers={key:set() for key in PROFILES};target=self.target()
        try:mode_target=self.mode_target(task_type,training_mode)
        except ProjectError:return passed,optimizers
        record = active(self.storage.path.parent)
        if record and record['target']['id']==target['id']:
            filename = mode_target['weights']['path'].rsplit('/',1)[-1]
            proof = record['tests'].get(filename)
            if proof:
                profile='official26' if filename.startswith('yolo26') else 'precision'
                passed[profile].append(proof['run_id'])
                optimizers[profile].add('auto')
                if profile=='official26':
                    passed['custom26'].append(proof['run_id']);optimizers['custom26'].update(('auto','AdamW'))
                    return passed,optimizers
                optimizers['precision'].add('auto')
                if task_type=='segment' and training_mode=='standard':
                    passed['geometry640'].append(proof['run_id'])
                    optimizers['geometry640'].add('auto')
            return passed,optimizers
        with self.storage.connect() as db:
            for row in db.execute("SELECT j.id,j.snapshot_json,d.result_json FROM training_jobs j JOIN training_dispatches d ON d.job_id=j.id WHERE j.state='SUCCEEDED' AND d.state='SUCCEEDED'"):
                snapshot=json.loads(row['snapshot_json']);cfg=snapshot.get('configuration')
                if snapshot.get('task_type','detect')!=task_type:continue
                if snapshot.get('training_mode','standard')!=training_mode or snapshot['target']['weights']['sha256']!=mode_target['weights']['sha256']:continue
                if not cfg or cfg.get('profile') not in passed:continue
                if snapshot['target']['id']!=target['id'] or snapshot['target'].get('verified_gpu_uuid')!=target['verified_gpu_uuid']:continue
                receipt=json.loads(row['result_json']);report=receipt['report']
                if all(report.get('reload_passed',{}).get(n) is True for n in ('best.pt','last.pt')):
                    passed[cfg['profile']].append(row['id']);optimizers[cfg['profile']].add(cfg.get('optimizer','auto'))
        evidence=self.storage.path.parent/'compute-evidence'/f'n21-{task_type}.json'
        if training_mode=='standard' and evidence.is_file():
            try:
                proof=json.loads(evidence.read_text('utf-8'));weights=self.segment_target()[0]['weights'] if task_type=='segment' else target['weights']
                if proof.get('verified') is True and proof['gpu_uuid']==target['verified_gpu_uuid'] and proof['runtime']==target['runtime'] and proof['weights_sha256']==weights['sha256'] and proof['requested_configuration']==PRECISION_CONFIG:
                    passed['precision'].append(proof['run_id']);optimizers['precision'].add('auto')
            except (ValueError,KeyError,TypeError,ProjectError):pass
        # Isolated validation runs can prove a new target without adding QA jobs
        # to the production database. Old target evidence never unlocks new GPUs.
        evidence=self.storage.path.parent/'compute-evidence'/'training-validation.json'
        if evidence.is_file():
            try:
                proofs=json.loads(evidence.read_text('utf-8'))
                weights=mode_target['weights']
                for proof in proofs:
                    cfg=configuration(proof['configuration']);report=proof['report'];identity=proof['target']
                    if proof.get('task_type')!=task_type or proof.get('verified') is not True:continue
                    if identity['id']!=target['id'] or identity['verified_gpu_uuid']!=target['verified_gpu_uuid'] or identity['runtime']!=target['runtime'] or identity['weights']['sha256']!=weights['sha256']:continue
                    if report.get('status')!='completed' or report.get('training_executed') is not True or report.get('forward_passed') is not True or report.get('geometry_check_passed') is not True:continue
                    if task_type=='segment' and report.get('mask_geometry_passed') is not True:continue
                    if report.get('weights_sha256')!=weights['sha256'] or report.get('requested_parameters')!=parameters(cfg,1):continue
                    env=report.get('environment',{})
                    if env.get('ready') is not True or target['verified_gpu_uuid'] not in env.get('nvidia_smi',''):continue
                    if cfg.get('batch')==-1 and (report.get('autobatch_estimation_passed') is not True or type(report.get('resolved_batch_size')) is not int or report['resolved_batch_size']<1):continue
                    if not all(report.get('reload_passed',{}).get(n) is True for n in ('best.pt','last.pt')):continue
                    if proof.get('remote_process_exited') is not True or proof.get('resources',{}).get('checked') is not True or proof['resources'].get('free') is not True:continue
                    passed[cfg['profile']].append(proof['run_id']);optimizers[cfg['profile']].add(cfg['optimizer'])
            except (OSError,ValueError,KeyError,TypeError,AttributeError,ProjectError):pass
        return passed,optimizers

    def segment_target(self, *, require_verified=True):
        target=self.target();path=self.storage.path.parent/'segment-training.json'
        record = active(self.storage.path.parent)
        if record and record['target']['id']==target['id']:
            filename='yolo26n-seg.pt' if 'yolo26n-seg.pt' in record['weights'] else 'yolo11n-seg.pt'
            if filename not in record['weights']:raise ProjectError(409,'SEGMENT_NOT_VERIFIED','当前环境缺少分割权重，请自行准备后重新校验。')
            return {**target,'weights':record['weights'][filename]}, record['tests'][filename]
        try:
            proof=json.loads(path.read_text('utf-8'));weight=proof['weights']
            if proof['gpu_uuid']!=target['verified_gpu_uuid'] or proof['runtime']!=target['runtime'] or not re.fullmatch('[0-9a-f]{64}',weight['sha256']) or not weight['path'].startswith('/') or not weight['path'].endswith('/yolo11n-seg.pt') or (require_verified and proof.get('verified') is not True):raise ValueError('unverified')
        except (OSError,ValueError,KeyError,TypeError) as exc:raise ProjectError(409,'SEGMENT_NOT_VERIFIED','分割训练尚未在当前GPU完成验证，请先完成计算环境核对。') from exc
        return {**target,'weights':weight},proof

    def options(self,project_id=None,task_id=None,version_id=None,training_mode='standard'):
        if training_mode not in MODES:raise ProjectError(422,'INVALID_TRAINING_MODE','训练模式无效。')
        task_type=self.versions.tasks.get(project_id,task_id)['task_type'] if task_id else 'detect'
        passed,optimizers=self.capabilities(task_type,training_mode);target=self.target();ready=True
        if task_type=='segment':
            try:target,_=self.segment_target()
            except ProjectError:ready=False
            if ready and training_mode=='standard' and not target['weights']['path'].rsplit('/',1)[-1].startswith('yolo26'):
                passed['geometry640'].append('segment-640-verified');optimizers['geometry640'].add('auto')
        if training_mode=='high_accuracy':
            try:target=self.mode_target(task_type,training_mode)
            except ProjectError:ready=False
        if (self.storage.path.parent/'ssh-target-change.json').is_file():target.update(enabled=False,reason='新实例正在核对，旧计算目标保留；暂不提交训练')
        with self.storage.connect() as db:
            if unavailable_target(db,target):target={**target,'enabled':False,'unavailable':True,'reason':'设备中止状态尚待核实，请重新核对并启用设备。'}
        result={'task_type':task_type,'training_mode':training_mode,'model_name':model_name(task_type),'training_modes':[{'id':key,'name':name,'enabled':self.mode_ready(task_type,key)} for key,name in MODES.items()],'segment_verified':ready,'default_configuration':dict(OFFICIAL_CONFIG),'targets':[target],
                'validation_basis':target.get('verification_kind','training'),
                'future_targets':[{'id':'self-hosted','name':'自有训练机','enabled':False,'reason':'环境重新验证后接入'}],
                'parameters':parameters(OFFICIAL_CONFIG,100),'default_epochs':100,'epochs_min':1,'epochs_max':300,
                'executor_connected':False,'manual_execution_available':True,
                'configuration_profiles':[{'id':key,**value,'enabled':bool(passed[key]),'evidence_runs':passed[key],'verified_optimizers':sorted(optimizers[key])} for key,value in PROFILES.items()],
                'verified_optimizers':sorted(set().union(*optimizers.values())), 'legacy_enabled':False if training_mode=='high_accuracy' else ready if task_type=='segment' else self.legacy_verified()}
        if version_id:
            from .training_recommendations import recommend
            result['recommendation']=recommend(self,project_id,task_id,version_id,training_mode)
        return result

    def mode_ready(self,task_type,mode):
        try:self.mode_target(task_type,mode);return True
        except ProjectError:return False

    def legacy_verified(self):
        target=self.target()
        if target.get('managed_environment'):return False
        if target['id']==TARGET['id'] and target['verified_gpu_uuid']==TARGET['verified_gpu_uuid']:return True
        with self.storage.connect() as db:
            for row in db.execute("SELECT j.snapshot_json,d.result_json FROM training_jobs j JOIN training_dispatches d ON d.job_id=j.id WHERE j.state='SUCCEEDED' AND d.state='SUCCEEDED'"):
                snapshot=json.loads(row['snapshot_json']);identity=snapshot['target']
                if snapshot.get('configuration') or identity['id']!=target['id'] or identity.get('verified_gpu_uuid')!=target['verified_gpu_uuid']:continue
                report=json.loads(row['result_json'])['report']
                if all(report.get('reload_passed',{}).get(name) is True for name in ('best.pt','last.pt')):return True
        return False

    def public(self,db,row):
        value={k:row[k] for k in ('id','sequence','task_id','version_id','name','state','created_at','updated_at','snapshot_sha256')}
        value['snapshot']=json.loads(row['snapshot_json'])
        model=db.execute('SELECT id,number FROM model_versions WHERE training_run_id=?',(row['id'],)).fetchone()
        value['model_version']=dict(model) if model else None
        value['queue_position']=db.execute("SELECT count(*) FROM training_jobs WHERE state='QUEUED' AND sequence<=? AND json_extract(snapshot_json,'$.target.id')=?",(row['sequence'],value['snapshot']['target']['id'])).fetchone()[0] if row['state']=='QUEUED' else None
        value['execution_message']={'QUEUED':'等待生成执行包，尚未在 GPU 运行','CANCELLED':'已取消，未执行',
            'DISPATCHED':'已生成执行包并预留目标，等待实例运行及结果回传',
            'RUNNING':'实例进程已启动；以最后更新时间和实际阶段为准',
            'SUCCEEDED':'实例报告训练完成，结果文件校验通过；模型将自动生成',
            'FAILED':'实例运行失败，查看回传日志后可按配置新建',
            'INTERRUPTED':'实例执行已中断并确认锁已释放，可按配置新建'}[row['state']]
        if model:value['execution_message']=f'模型 v{model["number"]} · 可查看模型或直接进行图片推理'
        dispatch=db.execute('SELECT * FROM training_dispatches WHERE job_id=?',(row['id'],)).fetchone()
        value['dispatch']=None if dispatch is None else {k:dispatch[k] for k in ('id','state','package_sha256','created_at','updated_at')}
        if dispatch is not None:value['dispatch']['result']=json.loads(dispatch['result_json']) if dispatch['result_json'] else None
        remote=db.execute('SELECT phase,cancel_requested,progress_json,error,updated_at FROM training_remote_sessions WHERE job_id=?',(row['id'],)).fetchone()
        value['remote']=None if remote is None else {**dict(remote),'progress':json.loads(remote['progress_json'])}
        if value['remote']:value['remote'].pop('progress_json')
        visibility=db.execute('SELECT deleted_at,revision FROM training_record_visibility WHERE job_id=?',(row['id'],)).fetchone()
        value['deleted_at']=visibility['deleted_at'] if visibility else None
        value['visibility_revision']=visibility['revision'] if visibility else 0
        value['can_delete']=not value['deleted_at'] and row['state'] not in ('DISPATCHED','RUNNING') and not (dispatch and dispatch['state']=='AWAITING_RESULT')
        value['can_finish_stop']=not value['deleted_at'] and row['state'] in ('DISPATCHED','RUNNING') and bool(value['remote'] and value['remote']['cancel_requested'])
        value['can_abandon']=not value['deleted_at'] and row['state'] in ('DISPATCHED','RUNNING') and bool(dispatch and dispatch['state']=='AWAITING_RESULT' and value['remote'] and value['remote']['phase'] in ('ERROR','PAUSED'))
        if value['remote'] and value['remote']['progress'].get('local_abandonment'):
            value['execution_message']='设备不可用，本地已结束跟踪；远端停止状态未核实。'
            if value['remote']['progress'].get('local_abandonment_resolution'):value['execution_message']='原任务进程和锁已核实释放，设备可继续使用。'
        return value

    def row(self,db,project_id,task_id,job_id):
        self.versions.tasks.get(project_id,task_id,db)
        row=db.execute('SELECT * FROM training_jobs WHERE task_id=? AND id=?',(task_id,job_id)).fetchone()
        if row is None:raise ProjectError(404,'TRAINING_NOT_FOUND','训练记录不存在或不属于当前任务。')
        return row

    def get(self,project_id,task_id,job_id):
        with self.storage.connect() as db:return self.public(db,self.row(db,project_id,task_id,job_id))

    def list(self,project_id,task_id,view='active'):
        if view not in ('active','trash'):raise ProjectError(422,'INVALID_TRAINING_VIEW','训练列表范围无效。')
        with self.storage.connect() as db:
            self.versions.tasks.get(project_id,task_id,db)
            condition='IS NULL' if view=='active' else 'IS NOT NULL'
            rows=db.execute(f'SELECT j.* FROM training_jobs j LEFT JOIN training_record_visibility v ON v.job_id=j.id WHERE j.task_id=? AND v.deleted_at {condition} ORDER BY j.sequence DESC',(task_id,)).fetchall()
            return [self.public(db,r) for r in rows]

    def change_visibility(self,project_id,task_id,records,restore=False):
        if not isinstance(records,list) or not 1<=len(records)<=200:raise ProjectError(422,'INVALID_TRAINING_SELECTION','请选择1至200条训练记录。')
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE');checked=[];ids=set()
            for item in records:
                if not isinstance(item,dict) or set(item)!={'id','confirmation_name','visibility_revision'} or not isinstance(item['id'],str) or not re.fullmatch('[0-9a-f]{32}',item['id']) or item['id'] in ids or type(item['visibility_revision']) is not int or item['visibility_revision']<0:
                    raise ProjectError(422,'INVALID_TRAINING_SELECTION','训练记录标识或修订无效。')
                ids.add(item['id']);row=self.row(db,project_id,task_id,item['id']);job=self.public(db,row)
                if item['confirmation_name']!=job['name']:raise ProjectError(409,'TRAINING_CONFIRMATION_CHANGED','训练名称核对不符，请刷新列表。')
                if (restore and not job['deleted_at']) or (not restore and job['deleted_at']):continue
                if item['visibility_revision']!=job['visibility_revision']:raise ProjectError(409,'TRAINING_VISIBILITY_CHANGED','此记录已被删除或恢复过，请刷新后重试。')
                if not restore and not job['can_delete']:raise ProjectError(409,'TRAINING_ACTIVE','训练正在执行或尚未收尾，不能删除；请先中止并确认资源释放。')
                checked.append((row,job))
            for row,job in checked:
                if not restore and row['state']=='QUEUED':db.execute("UPDATE training_jobs SET state='CANCELLED',updated_at=? WHERE id=?",(timestamp(),row['id']))
                now=timestamp()
                db.execute('INSERT INTO training_record_visibility(job_id,deleted_at,revision,updated_at) VALUES(?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET deleted_at=excluded.deleted_at,revision=excluded.revision,updated_at=excluded.updated_at',
                    (row['id'],None if restore else now,job['visibility_revision']+1,now))
            return {'changed':len(checked),'ids':[r['id'] for r,_ in checked],'restored':restore,'files_removed':False}

    @staticmethod
    def validate(values,target_id=None):
        fields={}
        if set(values)-{'name','version_id','target_id','epochs','idempotency_key','configuration','training_mode','execution_method','source_model_id'}:
            raise ProjectError(422,'UNKNOWN_FIELDS','训练请求含不支持的字段。')
        name=values.get('name')
        if not isinstance(values.get('training_mode','standard'),str) or values.get('training_mode','standard') not in MODES:fields['training_mode']='请选择通用模式或高精度模式。'
        if values.get('execution_method','ssh') not in ('ssh','manual'):fields['execution_method']='请选择SSH GPU或手动执行包。'
        if not isinstance(name,str) or not 1<=len(name.strip())<=60 or any(unicodedata.category(c)=='Cc' for c in name):fields['name']='训练名称需为 1–60 个字符，不能含控制字符。'
        if not isinstance(values.get('version_id'),str) or not re.fullmatch('[0-9a-f]{32}',values['version_id']):fields['version_id']='请选择已发布的数据版本。'
        if values.get('target_id')!=(target_id or TARGET['id']):fields['target_id']='请选择当前已核对的计算目标。'
        if type(values.get('epochs')) is not int or not 1<=values['epochs']<=300:fields['epochs']='训练轮数需为 1–300 的整数。'
        if not isinstance(values.get('idempotency_key'),str) or not re.fullmatch('[0-9a-f]{32}',values['idempotency_key']):fields['idempotency_key']='提交标识无效，请重新打开配置。'
        if fields:raise ProjectError(422,'INVALID_TRAINING','请检查训练配置。',fields)
        if 'source_model_id' in values and (not isinstance(values['source_model_id'],str) or not re.fullmatch('[0-9a-f]{32}',values['source_model_id'])):
            raise ProjectError(422,'INVALID_SOURCE_MODEL','请选择用于迭代训练的模型。')
        if 'configuration' in values:
            try:values={**values,'configuration':configuration(values['configuration'])}
            except ValueError as exc:raise ProjectError(422,'INVALID_CONFIGURATION',str(exc),{'configuration':str(exc)}) from exc
        return {**values,'name':name.strip()}

    def reused(self,db,task_id,key,digest):
        row=db.execute('SELECT * FROM training_jobs WHERE task_id=? AND idempotency_key=?',(task_id,key)).fetchone()
        if row is None:return None
        if row['request_sha256']!=digest:raise ProjectError(409,'IDEMPOTENCY_CONFLICT','同一次提交的配置已变化，请重新打开配置后提交。')
        return {'job':self.public(db,row),'reused':True}

    def create(self,project_id,task_id,values,*,allow_validation=False):
        task=self.versions.tasks.get(project_id,task_id);segment=task['task_type']=='segment'
        mode=values.get('training_mode','standard')
        target=self.mode_target(task['task_type'],mode,require_verified=not allow_validation)
        family26=target['weights']['path'].rsplit('/',1)[-1].startswith('yolo26')
        if (values.get('configuration') or {}).get('profile') in ('official26','custom26') and not family26 and not allow_validation:
            raise ProjectError(409,'YOLO26_NOT_VERIFIED','新训练使用YOLO26，请先连接并验证新实例。')
        if family26 and not values.get('configuration'):values={**values,'configuration':OFFICIAL_CONFIG}
        if segment:
            cfg=values.get('configuration')
            if cfg not in (None,SEGMENT_CONFIG) and (not isinstance(cfg,dict) or cfg.get('profile') not in ('precision','official26','custom26')):raise ProjectError(422,'INVALID_CONFIGURATION','分割任务请选择精度优先或640兼容方案。')
            if cfg is None:values={**values,'configuration':SEGMENT_CONFIG}
        values=self.validate(values,target['id']);digest=hashlib.sha256(encoded(values)).hexdigest()
        with self.storage.connect() as db:
            self.versions.row(db,project_id,task_id,values['version_id'])
            old=self.reused(db,task_id,values['idempotency_key'],digest)
            if old:return old
        if (self.storage.path.parent/'ssh-target-change.json').is_file():raise ProjectError(409,'SSH_TARGET_CHANGE_PENDING','新实例尚未完成身份及环境核对，未创建新训练。原记录与模型保留。')
        if not family26 and not allow_validation and any('yolo26n.pt' in read(path).get('weights',{}) for path in (self.storage.path.parent/'compute-instances').glob('*.json')):
            raise ProjectError(409,'YOLO26_NOT_SELECTED','YOLO26实例已通过验证，请在原任务收尾后启用该训练目标。')
        if target.get('enabled') is False and not allow_validation:raise ProjectError(409,'COMPUTE_TARGET_NOT_VALIDATED','新计算目标尚未通过计算验证，普通训练暂不可提交。')
        with self.storage.connect() as db:
            if unavailable_target(db,target):raise ProjectError(409,'COMPUTE_INSTANCE_UNAVAILABLE','设备中止状态尚待核实，请在“训练设备与连接”中重新核对并启用设备。')
        cfg=values.get('configuration')
        if mode=='high_accuracy' and not cfg:raise ProjectError(422,'INVALID_CONFIGURATION','高精度模式请选择自定义方案，不能沿用旧默认增强。')
        if not cfg and not allow_validation and not self.legacy_verified():raise ProjectError(409,'CONFIGURATION_NOT_VERIFIED','原默认增强方案尚未在当前GPU验证，请选择已有实测的训练方案。')
        if cfg and not allow_validation and (family26 or not segment or cfg['profile'] in ('precision','official26')):
            passed,optimizers=self.capabilities(task['task_type'],mode)
            if not passed[cfg['profile']] or cfg.get('optimizer','auto') not in optimizers[cfg['profile']]:
                raise ProjectError(409,'CONFIGURATION_NOT_VERIFIED','此训练方案尚未通过实例验证，可以预览；暂不能提交训练。')
        report=self.versions.readiness(project_id,task_id,values['version_id'])
        if not report['passed']:raise ProjectError(409,'TRAINING_DATA_INVALID','版本数据核对未通过，不能提交训练。')
        initial_model=None
        if values.get('source_model_id'):
            from .models import ModelRepository
            from .executions import ExecutionRepository
            models=ModelRepository(self.storage,ExecutionRepository(self.storage,self))
            model=models.get(project_id,task_id,values['source_model_id'])
            if not model['available']:raise ProjectError(409,'MODEL_DAMAGED','来源模型完整性检查未通过。')
            source=model['manifest']['training_snapshot']
            if source.get('training_mode','standard')!=mode:
                raise ProjectError(409,'SOURCE_MODEL_ARCHITECTURE','请选择来源模型的通用或高精度模式。')
            architecture=source.get('initial_model',{}).get('architecture',source['target']['weights']['path'].rsplit('/',1)[-1])
            if not re.fullmatch(r'yolo(?:11|26)[nm](?:-seg)?\.pt',architecture):
                raise ProjectError(409,'SOURCE_MODEL_ARCHITECTURE','来源模型架构尚不支持迭代训练。')
            with self.storage.connect() as db:
                data=json.loads(self.versions.row(db,project_id,task_id,values['version_id'])['manifest_json'])
            if model['manifest']['classes']!=data['classes']:
                raise ProjectError(409,'SOURCE_MODEL_CLASSES','新数据版本的类别及顺序需与来源模型一致。')
            initial_model={'id':model['id'],'number':model['number'],'manifest_sha256':model['manifest_sha256'],
                           'training_run_id':model['training_run_id'],'weights_sha256':model['manifest']['artifacts']['best.pt']['sha256'],'artifact':'best.pt','architecture':architecture}
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=self.versions.row(db,project_id,task_id,values['version_id'])
            old=self.reused(db,task_id,values['idempotency_key'],digest)
            if old:return old
            if report['version_id']!=row['id'] or report['manifest_sha256']!=row['manifest_sha256']:
                raise ProjectError(409,'TRAINING_DATA_CHANGED','版本核对结果与选择不一致，请重试。')
            manifest=json.loads(row['manifest_json']);now=timestamp();job_id=uuid.uuid4().hex
            snapshot={'format_version':1,'task_id':task_id,'data_version_id':row['id'],'data_version_number':row['number'],
                      'manifest_sha256':row['manifest_sha256'],'image_count':len(manifest['images']),
                      'class_count':len(manifest['classes']),'distribution':manifest['distribution'],
                      'target':target, 'requested_parameters':parameters(cfg,values['epochs']),
                      'data_check':{'passed':True,'checked_at':now,'source_integrity_checked':True},
                      'runtime_check_policy':'执行前在目标实例重新核对环境、磁盘、数据与权重；此记录尚未执行。'}
            snapshot.update(training_mode=mode,execution_method=values.get('execution_method','ssh'))
            if initial_model:snapshot['initial_model']=initial_model
            if cfg:snapshot.update(configuration=cfg,configuration_contract=2,validation_only=allow_validation)
            if allow_validation:snapshot['validation_only']=True
            if segment:snapshot['task_type']='segment'
            payload=encoded(snapshot);sha=hashlib.sha256(payload).hexdigest()
            db.execute('''INSERT INTO training_jobs(id,task_id,version_id,idempotency_key,request_sha256,name,state,snapshot_json,snapshot_sha256,created_at,updated_at)
                VALUES(?,?,?,?,?,?,'QUEUED',?,?,?,?)''',(job_id,task_id,row['id'],values['idempotency_key'],digest,values['name'],payload.decode('utf-8'),sha,now,now))
            return {'job':self.public(db,self.row(db,project_id,task_id,job_id)),'reused':False}

    def cancel(self,project_id,task_id,job_id):
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE');row=self.row(db,project_id,task_id,job_id)
            if row['state'] in ('DISPATCHED','RUNNING'):raise ProjectError(409,'TRAINING_DISPATCHED','执行包已发出，请使用远程中止并等待进程结束；手动模式需在实例结束后回传结果。')
            if row['state']=='QUEUED':
                db.execute("UPDATE training_jobs SET state='CANCELLED',updated_at=? WHERE id=? AND state='QUEUED'",(timestamp(),job_id))
            return self.public(db,self.row(db,project_id,task_id,job_id))
