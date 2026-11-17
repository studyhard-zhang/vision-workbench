"""Immutable model registry. Reuses verified remote reload evidence; never loads pickle/torch."""
import hashlib
import json
import shutil
import tempfile
import uuid
import sqlite3
from pathlib import Path

from .images import comparable_path
from .storage import ProjectError, timestamp
from .versions import encoded, sha_file


SELECTION_RULE = {
    'runtime': 'Ultralytics 8.3.241 / detect', 'selected_artifact': 'best.pt',
    'description': 'best.pt 由训练器按训练内置验证 fitness 保存；检测 fitness 为 mAP@0.5:0.95（权重 1.0），相同最佳值也会保存。last.pt 是训练结束权重，不代表最佳。',
    'sources': [
        'https://github.com/ultralytics/ultralytics/blob/v8.3.241/ultralytics/engine/trainer.py',
        'https://github.com/ultralytics/ultralytics/blob/v8.3.241/ultralytics/utils/metrics.py'],
    'scope': '训练器选择规则；不是独立评估或业务达标结论。',
}


class ModelRepository:
    def automatic(self,project_id,task_id,job_id):
        try:return {'model':self.register(project_id,task_id,job_id)['model']}
        except (ProjectError,OSError,sqlite3.Error) as exc:
            # The validated training receipt remains committed if publication needs storage repair.
            return {'model_error':{'code':getattr(exc,'code','MODEL_IO_FAILED'),'message':str(exc)}}

    def sync_successful(self,project_id=None,task_id=None):
        with self.storage.connect() as db:
            if task_id:self.versions.tasks.get(project_id,task_id,db)
            scope=' AND j.task_id=?' if task_id else ''
            pending=[dict(r) for r in db.execute("SELECT j.id,j.task_id,t.project_id FROM training_jobs j JOIN vision_tasks t ON t.id=j.task_id LEFT JOIN model_versions m ON m.training_run_id=j.id LEFT JOIN training_record_visibility v ON v.job_id=j.id WHERE j.state='SUCCEEDED' AND m.id IS NULL AND v.deleted_at IS NULL"+scope+" ORDER BY j.sequence",(task_id,) if task_id else ())]
        return [self.automatic(r['project_id'],r['task_id'],r['id']) for r in pending]

    def __init__(self, storage, executions):
        self.storage=storage;self.executions=executions;self.training=executions.training;self.versions=executions.versions
        self.root=(storage.path.parent/'models').resolve();self.root.mkdir(exist_ok=True)
        self.staging=self.root/'.staging';self.staging.mkdir(exist_ok=True)

    def owned(self, *parts):
        path=self.root.joinpath(*parts).resolve();key,root=comparable_path(path),comparable_path(self.root)
        if key==root or not key.is_relative_to(root):raise ProjectError(400,'INVALID_PATH','模型文件路径无效。')
        return path

    def row(self, db, project_id, task_id, model_id):
        self.versions.tasks.get(project_id,task_id,db)
        row=db.execute('SELECT * FROM model_versions WHERE task_id=? AND id=?',(task_id,model_id)).fetchone()
        if row is None:raise ProjectError(404,'MODEL_NOT_FOUND','模型版本不存在或不属于当前任务。')
        return dict(row)

    def public(self, row):
        manifest=json.loads(row['manifest_json']);issues=[]
        def check(path, spec):
            try:valid=path.is_file() and path.stat().st_size==spec['bytes'] and sha_file(path)==spec['sha256']
            except OSError:valid=False
            if not valid:issues.append(path.name)
        raw=row['manifest_json'].encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=row['manifest_sha256']:issues.append('数据库清单')
        check(self.owned(row['id'],'manifest.json'),{'bytes':len(raw),'sha256':row['manifest_sha256']})
        for name,spec in manifest['artifacts'].items():check(self.owned(row['id'],name),spec)
        with self.storage.connect() as db:
            latest=db.execute('SELECT id,state,snapshot_json FROM evaluation_jobs WHERE model_id=? ORDER BY sequence DESC LIMIT 1',(row['id'],)).fetchone()
            success=db.execute("SELECT id FROM evaluation_jobs WHERE model_id=? AND state='SUCCEEDED' ORDER BY sequence DESC LIMIT 1",(row['id'],)).fetchone()
        return {k:row[k] for k in ('id','task_id','number','training_run_id','task_data_version_id','created_at','manifest_sha256')} | {
            'manifest':manifest,'integrity':{'ok':not issues,'issues':issues},
            'available':not issues,'evaluation_status':'EVALUATED' if success else 'NOT_EVALUATED',
            'latest_evaluation':None if latest is None else {'id':latest['id'],'state':latest['state'],'subset':json.loads(latest['snapshot_json'])['subset']},
            'latest_successful_evaluation_id':success['id'] if success else None}

    def get(self, project_id, task_id, model_id):
        with self.storage.connect() as db:row=self.row(db,project_id,task_id,model_id)
        return self.public(row)

    def list(self, project_id, task_id):
        with self.storage.connect() as db:
            self.versions.tasks.get(project_id,task_id,db)
            rows=[dict(r) for r in db.execute('SELECT * FROM model_versions WHERE task_id=? ORDER BY number DESC',(task_id,))]
        return [self.public(r) for r in rows]

    def register(self, project_id, task_id, job_id):
        destination=None;published=False;committed=False
        try:
            # Serialize allocation and publication across server threads and repository instances.
            with self.storage.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                job=self.training.public(db,self.training.row(db,project_id,task_id,job_id))
                old=db.execute('SELECT * FROM model_versions WHERE training_run_id=?',(job_id,)).fetchone()
                if old:return {'model':self.public(dict(old)),'reused':True}
                if job['deleted_at']:raise ProjectError(409,'TRAINING_IN_TRASH','请先从回收站恢复此训练记录，再登记新模型。')
                if job['state']!='SUCCEEDED' or not job['dispatch'] or job['dispatch']['state']!='SUCCEEDED':
                    raise ProjectError(409,'MODEL_TRAINING_INCOMPLETE','仅可登记已成功完成并校验的训练；失败、中断或恢复权重不能成为正式模型。')
                dispatch=db.execute('SELECT * FROM training_dispatches WHERE job_id=?',(job_id,)).fetchone()
                request=json.loads(dispatch['request_json']);snapshot=job['snapshot']
                if hashlib.sha256(encoded(snapshot)).hexdigest()!=job['snapshot_sha256'] or request['snapshot']!=snapshot:
                    raise ProjectError(409,'MODEL_SOURCE_DAMAGED','训练快照校验不符，未登记模型。')
                result_path=self.executions.owned(job_id,'returned','result.zip')
                if not result_path.is_file() or sha_file(result_path)!=dispatch['result_sha256']:
                    raise ProjectError(409,'MODEL_SOURCE_DAMAGED','训练结果包缺失或校验不符，未登记模型。')
                receipt,files=self.executions.validate_result(result_path.read_bytes(),request)
                if receipt!=json.loads(dispatch['result_json']):raise ProjectError(409,'MODEL_SOURCE_DAMAGED','结果报告与已接收记录不一致。')
                if self.executions.owned(job_id,'returned','receipt.json').read_bytes()!=encoded(receipt):
                    raise ProjectError(409,'MODEL_SOURCE_DAMAGED','回传清单校验不符。')
                for name in files:self.executions.artifact(project_id,task_id,job_id,name)
                version=self.versions.row(db,project_id,task_id,job['version_id']);data=json.loads(version['manifest_json'])
                if hashlib.sha256(encoded(data)).hexdigest()!=version['manifest_sha256'] or version['manifest_sha256']!=snapshot['manifest_sha256'] or not self.versions.integrity(version,data)['ok']:
                    raise ProjectError(409,'MODEL_SOURCE_DAMAGED','来源数据版本完整性检查不通过。')
                self.versions.space(sum(len(v) for v in files.values())+1024**2)
                model_id=uuid.uuid4().hex;now=timestamp()
                number=db.execute('SELECT coalesce(max(number),0)+1 FROM model_versions WHERE task_id=?',(task_id,)).fetchone()[0]
                environment=receipt['report']['environment']
                runtime_version=environment.get('ultralytics_import',{}).get('version') or environment.get('packages',{}).get('ultralytics')
                selection_rule=SELECTION_RULE if runtime_version=='8.3.241' else {
                    'runtime':runtime_version or '版本未记录','selected_artifact':'best.pt',
                    'description':'默认使用该运行库输出的 best.pt；该实际运行库版本的 fitness 公式尚未核实。last.pt 为结束权重。',
                    'scope':'不推断其他版本的最佳权重算法；不是独立评估结论。','sources':[]}
                if snapshot.get('task_type')=='segment' and runtime_version=='8.3.241':
                    selection_rule={**SELECTION_RULE,'runtime':'Ultralytics 8.3.241 / segment','description':'分割 best.pt 按训练器的检测框 mAP@0.5:0.95 与掩膜 mAP@0.5:0.95 之和选择；last.pt 是结束权重。详情顶部展示所选轮次的掩膜指标，不等同 best.pt 指标。'}
                manifest={'format_version':1,'model_version_id':model_id,'model_version_number':number,'task_id':task_id,
                    'training_run_id':job_id,'training_name':job['name'],'task_data_version_id':job['version_id'],
                    'data_version_number':snapshot['data_version_number'],'created_at':now,'classes':data['classes'],
                    'training_snapshot':snapshot,'snapshot_sha256':job['snapshot_sha256'],
                    'result_sha256':dispatch['result_sha256'],'receipt':receipt,
                    'selection_rule':selection_rule,'default_artifact':'best.pt',
                    'load_check':{'method':'remote_reload_then_sha256','reload_passed':receipt['report']['reload_passed'],
                                  'description':'复用该训练回传的实例重载证据；控制端只核对字节与SHA-256，不重新加载模型。'},
                    'evaluation':{'status':'NOT_EVALUATED','reason':'尚未执行独立评估；训练内置验证不作为独立评估。'},
                    'artifacts':receipt['artifacts']}
                raw=encoded(manifest);digest=hashlib.sha256(raw).hexdigest();destination=self.owned(model_id)
                with tempfile.TemporaryDirectory(dir=self.staging) as temporary:
                    stage=Path(temporary)
                    for name,content in files.items():(stage/name).write_bytes(content)
                    (stage/'manifest.json').write_bytes(raw)
                    for name,spec in receipt['artifacts'].items():
                        if sha_file(stage/name)!=spec['sha256']:raise ProjectError(409,'MODEL_COPY_DAMAGED','模型副本校验失败，未登记。')
                    stage.rename(destination);published=True
                    db.execute('INSERT INTO model_versions VALUES(?,?,?,?,?,?,?,?)',(model_id,task_id,number,job_id,job['version_id'],now,raw.decode('utf-8'),digest))
                    event={'format_version':1,'model_version_id':model_id,'task_id':task_id,'training_run_id':job_id,
                           'task_data_version_id':job['version_id'],'artifact':'best.pt','sha256':receipt['artifacts']['best.pt']['sha256'],
                           'evaluation_policy':'DEFERRED_UNTIL_EVALUATION_NODE'}
                    db.execute("INSERT INTO model_events VALUES(?,?,'MODEL_REGISTERED',?,?)",(uuid.uuid4().hex,model_id,encoded(event).decode(),now))
            committed=True
            return {'model':self.get(project_id,task_id,model_id),'reused':False}
        except OSError as exc:
            raise ProjectError(507 if exc.errno==28 else 503,'MODEL_IO_FAILED','模型登记未完成，请检查空间和文件权限；原训练产物保留。') from exc
        finally:
            if published and not committed and destination and destination.exists():shutil.rmtree(destination)

    def artifact(self, project_id, task_id, model_id, name):
        model=self.get(project_id,task_id,model_id)
        if name not in {*model['manifest']['artifacts'],'manifest.json'}:raise ProjectError(404,'MODEL_ARTIFACT_NOT_FOUND','模型产物不存在。')
        if not model['available']:raise ProjectError(409,'MODEL_DAMAGED','模型文件缺失或校验不符，已阻止下载；请保留目录并检查存储。')
        return self.owned(model_id,name),f'M{model["number"]:03d}-{model_id[:8]}-{name}','application/json' if name=='manifest.json' else 'application/octet-stream'
