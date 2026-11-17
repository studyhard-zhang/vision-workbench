"""Read-only, bounded views of returned training metrics. No compute runtime imports."""
import csv
import hashlib
import io
import math

from .storage import ProjectError


SERIES = [
    ('train/box_loss','训练 box 损失','box'),('val/box_loss','验证 box 损失','box'),
    ('train/cls_loss','训练 cls 损失','cls'),('val/cls_loss','验证 cls 损失','cls'),
    ('train/dfl_loss','训练 dfl 损失','dfl'),('val/dfl_loss','验证 dfl 损失','dfl'),
    ('metrics/precision(B)','Precision','metrics'),('metrics/recall(B)','Recall','metrics'),
    ('metrics/mAP50(B)','mAP@0.5','metrics'),('metrics/mAP50-95(B)','mAP@0.5:0.95','metrics'),
    ('train/seg_loss','训练掩膜损失','seg'),('val/seg_loss','验证掩膜损失','seg'),
    ('metrics/precision(M)','掩膜 Precision','mask_metrics'),('metrics/recall(M)','掩膜 Recall','mask_metrics'),
    ('metrics/mAP50(M)','掩膜 mAP@0.5','mask_metrics'),('metrics/mAP50-95(M)','掩膜 mAP@0.5:0.95','mask_metrics'),
    ('lr/pg0','参数组 0 学习率','lr'),('lr/pg1','参数组 1 学习率','lr'),('lr/pg2','参数组 2 学习率','lr'),
]
MAX_CSV_BYTES = 2 * 1024**2


def parse_metrics(raw, epochs_requested, epochs_completed=None, task_type='detect'):
    """Keep unavailable observations as None, and reject ambiguous or non-finite data."""
    try:
        if not 0<len(raw)<=MAX_CSV_BYTES:raise ValueError('指标文件为空或超过2MiB。')
        table=list(csv.reader(io.StringIO(raw.decode('utf-8-sig')),strict=True))
        if not table:raise ValueError('缺少指标表头。')
        header=[cell.strip() for cell in table[0]]
        if len(header)>64 or any(not cell for cell in header) or len(set(header))!=len(header) or 'epoch' not in header:
            raise ValueError('指标表头缺失、重复或过多。')
        segment=task_type=='segment' or any('(M)' in key or 'seg_loss' in key for key in header)
        active=[{'key':key,'label':label,'group':group,'present':key in header} for key,label,group in SERIES if segment or group not in ('seg','mask_metrics')]
        rows=[];last_epoch=0
        for cells in table[1:]:
            if not cells or not any(cell.strip() for cell in cells):continue
            if len(cells)!=len(header):raise ValueError('指标行列数与表头不一致。')
            source=dict(zip(header,(cell.strip() for cell in cells)))
            epoch=float(source['epoch'])
            if not math.isfinite(epoch) or not epoch.is_integer() or not last_epoch<epoch<=epochs_requested:
                raise ValueError('轮次必须递增、不重复，且不能超过请求轮数。')
            last_epoch=int(epoch);row={'epoch':last_epoch}
            for key in ['time',*(item['key'] for item in active)]:
                value=source.get(key,'')
                number=None if value=='' else float(value)
                if number is not None and (not math.isfinite(number) or number<0 or (key.startswith('metrics/') and number>1)):
                    raise ValueError('指标必须是范围内的有限数，缺失值需留空。')
                row[key]=number
            rows.append(row)
            if len(rows)>300:raise ValueError('指标最多300轮。')
        if not rows:raise ValueError('尚无已记录轮次。')
        if epochs_completed is not None and rows[-1]['epoch']>epochs_completed:
            raise ValueError('指标轮次超过已回传的完成轮数。')
        warnings=[]
        if [r['epoch'] for r in rows]!=list(range(1,rows[-1]['epoch']+1)):
            warnings.append('原始指标中部分轮次未记录；曲线不补点。')
        if epochs_completed is not None and rows[-1]['epoch']<epochs_completed:
            warnings.append('原始指标少于回传完成轮数，仅展示已记录部分。')
        if any(not item['present'] or any(row[item['key']] is None for row in rows) for item in active):
            warnings.append('部分指标缺失，显示为“未记录”；不会补零或插值。')
        return {'rows':rows,'series':active,'warnings':warnings}
    except (ValueError,UnicodeError,csv.Error,OverflowError) as exc:
        raise ProjectError(422,'TRAINING_METRICS_INVALID','无法显示训练曲线：'+str(exc)) from exc


class TrainingReportRepository:
    def __init__(self, executions):
        self.executions=executions;self.training=executions.training

    def get(self, project_id, task_id, job_id):
        job=self.training.get(project_id,task_id,job_id)
        receipt=job['dispatch']['result'] if job['dispatch'] else None
        report=receipt['report'] if receipt else {}
        actual=report.get('actual_parameters',{})
        requested=job['snapshot']['requested_parameters']
        parameters=[]
        for key in sorted(set(requested)|set(actual)):
            note=''
            if key=='optimizer':note='请求/解析配置与最终优化器分开记录。'
            if key=='batch' and requested.get('batch')==-1:note=f"AutoBatch目标约60%显存；本次实际批大小：{report.get('resolved_batch_size','未回传')}。"
            if key in ('lr0','momentum') and requested.get('optimizer')=='auto':
                note='auto 优化器会自动选择此项；右侧只是解析配置，不能视作优化器最终值。逐轮学习率见 lr/pg*。'
            parameters.append({'key':key,'requested_present':key in requested,'requested':requested.get(key),
                'actual_present':key in actual,'actual':actual.get(key),'note':note})
        value={'task_type':job['snapshot'].get('task_type','detect'),'job_id':job_id,'job_state':job['state'],'snapshot_sha256':job['snapshot_sha256'],
            'requested_epochs':requested['epochs'],'completed_epochs':report.get('epochs_completed'),
            'parameters':parameters,'actual_optimizer':report.get('actual_optimizer'),
            'metrics_scope':'TRAINING_INTERNAL_VALIDATION',
            'scope_description':'指标来自本次训练的 results.csv，验证集用于训练监控，不是独立评估或测试集结果；不自动判定业务达标。',
            'rows':[],'series':[],'warnings':[],'csv_sha256':None,'status':'NOT_AVAILABLE'}
        if not receipt or 'results.csv' not in receipt['artifacts']:
            value['reason']='尚无已回传的逐轮指标。训练进行中可查看阶段与日志，结果回传后显示曲线。' if job['state'] in ('QUEUED','DISPATCHED','RUNNING') else '该训练没有回传逐轮指标，不能生成曲线。'
            return value
        path,_,_=self.executions.artifact(project_id,task_id,job_id,'results.csv')
        try:
            with path.open('rb') as stream:raw=stream.read(MAX_CSV_BYTES+1)
        except OSError as exc:raise ProjectError(409,'ARTIFACT_DAMAGED','训练指标读取失败，请检查存储后重新读取。') from exc
        if len(raw)>MAX_CSV_BYTES:raise ProjectError(422,'TRAINING_METRICS_INVALID','指标文件超过2MiB，无法显示曲线。')
        if hashlib.sha256(raw).hexdigest()!=receipt['artifacts']['results.csv']['sha256']:
            raise ProjectError(409,'ARTIFACT_DAMAGED','训练指标在读取时发生变化，请检查存储。')
        parsed=parse_metrics(raw,requested['epochs'],report.get('epochs_completed'),job['snapshot'].get('task_type','detect'))
        value.update(parsed,status='AVAILABLE',csv_sha256=receipt['artifacts']['results.csv']['sha256'])
        from .training_diagnosis import diagnose
        value['diagnosis']=diagnose(job,value['rows'])
        if job['state']!='SUCCEEDED':value['warnings'].append('该训练未成功完成；这里只展示已回传的部分指标，不代表正式可用模型。')
        return value
