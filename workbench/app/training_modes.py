"""Model capacity is independent of input-size/augmentation configuration."""
import json,re
from .storage import ProjectError

MODES={'standard':'通用模式','high_accuracy':'高精度模式'}

def model_name(task_type):
    return 'SegDL' if task_type=='segment' else 'DetectDL'

def high_accuracy_target(training,task_type,require_verified=True):
    target=training.target()
    from .compute_instances import active
    record = active(training.storage.path.parent)
    if record and record['target']['id']==target['id']:
        filename='yolo26m-seg.pt' if task_type=='segment' else 'yolo26m.pt'
        if filename not in record['weights']:raise ProjectError(409,'YOLO26_NOT_VERIFIED','请连接并验证YOLO26训练实例。')
        return {**target,'weights':record['weights'][filename]}
    try:
        proof=json.loads((training.storage.path.parent/'high-accuracy-training.json').read_text('utf-8'))[task_type]
        weight=proof['weights'];filename='yolo11m-seg.pt' if task_type=='segment' else 'yolo11m.pt'
        if proof['gpu_uuid']!=target['verified_gpu_uuid'] or proof['runtime']!=target['runtime'] or not re.fullmatch('[0-9a-f]{64}',weight['sha256']) or not weight['path'].startswith('/') or weight['path'].split('/')[-1]!=filename or (require_verified and proof.get('verified') is not True):raise ValueError()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise ProjectError(409,'HIGH_ACCURACY_NOT_VERIFIED','高精度模式尚未在当前GPU完成验证，通用模式仍可使用。') from exc
    return {**target,'weights':{**weight,'name':model_name(task_type)}}
