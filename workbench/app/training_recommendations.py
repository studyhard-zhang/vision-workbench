"""Explainable starting parameters, never a claim of optimal hyperparameters."""
from tools.training_config import parameters, OFFICIAL_CONFIG
from .small_targets import analyze


def recommend(training,pid,tid,vid,training_mode='standard'):
    version=training.versions.read(pid,tid,vid)
    from .storage import ProjectError
    if not version['integrity']['ok']:raise ProjectError(409,'VERSION_DAMAGED','冻结版本校验不通过，无法推荐参数。')
    data=version['manifest'];task=training.versions.tasks.get(pid,tid);kind=task['task_type']
    # New training starts from the pinned release defaults. Manual profiles remain explicit choices.
    params=parameters(OFFICIAL_CONFIG,100)
    return {'version_id':vid,'mode':'official','configuration':dict(OFFICIAL_CONFIG),'epochs':100,
            'parameters':params,'source':None,'reasons':['使用 Ultralytics 8.4.203 官方默认训练参数。'],
            'warnings':[],'notice':'默认输入640、batch16、100轮，开启官方默认增强和AMP。optimizer=auto由训练器决定优化器及学习率。可切换手动方案。',
            'small_targets':analyze(data,640,kind,False)}

