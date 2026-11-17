"""Facts from frozen snapshots and recorded metrics; no causal claims from one run."""
def diagnose(job,rows):
    snap=job['snapshot'];params=snap['requested_parameters'];report=(job.get('dispatch') or {}).get('result') or {};report=report.get('report',{})
    suffix='M' if snap.get('task_type')=='segment' else 'B';last=rows[-1];metric=lambda key:last.get(f'metrics/{key}({suffix})')
    train=snap['distribution']['train']['image_count'];val=snap['distribution']['val']['image_count'];notes=[]
    actual=report.get('resolved_batch_size',report.get('actual_parameters',{}).get('batch'))
    if params['batch']==-1:notes.append(f'请求AutoBatch -1，本次实际批大小为{actual if actual is not None else "未回传"}；不能仅由-1判断批大小增大或精度下降。')
    notes.append(f'输入{params["imgsz"]}，训练{train}张、验证{val}张。尺寸、学习率与数据质量都会影响收敛，50轮等相同轮数不保证相同效果。')
    if val<20:notes.append(f'验证集只有{val}张，Precision/Recall/mAP容易波动；建议补充有代表性的验证集和负样本。')
    if metric('precision') is not None and metric('precision')<.1 and (metric('recall') or 0)>.8:
        notes.append('末轮Precision偏低而Recall较高，提示误检/分类置信度需要检查；请在独立评估与图片推理中查看候选结果、负样本和阈值，不能当作总体分类正确率。')
    peak=max((r.get(f'metrics/mAP50({suffix})') or 0 for r in rows),default=0)
    if peak>(metric('mAP50') or 0)+.1:notes.append(f'本次mAP@0.5最高记录为{peak:.2%}，末轮为{metric("mAP50"):.2%}；存在明显轮间波动。最高单项指标轮次不等于best.pt选模轮次。')
    checkpoint={k:report.get('metrics',{}).get(f'metrics/{key}({suffix})') for k,key in (('precision','precision'),('recall','recall'),('map50','mAP50'),('map50_95','mAP50-95'))}
    if all(type(v) in (int,float) for v in checkpoint.values()):notes.append(f'best.pt回传验证：Precision {checkpoint["precision"]:.2%}、Recall {checkpoint["recall"]:.2%}、mAP@0.5 {checkpoint["map50"]:.2%}。这是训练结束时重载best.pt的内置验证，不是独立测试。')
    return {'notes':notes,'last_epoch':last['epoch'],'checkpoint_metrics':checkpoint,'metric_scope':'TRAINING_INTERNAL_VALIDATION','causal_conclusion':'不能由一次训练证明某参数导致精度下降。'}
