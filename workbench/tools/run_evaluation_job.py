"""Independent YOLO11 detection evaluation on the existing Linux target, never training."""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

try:from . import run_training_job as support
except ImportError:import training_support as support

PLOTS=('BoxPR_curve.png','BoxF1_curve.png','BoxP_curve.png','BoxR_curve.png','MaskPR_curve.png','MaskF1_curve.png','MaskP_curve.png','MaskR_curve.png','confusion_matrix.png','confusion_matrix_normalized.png')
MAX_ARTIFACT_BYTES=60*1024**2
MAX_RETURN_BYTES=64*1024**2
MAX_SOURCE_REPORT_BYTES=256*1024**2


def mask_contours(mask,original_shape,input_shape=None,ratio_pad=None):
    """All mask contours, including holes, in EXIF-canonical original image pixels."""
    import cv2
    import numpy as np
    from ultralytics.utils import ops
    binary=mask.detach().cpu().numpy().astype(np.uint8)
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return []
    shape=input_shape or binary.shape;result=[]
    for i,contour in enumerate(contours):
        if len(contour)<3:continue
        # Bound returned material; explicit failure avoids silently truncating a mask.
        if len(contour)>4096:raise RuntimeError('分割轮廓顶点超过当前单轮廓4096点上限。')
        points=contour.reshape(-1,2).astype(np.float32)
        points[:,0]*=shape[1]/binary.shape[1];points[:,1]*=shape[0]/binary.shape[0]
        points=ops.scale_coords(shape,points,original_shape,ratio_pad=ratio_pad)
        depth=0;parent=int(hierarchy[0][i][3])
        while parent>=0:depth+=1;parent=int(hierarchy[0][parent][3])
        result.append((depth,{'points':points.tolist(),'hole':bool(depth%2)}))
    result.sort(key=lambda row:row[0])
    if len(result)>256 or sum(len(r['points']) for _,r in result)>32768:raise RuntimeError('分割轮廓复杂度超过当前报告上限。')
    return [r for _,r in result]


def check_input(directory):
    request=json.loads((directory/'request.json').read_text(encoding='utf-8'));snap=request['snapshot']
    if request.get('kind')!='evaluation' or snap.get('kind')!='evaluation' or request['format_version']!=1:raise ValueError('不是独立评估执行包。')
    if support.sha(directory/'model.pt')!=snap['weights_sha256'] or snap['weights_sha256']!=request['weights_sha256']:raise ValueError('指定正式模型的权重SHA不符。')
    if hashlib.sha256(support.encoded(snap)).hexdigest()!=request['snapshot_sha256']:raise ValueError('评估快照SHA不符。')
    if set(request['scripts'])!={'run_training_job.py','training_support.py','compute_probe.py','verify_dataset.py'}:raise ValueError('评估脚本清单无效。')
    for name,sha in request['scripts'].items():
        if support.sha(directory/name)!=sha:raise ValueError('脚本SHA不符：'+name)
    if support.sha(directory/'dataset.zip')!=request['dataset_sha256']:raise ValueError('数据包SHA不符。')
    try:from .verify_dataset import audit
    except ImportError:from verify_dataset import audit
    result=audit(directory/'dataset.zip')
    if not result['passed'] or result['manifest_sha256']!=snap['manifest_sha256'] or result['version_id']!=snap['data_version_id']:raise ValueError('冻结版本核对未通过。')
    if snap['subset'] not in ('val','test') or snap['target']['gpu_selector']!='0' or snap['target']['kind'] not in ('autodl_instance','self_hosted'):raise ValueError('评估划分或目标无效。')
    p=snap['requested_parameters']
    if set(p)!={'imgsz','batch','workers','half','device','conf','iou','max_det','augment'} or any(p[k]!=v for k,v in {'batch':1,'workers':0,'half':False,'device':0,'max_det':300,'augment':False}.items()):raise ValueError('评估配置超出当前支持范围。')
    if type(p['imgsz']) is not int or not 320<=p['imgsz']<=2048 or p['imgsz']%32:raise ValueError('输入尺寸无效。')
    for k in ('conf','iou'):
        if type(p[k]) not in (float,int) or not 0.001<=p[k]<=0.95:raise ValueError('评估阈值无效。')
    return request


def evaluate(directory,request,report):
    snap=request['snapshot'];params=snap['requested_parameters'];target=snap['target']
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['YOLO_AUTOINSTALL']='false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    try:from .compute_probe import inspect_environment,checkpoint_dependencies
    except ImportError:from compute_probe import inspect_environment,checkpoint_dependencies
    support.progress(directory,'checking_dependencies',images_completed=0,images_requested=snap['sample_count'])
    dep=checkpoint_dependencies(directory/'model.pt')
    if dep['missing'] or not dep['scan_supported']:raise RuntimeError('权重额外依赖缺失或无法核对；不自动安装。'+str(dep['missing']))
    env=inspect_environment(directory,target['expected_gpu']);report['environment']=env
    if not env['ready'] or request['expected_gpu_uuid'] not in env.get('nvidia_smi',''):raise RuntimeError('GPU环境或身份不符。')
    for k,expected in target['runtime'].items():
        actual=env['python'] if k=='python' else env['packages'].get(k)
        if actual!=expected:raise RuntimeError('运行库版本改变：'+k)
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    if ultralytics.__version__ not in ('8.3.241','8.4.203'):raise RuntimeError('当前仅核对8.3.241评估口径。')
    work=directory/'work';work.mkdir()
    with zipfile.ZipFile(directory/'dataset.zip') as archive:archive.extractall(work)
    data=json.loads((work/'manifest.json').read_text(encoding='utf-8'))
    images={i['id']:i for i in data['images'] if i['subset']==snap['subset']}
    if not images or len(images)!=snap['sample_count'] or data['classes']!=snap['classes']:raise RuntimeError('评估样本或类别映射不符。')
    config=json.loads((work/'data.yaml').read_text(encoding='utf-8'));config['path']=str(work.resolve())
    (work/'data.yaml').write_bytes(support.encoded(config))
    model=YOLO(str(directory/'model.pt'));task=snap.get('task_type','detect')
    if task not in ('detect','segment') or model.task!=task or (data['label_format']=='yolo-segment-polygon-normalized-v1')!=(task=='segment') or [model.names[i] for i in range(len(model.names))]!=[c['name'] for c in snap['classes']]:raise RuntimeError('模型任务类型、类别与冻结版本不一致。')
    if task=='segment':
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        Validator=SegmentationValidator
    else:Validator=DetectionValidator
    captured=[];actual={}
    class CaptureValidator(Validator):
        def init_metrics(self,network):
            if self.device.type!='cuda' or self.device.index not in (None,0):raise RuntimeError('评估未使用指定CUDA设备。')
            for k,v in params.items():
                resolved=support.resolved_parameter(self.args,k)
                if (str(resolved)!=str(v) if k=='device' else resolved!=v):raise RuntimeError('运行库实际参数不同：'+k)
                actual[k]=v
            super().init_metrics(network)
        def update_metrics(self,preds,batch):
            for si,pred in enumerate(preds):
                prepared=self._prepare_batch(si,batch)
                image=images[Path(prepared['im_file']).stem]
                # Scaling every full-resolution mask would multiply GPU memory by max_det.
                native=DetectionValidator.scale_preds(self,pred,prepared)
                boxes=native['bboxes'].detach().cpu().tolist()
                scores=native['conf'].detach().cpu().tolist();classes=native['cls'].detach().cpu().tolist()
                predictions=[{'class_index':int(c),'confidence':float(s),'xyxy':[max(0,min(float(v),image['width'] if idx%2==0 else image['height'])) for idx,v in enumerate(box)]} for c,s,box in zip(classes,scores,boxes)]
                if task=='segment':
                    if len(pred['masks'])!=len(predictions):raise RuntimeError('掩膜与检测结果数量不一致。')
                    for row,mask in zip(predictions,pred['masks']):row['segments']=mask_contours(mask,(image['height'],image['width']),prepared['imgsz'],prepared['ratio_pad'])
                captured.append({'id':image['id'],'width':image['width'],'height':image['height'],'predictions':predictions})
            super().update_metrics(preds,batch)
            support.progress(directory,'evaluating',images_completed=len(captured),images_requested=len(images))
    report.update(evaluation_executed=True,weights_sha256=support.sha(directory/'model.pt'))
    started=time.perf_counter()
    metrics=model.val(validator=CaptureValidator,data=str(work/'data.yaml'),split=snap['subset'],**params,
                      plots=True,save_json=False,save_txt=False,project=str(work),name='evaluation',exist_ok=False,verbose=False)
    if len(captured)!=len(images):raise RuntimeError('实际评估图片数量不完整。')
    keys=('precision','recall','map50','map50_95')
    counts=[sum(b['class_id']==c['id'] for image in images.values() for b in image['boxes']) for c in snap['classes']]
    primary=metrics.seg if task=='segment' else metrics.box
    indices={int(c):i for i,c in enumerate(primary.ap_class_index)}
    per_class=[]
    for idx,c in enumerate(snap['classes']):
        values=primary.class_result(indices[idx]) if idx in indices and counts[idx] else [None]*4
        per_class.append({'index':idx,'name':c['name'],'instances':counts[idx],'metrics':{k:None if v is None else float(v) for k,v in zip(keys,values)}})
    matrix=getattr(metrics,'confusion_matrix',None)
    report_data={'format_version':1,'snapshot':snap,'runtime':ultralytics.__version__,'sample_count':len(images),
        'metrics':dict(zip(keys,[float(v) for v in primary.mean_results()] if any(counts) else [None]*4)),
        'per_class':per_class,'images':captured,'confusion_matrix':matrix.matrix.tolist() if matrix is not None else None,
        'metric_semantics':{'precision_recall':'运行库在置信度曲线的平均F1最佳点汇总P/R；不是界面置信度处的单点P/R。',
            'ap':'运行库PR积分；mAP50-95对IoU 0.50至0.95、步长0.05取平均。提高最低置信度会截断PR候选。',
            'confusion_confidence':0.25 if params['conf']==0.001 else params['conf'],'confusion_match_iou':0.45,
            'matrix_axes':'行=预测类别，列=真值类别；最后一行/列为background。',
            'subset':'来源冻结验证集，曾参与训练选模，不等同独立测试集。' if snap['subset']=='val' else '来源冻结测试集；训练未使用该划分。'},
        'timing':{'device':target['name'],'batch':1,'imgsz':params['imgsz'],'runtime_speed_ms_per_image':{k:float(v) for k,v in metrics.speed.items()},
                  'validation_wall_seconds':time.perf_counter()-started,'scope':'运行库逐图速度含其预处理/推理/后处理；模型加载、SSH传输和网页时间不计入，运行库执行预热。'},
        'missing':[]}
    if task=='segment':
        report_data['metric_scope']='mask';report_data['box_metrics']=dict(zip(keys,[float(v) for v in metrics.box.mean_results()] if any(counts) else [None]*4))
        report_data['metric_semantics']['matrix_scope']='混淆矩阵基于检测框匹配；上方主指标及Mask曲线基于分割掩膜。'
    for name in PLOTS:
        if name.startswith('Mask') and task!='segment':continue
        if not (work/'evaluation'/name).is_file():report_data['missing'].append(name+'：运行库未生成，可能无有效真值或预测。')
    (directory/'report.json').write_bytes(support.encoded(report_data))
    report.update(status='completed',images_completed=len(captured),actual_parameters=actual)


def pack_result(directory,request,report):
    paths={'report.json':directory/'report.json','console.log':directory/'console.log',**{n:directory/'work/evaluation'/n for n in PLOTS}}
    if paths['report.json'].is_file() and paths['report.json'].stat().st_size>MAX_SOURCE_REPORT_BYTES:
        raise RuntimeError('原始结果报告超出处理上限，保留目录。')
    files={n:p.read_bytes() for n,p in paths.items() if p.is_file()}
    # Whitespace can dominate native mask coordinates. Preserve every value and hole;
    # only the transport copy changes. The original on-disk report remains evidence.
    if 'report.json' in files:
        files['report.json']=(json.dumps(json.loads(files['report.json']),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
    if len(files.get('console.log',b''))>4*1024**2:files['console.log']=files['console.log'][-4*1024**2:]
    if sum(map(len,files.values()))>MAX_ARTIFACT_BYTES:raise RuntimeError('结果材料超出回传上限，保留目录；不会截断预测或轮廓。')
    receipt={k:request[k] for k in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256','return_token')}
    receipt.update(format_version=1,kind=request.get('kind','evaluation'),report=report,artifacts={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()})
    temp=directory/'return-result.zip.tmp'
    payload=support.encoded(receipt)
    if len(payload)>1024**2 or len(payload)+sum(map(len,files.values()))>MAX_RETURN_BYTES:raise RuntimeError('结果清单或解压材料超出回传上限，保留目录。')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('receipt.json',payload)
        for n,b in files.items():archive.writestr(n,b)
    if temp.stat().st_size>MAX_RETURN_BYTES:raise RuntimeError('结果包超出回传上限，保留目录。')
    temp.replace(directory/'return-result.zip')


def main(*,check=None,execute=None,packer=None,kind="evaluation"):
    check=check or check_input;execute=execute or evaluate;packer=packer or pack_result
    label='推理' if kind=='inference' else '评估'
    parser=argparse.ArgumentParser(description=f'独立模型{label}，不训练、不安装运行库。')
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--recover-interrupted',action='store_true');parser.add_argument('--probe-lock',action='store_true')
    args=parser.parse_args();directory=args.directory.resolve();report=None;owned=False
    if args.probe_lock:
        try:
            with support.device_lock(directory):return 0
        except RuntimeError:return 3
    try:
        request=check(directory)
        if args.prepare_only:print(json.dumps({'prepared':True,'kind':kind,'inference_executed':False,'evaluation_executed':False,'training_executed':False}));return 0
        with support.device_lock(directory):
            owned=True
            if (directory/'return-result.zip').is_file():return 0
            registry=support.registry_path(request);prior=json.loads(registry.read_text(encoding='utf-8')) if registry.is_file() else None
            if prior:
                original=Path(prior['directory'])/'return-result.zip'
                if original.is_file():raise RuntimeError('原结果仍在，请回传原结果，不覆盖。'+str(original))
                if not args.recover_interrupted:raise RuntimeError('此派发已运行，不自动重跑。')
            if (directory/'started.json').exists() and not args.recover_interrupted:raise RuntimeError('原进程未收尾；不重新评估。')
            support.verify_lock(directory)
            report={'status':'failed','training_executed':False,'evaluation_executed':False,'inference_executed':False,'lock_check_passed':True,'images_completed':0}
            if args.recover_interrupted:report.update(status='interrupted',error=f'原进程已结束且设备锁空闲，关闭原{label}；没有重新{label}或训练。')
            else:
                for path in (directory/'started.json',registry):
                    with path.open('x',encoding='utf-8') as f:json.dump({'directory':str(directory),'pid':os.getpid(),'job_id':request['job_id']},f);f.flush();os.fsync(f.fileno())
                with (directory/'console.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(support.Tee(sys.stdout,log)),contextlib.redirect_stderr(support.Tee(sys.stderr,log)):
                    try:execute(directory,request,report)
                    except KeyboardInterrupt:report.update(status='interrupted',error=f'{label}被中止。')
                    except Exception as exc:
                        report.update(error=(f'{type(exc).__name__}: {exc}')[:4000],diagnostic=support.failure_details(exc,kind));traceback.print_exc()
            packer(directory,request,report)
            try:
                support.progress(directory,report['status'],images_completed=report['images_completed'],images_requested=request['snapshot']['sample_count'])
                registry.write_bytes(support.encoded({'directory':str(directory),'result_path':str(directory/'return-result.zip')}))
            except OSError as exc:support.safe_print('结果包已保存，元数据未更新：'+str(exc))
            return 0 if report['status']=='completed' else 2
    except Exception as exc:
        if owned and report is not None:
            try:support.progress(directory,'needs_attention',diagnostic=support.failure_details(exc,'finalization'))
            except OSError:pass
        support.safe_print(f'{label}未完成，请保留目录：'+str(exc));return 2


if __name__=='__main__':raise SystemExit(main())
