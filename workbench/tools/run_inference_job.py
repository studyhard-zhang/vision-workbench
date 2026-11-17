"""One bounded inference batch, using the existing remote runtime and lifecycle locks."""
import hashlib
import json
import os
import re
import stat
import time
import zipfile
from pathlib import Path

try:
    from . import run_evaluation_job as lifecycle
    from . import run_training_job as support
except ImportError:
    import inference_support as lifecycle
    import training_support as support


def check_input(directory):
    request=json.loads((directory/'request.json').read_text(encoding='utf-8'));snap=request['snapshot']
    if request.get('kind')!='inference' or snap.get('kind')!='inference' or request['format_version']!=1:raise ValueError('不是图片推理执行包。')
    if hashlib.sha256(support.encoded(snap)).hexdigest()!=request['snapshot_sha256']:raise ValueError('推理快照SHA不符。')
    scripts={'run_training_job.py','training_support.py','inference_support.py','compute_probe.py','verify_dataset.py'}
    if snap.get('backend')=='tensorrt':scripts|={'tensorrt_support.py','deployment_build.py','deployment_predict_trt.py'}
    if snap.get('backend','pytorch') not in ('pytorch','tensorrt'):raise ValueError('推理后端无效。')
    if set(request['scripts'])!=scripts:raise ValueError('推理脚本清单无效。')
    for n,digest in request['scripts'].items():
        if support.sha(directory/n)!=digest:raise ValueError('推理脚本SHA不符。')
    if support.sha(directory/'model.pt')!=request['weights_sha256'] or request['weights_sha256']!=snap['weights_sha256']:raise ValueError('指定模型权重SHA不符。')
    if support.sha(directory/'dataset.zip')!=request['dataset_sha256']:raise ValueError('推理输入包SHA不符。')
    if snap['coordinate_system']!='exif-oriented-pixels' or snap['target']['gpu_selector']!='0' or snap['target']['kind'] not in ('autodl_instance','self_hosted'):raise ValueError('输入坐标口径或计算目标无效。')
    p=snap['requested_parameters'];fixed={'batch':1,'workers':0,'half':snap.get('backend')=='tensorrt','device':0,'augment':False,'rect':False,'max_det':300}
    if set(p)!=set(fixed)|{'imgsz','conf','iou'} or any(p[k]!=v for k,v in fixed.items()) or any(type(p[k]) not in (int,float) or not 0.001<=p[k]<=0.95 for k in ('conf','iou')):raise ValueError('推理参数超出当前支持范围。')
    if type(p['imgsz']) is not int or not 320<=p['imgsz']<=2048 or p['imgsz']%32:raise ValueError('输入尺寸无效。')
    images=snap['inputs']
    if not 1<=len(images)<=50 or len({i['id'] for i in images})!=len(images) or snap['sample_count']!=len(images):raise ValueError('推理输入数无效。')
    ready=[i for i in images if i['status']=='READY']
    if not ready or len(ready)!=snap['valid_count']:raise ValueError('没有有效推理输入。')
    with zipfile.ZipFile(directory/'dataset.zip') as z:
        names=z.namelist();allowed={'manifest.json',*(i['file_path'] for i in ready)}
        if len(names)!=len(set(names)) or set(names)!=allowed or sum(i.file_size for i in z.infolist())>129*1024**2 or any(i.is_dir() or i.flag_bits&1 or stat.S_ISLNK(i.external_attr>>16) for i in z.infolist()):raise ValueError('推理输入包成员无效。')
        manifest_raw=z.read('manifest.json')
        if hashlib.sha256(manifest_raw).hexdigest()!=snap['manifest_sha256']:raise ValueError('输入清单SHA不符。')
        manifest=json.loads(manifest_raw)
        if manifest['kind']!='inference_inputs' or manifest['images']!=images:raise ValueError('输入清单内容不符。')
        for image in ready:
            if not re.fullmatch(r'inputs/[0-9a-f]{32}\.(png|jpg)',image['file_path']):raise ValueError('输入图片路径无效。')
            raw=z.read(image['file_path'])
            if len(raw)!=image['display_bytes'] or hashlib.sha256(raw).hexdigest()!=image['display_sha256']:raise ValueError('输入图片SHA不符。')
    return request


def infer(directory,request,report):
    snap=request['snapshot'];target=snap['target'];params=snap['requested_parameters']
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['YOLO_AUTOINSTALL']='false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    try:from .compute_probe import inspect_environment,checkpoint_dependencies
    except ImportError:from compute_probe import inspect_environment,checkpoint_dependencies
    support.progress(directory,'checking_dependencies',images_completed=0,images_requested=snap['sample_count'])
    dependencies=checkpoint_dependencies(directory/'model.pt')
    if not dependencies['scan_supported'] or dependencies['missing']:raise RuntimeError('权重依赖缺失或无法静态核对；不会安装依赖。')
    env=inspect_environment(directory,target['expected_gpu']);report['environment']=env
    if not env['ready'] or request['expected_gpu_uuid'] not in env.get('nvidia_smi',''):raise RuntimeError('GPU身份或环境检查未通过。')
    for k,expected in target['runtime'].items():
        actual=env['python'] if k=='python' else env['packages'].get(k)
        if actual!=expected:raise RuntimeError('运行库版本变化：'+k)
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__ not in ('8.3.241','8.4.203'):raise RuntimeError('尚未核对此运行库版本的推理接口。')
    work=directory/'work';work.mkdir()
    with zipfile.ZipFile(directory/'dataset.zip') as z:z.extractall(work)
    model=YOLO(str(directory/'model.pt'));task=snap.get('task_type','detect')
    if task not in ('detect','segment') or model.task!=task or [model.names[i] for i in range(len(model.names))]!=[c['name'] for c in snap['classes']]:raise RuntimeError('模型类型或类别与模型清单不一致。')
    cache=None;acceleration=None
    if snap.get('backend')=='tensorrt':
        try:from . import tensorrt_support as accelerator
        except ImportError:import tensorrt_support as accelerator
        from PIL import Image
        first=next(i for i in snap['inputs'] if i['status']=='READY')
        with Image.open(work/first['file_path']) as source:sample=source.convert('RGB')
        model,cache,acceleration=accelerator.prepare(directory,request,model,sample,support)
        sample.close()
    rows=[{'id':i['id'],'filename':i['filename'],'status':'FAILED' if i['status']=='INVALID' else 'NOT_PROCESSED',
           'width':i.get('width'),'height':i.get('height'),'source_width':i.get('source_width'),'source_height':i.get('source_height'),
           'error_code':i.get('error_code','NOT_PROCESSED'),'error':i.get('error','任务尚未处理该图片。'),'predictions':[]} for i in snap['inputs']]
    data={'format_version':1,'snapshot':snap,'runtime':ultralytics.__version__,'coordinate_system':'exif-oriented-pixels','complete':False,'images':rows,
          'timing':{'device':target['name'],'imgsz':snap['requested_parameters']['imgsz'],'batch':1,'warmup':'运行库在新预测器首次调用时预热。',
                    'scope':'逐图speed为运行库预处理/推理/后处理毫秒；不含模型加载、SSH或网页。批次墙钟包含首次预测器预热。'}}
    if acceleration:
        data['acceleration']=acceleration
        data['timing']['warmup']='计时前已完成5次预热。'
        data['timing']['scope']='逐图processing_ms含文件读取、预处理、模型、后处理及轮廓提取；不含模型加载、引擎编译、SSH、曝光或PLC。benchmark单独使用内存图片。'
    report.update(inference_executed=True,weights_sha256=support.sha(directory/'model.pt'))
    start=time.perf_counter();actual={}
    for image,row in zip(snap['inputs'],rows):
        try:
            if image['status']=='INVALID':continue
            if acceleration:
                import torch
                torch.cuda.synchronize()
            image_started=time.perf_counter()
            result=model.predict(source=str(work/image['file_path']),**params,save=False,stream=False,verbose=False)[0]
            if tuple(result.orig_shape)!=(image['height'],image['width']):raise RuntimeError('推理坐标对应的图片尺寸不符。')
            predictor=model.predictor
            if predictor.device.type!='cuda' or predictor.device.index not in (None,0):raise RuntimeError('推理未使用指定CUDA设备。')
            for k,v in params.items():
                resolved=support.resolved_parameter(predictor.args,k)
                if acceleration and k=='imgsz' and isinstance(resolved,(list,tuple)) and list(resolved)==[v,v]:resolved=v
                if (str(resolved)!=str(v) if k=='device' else resolved!=v):raise RuntimeError('实际推理参数不同：'+k)
                actual[k]=v
            boxes=result.boxes
            if boxes is None:raise RuntimeError('检测模型未返回检测框结构。')
            coords=boxes.xyxy.detach().cpu().tolist();classes=boxes.cls.detach().cpu().tolist();scores=boxes.conf.detach().cpu().tolist()
            row.update(status='SUCCEEDED',error_code=None,error=None,
                predictions=[{'class_index':int(c),'class_name':snap['classes'][int(c)]['name'],'confidence':float(score),
                              'xyxy':[max(0,min(float(v),image['width'] if idx%2==0 else image['height'])) for idx,v in enumerate(box)]} for c,score,box in zip(classes,scores,coords)],
                speed_ms={k:float(v) for k,v in result.speed.items()})
            if task=='segment':
                masks=result.masks
                if len(coords) and (masks is None or len(masks.data)!=len(coords)):raise RuntimeError('模型未返回对应的分割掩膜。')
                for prediction,mask in zip(row['predictions'],[] if masks is None else masks.data):prediction['segments']=lifecycle.mask_contours(mask,(image['height'],image['width']))
            if acceleration:torch.cuda.synchronize()
            row['processing_ms']=(time.perf_counter()-image_started)*1000
        except Exception as exc:
            row.update(status='FAILED',error_code='INFERENCE_IMAGE_FAILED',error=(f'{type(exc).__name__}: {exc}')[:1000],predictions=[])
            if any(word in str(exc).lower() for word in ('cuda','设备','参数不同','尺寸不符')):raise
        finally:
            report['images_completed']=sum(i['status']!='NOT_PROCESSED' for i in rows)
            (directory/'report.json').write_bytes(support.encoded(data))
            support.progress(directory,'inferring',images_completed=report['images_completed'],images_requested=len(rows))
    data['complete']=True;data['timing']['batch_wall_seconds']=time.perf_counter()-start
    (directory/'report.json').write_bytes(support.encoded(data))
    if not any(i['status']=='SUCCEEDED' for i in rows):raise RuntimeError('本批次有效图片均未成功推理。')
    if acceleration:accelerator.deployment(directory,request,cache,acceleration,support)
    report.update(status='completed',actual_parameters=actual)


def pack_result(directory,request,report):
    if request['snapshot'].get('backend')!='tensorrt':return lifecycle.pack_result(directory,request,report)
    paths={'report.json':directory/'report.json','console.log':directory/'console.log'}
    files={name:path.read_bytes() for name,path in paths.items() if path.is_file()}
    if len(files.get('report.json',b''))>256*1024**2:raise RuntimeError('原始报告过大，保留原材料。')
    if 'report.json' in files:files['report.json']=(json.dumps(json.loads(files['report.json']),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
    if 'console.log' in files:files['console.log']=files['console.log'][-4*1024**2:]
    if report['status']=='completed':
        package=directory/'deployment.zip'
        if package.stat().st_size+sum(map(len,files.values()))>120*1024**2:
            report.update(status='failed',error='推理报告与产线包合计超过120MiB，请减少本批图片数后新建；已生成的引擎缓存和原始结果保留。')
        else:files['deployment.zip']=package.read_bytes()
    if sum(map(len,files.values()))>120*1024**2:raise RuntimeError('报告超过回传上限，保留原材料，不截断掩膜。')
    receipt={k:request[k] for k in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256','return_token')}
    receipt.update(format_version=1,kind='inference',report=report,artifacts={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()})
    payload=support.encoded(receipt)
    if len(payload)>1024**2:raise RuntimeError('结果清单过大。')
    temp=directory/'return-result.zip.tmp'
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('receipt.json',payload)
        for name,body in files.items():archive.writestr(name,body)
    if temp.stat().st_size>128*1024**2:raise RuntimeError('加速结果包超过128MiB，保留原材料。')
    temp.replace(directory/'return-result.zip')


if __name__=='__main__':raise SystemExit(lifecycle.main(check=check_input,execute=infer,packer=pack_result,kind='inference'))
