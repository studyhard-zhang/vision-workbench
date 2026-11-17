"""N-10 manual executor. Linux GPU execution is explicit; --prepare-only is CPU-only."""
import argparse
import contextlib
import errno
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import time
import subprocess
import tempfile
import zipfile
from pathlib import Path

_HELD_LOCKS=[]


def resolved_parameter(arguments,key):
    """Read precision after Ultralytics normalizes legacy half into quantize."""
    if key=='half' and not hasattr(arguments,'half'):
        precision=getattr(arguments,'quantize')
        if precision is None or type(precision) is int and precision==32:return False
        if type(precision) is int and precision==16:return True
        raise RuntimeError('无法核对运行库实际推理精度：'+str(precision))
    return getattr(arguments,key)


def failure_details(exc,phase='training',existing_code=None):
    """Classify observed failures without importing the training runtime or retrying."""
    chain=[];seen=set();current=exc
    while current is not None and id(current) not in seen and len(chain)<8:
        chain.append(current);seen.add(id(current));current=current.__cause__ or current.__context__
    messages=' '.join(str(item) for item in chain).lower()
    if any(isinstance(item,OSError) and item.errno in (errno.ENOSPC,getattr(errno,'EDQUOT',122)) for item in chain) or 'no space left on device' in messages:
        code='STORAGE_FULL'
    elif 'cuda out of memory' in messages or any(type(item).__name__=='OutOfMemoryError' and type(item).__module__.startswith('torch.cuda') for item in chain):code='CUDA_OUT_OF_MEMORY'
    elif any(isinstance(item,MemoryError) for item in chain):code='HOST_MEMORY_EXHAUSTED'
    elif any(isinstance(item,OSError) and item.errno in (errno.EACCES,errno.EPERM) for item in chain):code='STORAGE_PERMISSION_DENIED'
    elif existing_code in ('CHECKPOINT_DEPENDENCY_MISSING','COMPUTE_ENVIRONMENT_INVALID'):code=existing_code
    elif any(isinstance(item,ImportError) for item in chain):code='RUNTIME_DEPENDENCY_MISSING'
    elif '已有受管进程占用' in messages:code='DEVICE_BUSY'
    elif '损失非有限数' in messages:code='NONFINITE_TRAINING_LOSS'
    else:code='RESULT_FINALIZATION_FAILED' if phase=='finalization' else 'TRAINING_EXECUTION_FAILED'
    titles={'STORAGE_FULL':'实例存储空间或配额不足','CUDA_OUT_OF_MEMORY':'CUDA显存不足','HOST_MEMORY_EXHAUSTED':'实例系统内存不足','STORAGE_PERMISSION_DENIED':'实例文件写入权限不足',
        'CHECKPOINT_DEPENDENCY_MISSING':'权重额外依赖缺失','RUNTIME_DEPENDENCY_MISSING':'训练运行依赖缺失','COMPUTE_ENVIRONMENT_INVALID':'计算环境核对未通过','DEVICE_BUSY':'受管任务或设备仍被占用','NONFINITE_TRAINING_LOSS':'训练损失出现非有限数','RESULT_FINALIZATION_FAILED':'结果收尾未完成','TRAINING_EXECUTION_FAILED':'实例执行失败'}
    hints={'STORAGE_FULL':'先在实例释放空间或检查配额。若结果尚未回传，恢复原任务的收尾；不要删除活动输入、权重或锁文件。',
        'CUDA_OUT_OF_MEMORY':'显存不足：请先减小批大小，必要时降低输入尺寸或使用通用模型，再按修改后的配置新建训练。原任务和结果保留，不会自动改参数或重跑。',
        'HOST_MEMORY_EXHAUSTED':'检查实例系统内存和其他进程，处理后按配置新建；不会回退到控制端运行。',
        'STORAGE_PERMISSION_DENIED':'检查本任务实例目录和文件权限，保留已有材料；处理后先恢复连接核对原结果。',
        'CHECKPOINT_DEPENDENCY_MISSING':'补齐权重要求的依赖并先验证CPU加载；保留原失败记录，后续执行使用新记录。',
        'RUNTIME_DEPENDENCY_MISSING':'核对实例缺失的运行依赖，保留现有torch/CUDA版本；不自动安装或重试。',
        'COMPUTE_ENVIRONMENT_INVALID':'查看环境检查明细，核对原GPU身份和运行库后再按配置新建。',
        'DEVICE_BUSY':'保留原进程和设备锁，先核对已有任务；不能删除锁文件或重复启动。',
        'NONFINITE_TRAINING_LOSS':'检查日志、冻结标签和输入图片，修正输入后发布新版本再新建训练。',
        'RESULT_FINALIZATION_FAILED':'保留原任务目录，核对进程、空间和权限后恢复收尾，不重新训练。',
        'TRAINING_EXECUTION_FAILED':'下载本次日志定位原因；原记录不重跑，需要重试时按配置新建。'}
    return {'code':code,'title':titles[code],'next_step':hints[code],'phase':phase}


def safe_print(message):
    try:print(message,flush=True)
    except OSError:pass  # A full output volume must not turn a saved result into another execution.


def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()

def registry_path(request):
    path=Path.home()/'.vision-workbench-locks'/('dispatch-'+request['dispatch_id']+'.json')
    return path

def progress(directory,stage,**values):
    temporary=directory/'progress.json.tmp'
    temporary.write_bytes(encoded({'stage':stage,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values}));temporary.replace(directory/'progress.json')


def check_input(directory):
    request=json.loads((directory/'request.json').read_text(encoding='utf-8'))
    snapshot=request['snapshot'];params=snapshot['requested_parameters']
    if request['format_version']!=1 or hashlib.sha256(encoded(snapshot)).hexdigest()!=request['snapshot_sha256']:raise ValueError('配置快照校验不符。')
    try:from .training_config import validate_snapshot
    except ImportError:from training_config import validate_snapshot
    validate_snapshot(snapshot)
    if snapshot.get('initial_model'):
        initial=snapshot['initial_model']
        if initial.get('artifact')!='best.pt' or not re.fullmatch('[0-9a-f]{32}',initial.get('id','')) or sha(directory/'initial.pt')!=initial['weights_sha256']:
            raise ValueError('迭代训练来源权重校验不符。')
    if snapshot['target']['gpu_selector']!='0' or snapshot['target']['kind'] not in ('autodl_instance','self_hosted'):raise ValueError('计算目标不支持。')
    if set(request['scripts']) not in ({'run_training_job.py','verify_dataset.py','compute_probe.py'}, {'run_training_job.py','verify_dataset.py','compute_probe.py','training_config.py'},{'run_training_job.py','verify_dataset.py','compute_probe.py','training_config.py','polygon_geometry.py'}):raise ValueError('脚本清单无效。')
    for name,digest in request['scripts'].items():
        if sha(directory/name)!=digest:raise ValueError('脚本校验不符：'+name)
    if sha(directory/'dataset.zip')!=request['dataset_sha256']:raise ValueError('数据包SHA不符。')
    try:from .verify_dataset import audit
    except ImportError:from verify_dataset import audit
    report=audit(directory/'dataset.zip')
    if not report['passed'] or report['version_id']!=snapshot['data_version_id'] or report['manifest_sha256']!=snapshot['manifest_sha256']:raise ValueError('冻结数据核对未通过。')
    return request,report


@contextlib.contextmanager
def device_lock(directory):
    if platform.system()!='Linux':raise RuntimeError('当前远程执行器仅在Linux实例运行；控制端可用--prepare-only核对。')
    import fcntl
    locks=Path.home()/'.vision-workbench-locks';locks.mkdir(exist_ok=True)
    # Keep descriptors alive through result creation and interpreter teardown.
    handles=[]
    try:
        for path in (directory/'.job.lock',locks/'gpu-0.lock'):
            f=path.open('a+b');handles.append(f);fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except Exception:
        for f in handles:f.close()
        raise RuntimeError('已有受管进程占用此任务或GPU0；本次未开始，不生成另一份结果。')
    _HELD_LOCKS.extend(handles)
    yield  # File objects deliberately remain alive to process teardown, including error exits.


def verify_lock(directory):
    with tempfile.TemporaryDirectory(dir=directory) as other:
        child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--directory',other,'--probe-lock'],capture_output=True,text=True,timeout=15)
        if child.returncode!=3:raise RuntimeError('跨进程GPU锁检查未通过，不开始训练。')
    print('跨进程GPU锁检查通过。',flush=True)


class Tee(io.TextIOBase):
    def __init__(self,terminal,log):self.terminal=terminal;self.log=log
    def write(self,value):
        self.terminal.write(value);self.log.write(value);self.flush();return len(value)
    def flush(self):
        if not self.terminal.closed:self.terminal.flush()
        if not self.log.closed:self.log.flush()
    @property
    def encoding(self):return 'utf-8'
    def isatty(self):return False


def verify_preview_geometry(work,size):
    """On the authorized instance, compare the controller geometry with installed Ultralytics."""
    import random
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox,RandomFlip
    from ultralytics.utils.instance import Instances
    try:from .training_config import geometry
    except ImportError:from training_config import geometry
    manifest=json.loads((work/'manifest.json').read_text(encoding='utf-8'))
    members=[image for image in manifest['images'] if image['subset']=='train'][:3]
    state=random.getstate();checked=0;mask_checked=0
    try:
        for image in members:
            width,height=image['width'],image['height'];boxes=image['boxes']
            original=cv2.imread(str(work/image['image_path']))
            if original is None or original.shape[:2]!=(height,width):raise RuntimeError('几何核对图片尺寸不符。')
            layout=geometry(width,height,size,boxes)
            resized=cv2.resize(original,(layout['resized_width'],layout['resized_height']),interpolation=cv2.INTER_LINEAR)
            normalized=np.array([[b['x']/width,b['y']/height,(b['x']+b['width'])/width,(b['y']+b['height'])/height] for b in boxes],dtype=np.float32).reshape(-1,4)
            for horizontal,vertical in ((False,False),(True,False),(False,True),(True,True)):
                labels={'img':resized.copy(),'instances':Instances(normalized.copy(),segments=np.zeros((0,1000,2),dtype=np.float32),bbox_format='xyxy',normalized=True)}
                labels=LetterBox(new_shape=(size,size),auto=False)(labels)
                labels=RandomFlip(p=int(vertical),direction='vertical')(labels)
                labels=RandomFlip(p=int(horizontal),direction='horizontal')(labels)
                labels['instances'].convert_bbox(format='xyxy')
                expected=geometry(width,height,size,boxes,horizontal,vertical)
                coordinates=np.array([[b['x'],b['y'],b['x']+b['width'],b['y']+b['height']] for b in expected['boxes']],dtype=np.float32).reshape(-1,4)
                if labels['img'].shape[:2]!=(size,size) or not np.allclose(labels['instances'].bboxes,coordinates,rtol=0,atol=.001):
                    raise RuntimeError('实例LetterBox/RandomFlip与预览框坐标不一致，不开始训练。')
                for shape,transformed in zip(boxes,expected['boxes']):
                    if 'points' not in shape:continue
                    segments=np.array([shape['points']],dtype=np.float32)/np.array([width,height],dtype=np.float32)
                    sample={'img':resized.copy(),'instances':Instances(np.array([[shape['x']/width,shape['y']/height,(shape['x']+shape['width'])/width,(shape['y']+shape['height'])/height]],dtype=np.float32),segments=segments,bbox_format='xyxy',normalized=True)}
                    sample=LetterBox(new_shape=(size,size),auto=False)(sample);sample=RandomFlip(p=int(vertical),direction='vertical')(sample);sample=RandomFlip(p=int(horizontal),direction='horizontal')(sample)
                    if not np.allclose(sample['instances'].segments[0],np.array(transformed['points']),rtol=0,atol=.001):raise RuntimeError('实例分割顶点与预处理坐标核对不符。')
                    mask_checked+=1
                checked+=1
    finally:random.setstate(state)
    if checked==0:raise RuntimeError('没有可核对的训练图片。')
    return {'geometry_check_passed':True,'geometry_check_samples':checked,'geometry_tolerance_pixels':.001,'mask_geometry_passed':mask_checked>0,'mask_geometry_samples':mask_checked}


def train(directory,request,report):
    snapshot=request['snapshot'];target=snapshot['target'];params=snapshot['requested_parameters']
    task=snapshot.get('task_type','detect')
    if task not in ('detect','segment'):raise RuntimeError('训练任务类型无效。')
    os.environ['CUDA_VISIBLE_DEVICES']=target['gpu_selector']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ['YOLO_AUTOINSTALL']='false'
    try:from .compute_probe import inspect_environment,checkpoint_dependencies
    except ImportError:from compute_probe import inspect_environment,checkpoint_dependencies
    print('[1/4] 核对已有权重与运行依赖…',flush=True)
    progress(directory,'checking_dependencies',epochs_completed=0,epochs_requested=params['epochs'])
    weights=Path(target['weights']['path'])
    mode=snapshot.get('training_mode','standard')
    if mode not in ('standard','high_accuracy'):raise RuntimeError('训练模式无效。')
    architecture=snapshot.get('initial_model',{}).get('architecture',weights.name)
    family='yolo26' if architecture.startswith('yolo26') else 'yolo11'
    filename=family+('m' if mode=='high_accuracy' else 'n')+('-seg' if task=='segment' else '')+'.pt'
    if snapshot.get('initial_model'):
        weights=directory/'initial.pt'
        if sha(weights)!=snapshot['initial_model']['weights_sha256']:raise RuntimeError('迭代训练来源权重SHA不符。')
        report['initial_model']=snapshot['initial_model']
    elif not weights.is_absolute() or weights.name!=filename or not weights.is_file() or sha(weights)!=target['weights']['sha256']:raise RuntimeError('预训练权重缺失或SHA不符，不自动下载。')
    report['weights_sha256']=sha(weights)
    dependencies=checkpoint_dependencies(weights);report['checkpoint_dependency_check']=dependencies
    if dependencies['missing']:
        report.update(error_code='CHECKPOINT_DEPENDENCY_MISSING',completion_reason='checkpoint_dependency_missing')
        raise RuntimeError('权重运行依赖缺失：'+', '.join(dependencies['missing'])+'；未进行CUDA检查或训练，不自动安装或重试。')
    if not dependencies['scan_supported']:raise RuntimeError('此权重pickle格式暂不能完整静态核对，需显式兼容验证；未进行CUDA检查或训练。')
    print('[1/4] 核对实例环境、GPU和已有权重…',flush=True)
    progress(directory,'checking_environment',epochs_completed=0,epochs_requested=params['epochs'])
    environment=inspect_environment(directory,target['expected_gpu']);report['environment']=environment
    if not environment['ready']:
        report['error_code']='COMPUTE_ENVIRONMENT_INVALID'
        raise RuntimeError('环境检查未通过；不会安装依赖或回退CPU。')
    if request.get('expected_gpu_uuid') and request['expected_gpu_uuid'] not in environment.get('nvidia_smi',''):raise RuntimeError('实例GPU身份变化，请先重新核对计算目标。')
    for key,expected in target['runtime'].items():
        actual=environment['python'] if key=='python' else environment['packages'].get(key)
        if actual!=expected:raise RuntimeError(f'运行环境{key}变化：{actual}；需要重新核对。')
    import torch
    from ultralytics import YOLO
    trainer_type=None
    if params['batch']==-1:
        import logging
        from ultralytics.utils import LOGGER
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        from ultralytics.models.yolo.segment.train import SegmentationTrainer
        class CheckedAutoBatch(SegmentationTrainer if task=='segment' else DetectionTrainer):
            def auto_batch(self,*args,**kwargs):
                torch.backends.cudnn.benchmark=False
                warnings=[]
                class Capture(logging.Handler):
                    def emit(self,record):warnings.append(record.getMessage())
                handler=Capture();LOGGER.addHandler(handler)
                try:resolved=super().auto_batch(*args,**kwargs)
                finally:LOGGER.removeHandler(handler)
                if any('using default batch-size' in s.lower() for s in warnings):raise RuntimeError('AutoBatch显存估算未成功，已停止；请手动选择批大小或较低尺寸，不会自动降低输入分辨率。')
                if type(resolved) is not int or resolved<1:raise RuntimeError('AutoBatch没有返回有效批大小。')
                report.update(resolved_batch_size=resolved,autobatch_memory_fraction=.6,autobatch_estimation_passed=True)
                return resolved
        trainer_type=CheckedAutoBatch
    work=directory/'work';work.mkdir()
    with zipfile.ZipFile(directory/'dataset.zip') as archive:archive.extractall(work)  # audit validated every member
    data=json.loads((work/'data.yaml').read_text(encoding='utf-8'))
    if snapshot.get('configuration'):report.update(verify_preview_geometry(work,params['imgsz']))
    model=YOLO(str(weights))
    if snapshot.get('initial_model') and [model.names[i] for i in range(len(model.names))]!=data['names']:
        raise RuntimeError('来源模型类别与新数据版本不一致。')
    if model.task!=task or not Path(model.model.yaml.get('yaml_file','')).stem.startswith(family):raise RuntimeError('权重与训练任务类型不符。')
    print('[2/4] 执行CUDA前向检查…',flush=True)
    progress(directory,'checking_forward',epochs_completed=0,epochs_requested=params['epochs'])
    network=model.model.to('cuda:0').float().eval()
    with torch.inference_mode():network(torch.zeros((1,3,params['imgsz'],params['imgsz']),device='cuda:0'));torch.cuda.synchronize()
    report['forward_passed']=True;torch.cuda.empty_cache();batches=[];epochs=[]
    def start(trainer):
        if trainer.device.type!='cuda' or trainer.device.index not in (None,0):raise RuntimeError('训练器未使用指定CUDA设备。')
        report['resolved_batch_size']=int(trainer.batch_size)
        progress(directory,'training',epochs_completed=0,epochs_requested=params['epochs'])
    def batch_end(trainer):
        if not torch.isfinite(trainer.loss).all().item():raise RuntimeError('训练损失非有限数。')
        batches.append(1)
    def epoch_end(trainer):
        epochs.append(trainer.epoch);report.update(epochs_completed=len(epochs),batches_completed=len(batches))
        progress(directory,'training',epochs_completed=len(epochs),epochs_requested=params['epochs'],batches_completed=len(batches))
        print(f'[进度] 已完成 {len(epochs)}/{params["epochs"]} 轮，累计 {len(batches)} 批次',flush=True)
    model.add_callback('on_train_start',start);model.add_callback('on_train_batch_end',batch_end);model.add_callback('on_train_epoch_end',epoch_end)
    print('[3/4] 开始训练，终端和console.log同步输出…',flush=True)
    report['training_executed']=True;torch.cuda.reset_peak_memory_stats(0)
    train_args={'trainer':trainer_type} if trainer_type else {}
    model.train(**params,**train_args,data=str((work/'data.yaml').resolve()),project=str(work.resolve()),name='train',exist_ok=False,resume=False)
    if not epochs or not batches:raise RuntimeError('没有完成训练轮次/批次。')
    print('[4/4] 检查best/last权重及重新加载…',flush=True)
    report['reload_passed']={}
    for name in ('best.pt','last.pt'):
        checkpoint=work/'train/weights'/name
        if not checkpoint.is_file() or not checkpoint.stat().st_size:raise RuntimeError('缺少权重：'+name)
        loaded=YOLO(str(checkpoint))
        if loaded.task!=task or [loaded.names[i] for i in range(len(loaded.names))]!=data['names']:raise RuntimeError('权重重载或类别映射检查未通过。')
        report['reload_passed'][name]=True
    report.update(status='completed',completion_reason='completed_epochs' if len(epochs)==params['epochs'] else 'early_stopped',
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0))
    report['actual_parameters']={k:v if isinstance(v,(str,int,float,bool,type(None))) else str(v) for k,v in vars(model.trainer.args).items()}
    report['actual_optimizer']=type(model.trainer.optimizer).__name__
    report['metrics']={str(k):float(v) if math.isfinite(float(v)) else None for k,v in model.trainer.metrics.items()}


def pack_result(directory,request,report):
    files={}
    for name,path in [('console.log',directory/'console.log'),('args.yaml',directory/'work/train/args.yaml'),
                      ('results.csv',directory/'work/train/results.csv'),('best.pt',directory/'work/train/weights/best.pt'),('last.pt',directory/'work/train/weights/last.pt')]:
        if path.is_file():
            if report.get('status')=='interrupted' and name in ('best.pt','last.pt'):
                report.setdefault('retained_artifacts',{})[name]={'bytes':path.stat().st_size,'sha256':sha(path),'location':'work/train/weights/'+name}
                continue
            data=path.read_bytes()
            if name=='console.log' and len(data)>8*1024**2:data=b'[log tail; earlier output omitted]\n'+data[-8*1024**2:]
            files[name]=data
    if sum(len(v) for v in files.values())>120*1024**2:raise RuntimeError('产物超过当前回传上限，请保留目录联系检查。')
    receipt={k:request[k] for k in ('job_id','dispatch_id','snapshot_sha256','dataset_sha256','return_token')}
    receipt.update(format_version=1,report=report,artifacts={k:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for k,v in files.items()})
    payload=encoded(receipt);stage=directory/'return-result.zip.tmp'
    with zipfile.ZipFile(stage,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        archive.writestr('receipt.json',payload)
        for name,data in files.items():archive.writestr(name,data)
    os.replace(stage,directory/'return-result.zip')


def interrupted_progress(directory,requested_epochs):
    """Keep the last durable completed counters; a killed partial epoch is not guessed."""
    try:
        saved=json.loads((directory/'progress.json').read_text(encoding='utf-8'))
        epochs=saved['epochs_completed'];batches=saved['batches_completed']
        if type(epochs) is not int or not 0<=epochs<=requested_epochs or type(batches) is not int or batches<0:raise ValueError('invalid counters')
        return {'epochs_completed':epochs,'batches_completed':batches,'progress_source':'last_persisted_progress','progress_is_partial':True}
    except (OSError,ValueError,KeyError,TypeError):
        return {'epochs_completed':0,'batches_completed':0,'progress_source':'unknown','progress_is_partial':True}


def main():
    parser=argparse.ArgumentParser(description='在AutoDL实例执行工作台生成的训练包，不安装依赖或下载权重。')
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--recover-interrupted',action='store_true')
    parser.add_argument('--probe-lock',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();directory=args.directory.resolve()
    if args.probe_lock:
        try:
            with device_lock(directory):return 0
        except RuntimeError:return 3
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    print('视训工作台：检查执行包…',flush=True)
    report=None;owned_execution=False
    try:
        request,audit=check_input(directory)
        if args.prepare_only:
            print(json.dumps({'prepared':True,'training_executed':False,'job_id':request['job_id'],'data_check':audit},ensure_ascii=False,indent=2),flush=True);return 0
        # A process lifetime lock is intentionally kept until process exit. Never unlink lock files.
        with device_lock(directory):
            owned_execution=True
            if (directory/'return-result.zip').exists():
                print('此执行包已有结果，不重复训练。请回传：'+str(directory/'return-result.zip'),flush=True);return 0
            started=directory/'started.json'
            registry=registry_path(request);prior=json.loads(registry.read_text(encoding='utf-8')) if registry.exists() else None
            if prior and not args.recover_interrupted:raise RuntimeError('此派发已开始过，不在其他目录或缺失结果时重跑。原目录：'+str(prior['directory'])+'；请回传原结果，或确认结束后生成中断结果。')
            if args.recover_interrupted and prior:
                original=prior.get('result_path')
                if not original:
                    if not isinstance(prior.get('directory'),str):raise RuntimeError('原派发目录记录不完整，请保留材料并人工核对；不生成替代结果。')
                    original=str(Path(prior['directory'])/'return-result.zip')
                if Path(original).is_file():raise RuntimeError('原结果仍存在，请直接回传：'+original)
            verify_lock(directory)
            if started.exists() and not args.recover_interrupted:raise RuntimeError('上次执行未正常收尾，不会重跑。确认进程已结束后用--recover-interrupted生成中断结果。')
            report={'status':'failed','training_executed':False,'forward_passed':False,'requested_parameters':request['snapshot']['requested_parameters'],
                    'epochs_completed':0,'batches_completed':0,'lock_check_passed':True,'started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            clock=time.perf_counter()
            if args.recover_interrupted:
                report.update(interrupted_progress(directory,request['snapshot']['requested_parameters']['epochs']))
                report.update(status='interrupted',training_executed=None if started.exists() or prior else False,error='设备与任务锁均空闲，已关闭本次派发；保留最后已知的完整轮次/批次，未重新训练。')
            else:
                with started.open('x',encoding='utf-8') as f:json.dump({'job_id':request['job_id'],'pid':os.getpid()},f);f.flush();os.fsync(f.fileno())
                with registry.open('x',encoding='utf-8') as f:json.dump({'directory':str(directory),'job_id':request['job_id'],'pid':os.getpid()},f);f.flush();os.fsync(f.fileno())
                with (directory/'console.log').open('w',encoding='utf-8') as log:
                    with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                        try:train(directory,request,report)
                        except KeyboardInterrupt:report.update(status='interrupted',error='用户中断了实例执行。');print(report['error'],flush=True)
                        except Exception as exc:
                            import traceback
                            report['diagnostic']=failure_details(exc,'training' if report.get('training_executed') else 'preflight',report.get('error_code'))
                            report['error_code']=report['diagnostic']['code']
                            report['error']=(f'{type(exc).__name__}: {exc}')[:4000];traceback.print_exc()
            report.update(elapsed_seconds=round(time.perf_counter()-clock,4),finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            pack_result(directory,request,report)
            try:
                progress(directory,report['status'],epochs_completed=report['epochs_completed'],batches_completed=report['batches_completed'],epochs_requested=request['snapshot']['requested_parameters']['epochs'])
                registry.write_bytes(encoded({'directory':str(directory),'job_id':request['job_id'],'status':report['status'],'result_path':str(directory/'return-result.zip')}))
            except OSError as exc:safe_print('结果包已保存，进度或派发登记更新未完成，请保留原目录：'+str(exc))
            safe_print('执行结束：'+report['status']+'。请下载并回传：'+str(directory/'return-result.zip'))
            # Hold flock descriptors to interpreter exit, after GPU objects and imported modules terminate.
            return 0 if report['status']=='completed' else 2
    except Exception as exc:
        diagnostic=failure_details(exc,'finalization' if report else 'input',report.get('error_code') if report else None)
        if owned_execution and report is not None:
            try:progress(directory,'needs_attention',epochs_completed=report['epochs_completed'],batches_completed=report['batches_completed'],epochs_requested=request['snapshot']['requested_parameters']['epochs'],diagnostic=diagnostic)
            except OSError:pass
        safe_print(f'操作未完成，请保留目录：{type(exc).__name__}: {exc}');safe_print('故障分类：'+json.dumps(diagnostic,ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
