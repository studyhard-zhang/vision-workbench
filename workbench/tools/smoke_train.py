"""N-08 manual compute-target adapter, single-GPU/one-epoch YOLO11 smoke run.

Only execute on the chosen compute host. No automatic installation or weight download.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from .compute_probe import inspect_environment
    from .verify_dataset import audit
except ImportError:
    from compute_probe import inspect_environment
    from verify_dataset import audit


def file_hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def target_config(path):
    target=json.loads(Path(path).read_text(encoding='utf-8'))
    if target.get('format_version')!=1 or not isinstance(target.get('target_id'),str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}',target['target_id']):
        raise ValueError('执行目标ID或版本无效。')
    if target.get('kind') not in ('autodl_instance','self_hosted') or target.get('transport')!='manual_terminal':
        raise ValueError('当前仅支持目标机器内手动执行；工作台远程队列尚未接入。')
    selector=target.get('gpu_selector')
    if not isinstance(selector,str) or not re.fullmatch(r'(\d+|GPU-[A-Za-z0-9-]+)',selector):
        raise ValueError('一次只能选择一个GPU编号或UUID，不能回退CPU。')
    if not isinstance(target.get('expected_gpu'),str) or not target['expected_gpu'].strip():raise ValueError('请指定预期GPU型号以核对执行目标。')
    return target


def execute(args):
    target=target_config(args.target)
    os.environ['CUDA_VISIBLE_DEVICES']=target['gpu_selector']  # Before importing torch.
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free < args.bundle.stat().st_size*3+2*1024**3:
        raise ValueError('工作盘余量不足，请保留数据副本及至少2GiB烟测余量。')
    run=output/('smoke-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);run.mkdir()
    result={'format_version':1,'target_id':target['target_id'],'run_id':run.name,'status':'preparing','training_executed':False,'created_at':datetime.now(timezone.utc).isoformat()}
    started=time.perf_counter()
    try:
        shutil.copyfile(args.bundle,run/'input.zip')
        data_report=audit(run/'input.zip');write_json(run/'data-check.json',data_report)
        if not data_report['passed']:raise ValueError('冻结数据包核对失败，见data-check.json。')
        result.update(data_version_id=data_report['version_id'],manifest_sha256=data_report['manifest_sha256'],bundle_sha256=file_hash(run/'input.zip'))
        dataset=run/'dataset';dataset.mkdir()
        with zipfile.ZipFile(run/'input.zip') as archive:archive.extractall(dataset)  # Paths validated by audit on the private copy.
        parameters={'data':str((dataset/'data.yaml').resolve()),'device':0,'epochs':1,'imgsz':640,'batch':1,'workers':0,'cache':False,
                    'amp':False,'plots':False,'seed':42,'save':True,'project':str(run.resolve()),'name':'train','exist_ok':False}
        write_json(run/'run-spec.json',{'target':target,'data_version_id':result['data_version_id'],'manifest_sha256':result['manifest_sha256'],'parameters':parameters})
        if args.prepare_only:
            result['status']='prepared_only';return result,run
        probe=inspect_environment(output,target['expected_gpu']);write_json(run/'compute-env.json',probe)
        result['environment']=probe
        if not probe['ready']:raise ValueError('计算环境检查未通过，见compute-env.json；没有回退CPU或自动改装环境。')
        if args.weights is None or not args.weights.is_file() or args.weights.stat().st_size==0:
            raise ValueError('请用--weights指定已存在的本地yolo11n.pt；脚本不会自动下载。')
        if args.weights.name!='yolo11n.pt':raise ValueError('当前烟测固定使用yolo11n.pt，不自动换模型。')
        result['weights_sha256']=file_hash(args.weights)
        import torch
        from ultralytics import YOLO
        model=YOLO(str(args.weights.resolve()))
        if model.task!='detect':raise ValueError('权重不是目标检测模型。')
        yaml_name=Path(model.model.yaml.get('yaml_file','')).stem
        if not yaml_name.startswith('yolo11'):raise ValueError('加载的模型结构不是YOLO11。')
        network=model.model.to('cuda:0').float().eval()
        with torch.inference_mode():
            network(torch.zeros((1,3,640,640),device='cuda:0'));torch.cuda.synchronize()
        result['forward_passed']=True
        batches=[];epochs=[]
        def assert_device(trainer):
            if trainer.device.type!='cuda' or trainer.device.index not in (None,0):raise RuntimeError('训练器没有使用指定CUDA设备。')
        def batch_end(trainer):
            if not torch.isfinite(trainer.loss).all().item():raise RuntimeError('训练损失非有限数。')
            batches.append(1)
        model.add_callback('on_train_start',assert_device)
        model.add_callback('on_train_batch_end',batch_end)
        model.add_callback('on_train_epoch_end',lambda trainer:epochs.append(trainer.epoch))
        torch.cuda.reset_peak_memory_stats(0);result['training_executed']=True
        model.train(**parameters)
        checkpoint=run/'train'/'weights'/'last.pt'
        if len(epochs)!=1 or not batches or not checkpoint.is_file():raise RuntimeError('没有完成一轮训练或缺失last.pt，不能标为成功。')
        reloaded=YOLO(str(checkpoint.resolve()))
        if reloaded.task!='detect':raise RuntimeError('训练权重重新加载检查失败。')
        result.update(status='completed',epochs_completed=len(epochs),batches_completed=len(batches),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      checkpoint={'path':str(checkpoint.resolve()),'sha256':file_hash(checkpoint),'bytes':checkpoint.stat().st_size,'reload_passed':True})
        return result,run
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}');return result,run
    finally:
        result['elapsed_seconds']=round(time.perf_counter()-started,4);write_json(run/'result.json',result)


def main():
    parser=argparse.ArgumentParser(description='在指定计算实例内执行YOLO11单卡单轮测试，不安装依赖、不下载权重。')
    parser.add_argument('--target',type=Path,required=True);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--weights',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true',help='仅准备输入，不检查GPU、不训练，不能作为实机验收')
    args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    try:
        result,run=execute(args);print(json.dumps({'result':result,'run_directory':str(run)},ensure_ascii=False,indent=2))
        return 0 if result['status'] in ('completed','prepared_only') else 2
    except Exception as exc:print(json.dumps({'status':'failed','error':f'{type(exc).__name__}: {exc}'},ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
