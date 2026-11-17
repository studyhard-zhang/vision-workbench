"""N-08 compute-host preflight. Never installs packages or downloads weights."""
import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import pickletools
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def package_versions():
    result={}
    for name in ('torch','torchvision','ultralytics','numpy','Pillow','opencv-python','opencv-python-headless','dill'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


def checkpoint_dependencies(path):
    """Inspect pickle opcodes without unpickling/importing the model or optional modules.

    This detects known optional runtime dependencies in the pinned PyTorch ZIP;
    it is not a general pickle safety validator or a model loading test.
    """
    with zipfile.ZipFile(path) as archive:
        names=[name for name in archive.namelist() if name=='data.pkl' or name.endswith('/data.pkl')]
        if len(names)!=1 or archive.getinfo(names[0]).file_size>8*1024**2:raise ValueError('权重pickle元数据缺失、重复或过大。')
        roots=set();supported=True
        for operation,argument,_ in pickletools.genops(archive.read(names[0])):
            if operation.name=='GLOBAL':roots.add(argument.split(' ',1)[0].split('.',1)[0])
            elif operation.name=='STACK_GLOBAL':supported=False
    required=sorted(roots & {'dill'});versions={}
    for name in required:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'scan_supported':supported,'global_module_roots':sorted(roots),'optional_dependencies':versions,
            'missing':[name for name,version in versions.items() if version is None],
            'checkpoint_deserialized':False,'gpu_used':False}


def ultralytics_import():
    """A mirror may expose a source checkout without installed distribution metadata."""
    try:distribution=importlib.metadata.version('ultralytics')
    except importlib.metadata.PackageNotFoundError:distribution=None
    result={'distribution_version':distribution,'importable':False}
    try:
        module=importlib.import_module('ultralytics')
        result.update(version=str(getattr(module,'__version__','unknown')),origin=str(getattr(module,'__file__','unknown')))
        result['importable']=callable(getattr(module,'YOLO',None))
        if not result['importable']:result['error']='ultralytics模块未提供可调用的YOLO接口。'
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
    return result


def inspect_environment(workdir, expected_gpu='', metadata_only=False):
    folder=Path(workdir).resolve();checks=[]
    report={'format_version':1,'checked_at':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
            'python_executable':sys.executable,'platform':platform.platform(),'workdir':str(folder),
            'packages':package_versions(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'metadata_only':metadata_only,'training_executed':False,'checks':checks}
    def check(code,passed,message):checks.append({'code':code,'status':'passed' if passed else 'failed','message':message})
    check('python',sys.version_info>=(3,10),'需要Python3.10或以上。')
    check('workdir',folder.is_dir(),'工作目录必须已存在。')
    if folder.is_dir():
        report['disk_free_bytes']=shutil.disk_usage(folder).free
        check('disk',report['disk_free_bytes']>=2*1024**3,'烟测开始前至少保留2GiB；实际训练还需按数据/产物核算。')
    if metadata_only:
        report.update(ready=False,reason='仅收集元数据；GPU实算未执行。');return report
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        report['nvidia_smi']=result.stdout.strip() if result.returncode==0 else result.stderr.strip()
        check('driver_query',result.returncode==0,'nvidia-smi驱动查询。')
    except (OSError,subprocess.TimeoutExpired) as exc:
        report['nvidia_smi']=str(exc);check('driver_query',False,'无法执行nvidia-smi。')
    try:
        import torch
        report.update(torch_version=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),visible_gpu_count=torch.cuda.device_count())
        if not report['cuda_available'] or report['visible_gpu_count']<1:
            raise RuntimeError('当前解释器没有可用CUDA GPU；不能静默改用CPU。')
        props=torch.cuda.get_device_properties(0)
        report['gpu']={'logical_index':0,'name':props.name,'capability':list(torch.cuda.get_device_capability(0)),
                       'total_memory_bytes':props.total_memory,'uuid':str(getattr(props,'uuid','unavailable'))}
        check('expected_gpu',not expected_gpu or expected_gpu.casefold() in props.name.casefold(),'期望GPU：'+(expected_gpu or '未限制型号')+'；实际：'+props.name)
        with torch.no_grad():
            value=torch.ones((64,64),device='cuda:0');product=value@value;torch.cuda.synchronize()
            check('cuda_matmul',bool(torch.isfinite(product).all().item()) and abs(product.mean().item()-64)<0.001,'CUDA矩阵乘与同步。')
            del value,product
        import torchvision
        from torchvision.ops import nms
        report['torchvision_version']=torchvision.__version__
        boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.],[30.,30.,40.,40.]],device='cuda:0')
        scores=torch.tensor([.9,.8,.7],device='cuda:0');kept=nms(boxes,scores,.5);torch.cuda.synchronize()
        check('cuda_nms',kept.device.type=='cuda' and kept.tolist()==[0,2],'torchvision CUDA NMS。')
        del boxes,scores,kept
    except Exception as exc:check('cuda_runtime',False,f'{type(exc).__name__}: {exc}')
    report['ultralytics_import']=ultralytics_import()
    check('ultralytics',report['ultralytics_import']['importable'],'Ultralytics实际导入与YOLO接口检查；'+report['ultralytics_import'].get('error','来源：'+report['ultralytics_import'].get('origin','unknown')))
    report['ready']=bool(checks) and all(c['status']=='passed' for c in checks)
    return report


def main():
    parser=argparse.ArgumentParser(description='只探测当前计算实例，不安装依赖、不下载模型。')
    parser.add_argument('--workdir',type=Path,default=Path.cwd());parser.add_argument('--expected-gpu',default='')
    parser.add_argument('--metadata-only',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=inspect_environment(args.workdir,args.expected_gpu,args.metadata_only)
    payload=json.dumps(report,ensure_ascii=False,indent=2)
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    print(payload);args.output.write_text(payload+'\n',encoding='utf-8')
    return 0 if report['ready'] else 2


if __name__=='__main__':raise SystemExit(main())
