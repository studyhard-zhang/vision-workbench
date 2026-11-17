"""Resume N-08 using the mirror's existing weights/source; never installs or downloads."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

try:
    from .compute_probe import ultralytics_import,inspect_environment
    from .smoke_train import execute,target_config,write_json
except ImportError:
    from compute_probe import ultralytics_import,inspect_environment
    from smoke_train import execute,target_config,write_json


def locate_weights(value, directory):
    supplied=Path(value).expanduser()
    candidates=[supplied] if supplied.is_absolute() else [Path(directory)/supplied,Path('/root')/supplied,Path('/root/autodl-tmp')/supplied,Path('/')/supplied]
    found={p.resolve() for p in candidates if p.is_file() and p.name=='yolo11n.pt'}
    if len(found)!=1:
        raise ValueError('未唯一找到yolo11n.pt。请从JupyterLab复制完整绝对路径传给--weights；检查位置：'+'、'.join(str(p) for p in candidates))
    return found.pop()


def resolve_runtime(weights):
    details=ultralytics_import()
    if details['importable']:return details
    source=Path(weights).resolve().parent.parent
    if (source/'ultralytics'/'__init__.py').is_file():
        # Only the source tree directly associated with the explicitly selected weight.
        for name in list(sys.modules):
            if name=='ultralytics' or name.startswith('ultralytics.'):sys.modules.pop(name,None)
        sys.path.insert(0,str(source))
        details=ultralytics_import();details['candidate_source_root']=str(source)
    return details


def main():
    parser=argparse.ArgumentParser(description='复用镜像已有yolo11n权重及相邻源码，继续单轮GPU测试。')
    parser.add_argument('--weights',default='yolov11/model/yolo11n.pt');parser.add_argument('--directory',type=Path,default=Path.cwd())
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();directory=args.directory.resolve();diagnostic={'training_executed':False}
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    try:
        target=target_config(directory/'execution_target.json');os.environ['CUDA_VISIBLE_DEVICES']=target['gpu_selector']
        weights=locate_weights(args.weights,directory);diagnostic['weights']=str(weights)
        details=resolve_runtime(weights);diagnostic['ultralytics_import']=details
        write_json(directory/'import-resume-r1.json',diagnostic)
        if not details['importable']:
            raise RuntimeError('实际导入Ultralytics失败，详见import-resume-r1.json；可按说明执行bash repair_runtime.sh，不要重装torch或重新下载权重。')
        if args.check_only:
            report=inspect_environment(directory,target['expected_gpu']);write_json(directory/'compute-env-r1.json',report)
            print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['ready'] else 2
        result,run=execute(SimpleNamespace(target=directory/'execution_target.json',bundle=directory/'dataset.zip',weights=weights,output=directory/'runs',prepare_only=False))
        write_json(directory/'resume-result-r1.json',{'result':result,'run_directory':str(run)})
        print(json.dumps({'result':result,'run_directory':str(run)},ensure_ascii=False,indent=2))
        return 0 if result['status']=='completed' else 2
    except Exception as exc:
        diagnostic.update(status='failed',error=f'{type(exc).__name__}: {exc}');write_json(directory/'import-resume-r1.json',diagnostic)
        print(json.dumps(diagnostic,ensure_ascii=False,indent=2));return 2


if __name__=='__main__':raise SystemExit(main())
