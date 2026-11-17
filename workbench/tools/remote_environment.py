"""Read-only validation of a user-managed Linux training environment."""
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
import contextlib
import io
from pathlib import Path

WEIGHTS = {'yolo26m-seg.pt': '16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5', 'yolo26m.pt': '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7', 'yolo26n-seg.pt': '361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f', 'yolo26n.pt': '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'}
DEPENDENCIES = {'ultralytics': '8.4.203', 'opencv-python': '4.11.0.86', 'dill': '0.3.8', 'numpy': '1.26.4'}
RECIPES = {'cu121': ('2.3.0+cu121', '0.18.0+cu121'), 'cu128': ('2.7.1+cu128', '0.22.1+cu128')}
WEIGHT_SOURCE = 'https://huggingface.co/Ultralytics/YOLO26/resolve/070ac3c51435984d992ba92eb3f234daf4e4500d/'
ROOTS = ('/root/autodl-tmp/vision-workbench', '/root/.vision-workbench')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def packages():
    result = {}
    for name in ('torch', 'torchvision', *DEPENDENCIES):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def inspect(root_override=None):
    installed = packages()
    root = Path(ROOTS[0] if Path('/root/autodl-tmp').is_dir() else ROOTS[1])
    if root_override:
        if root_override not in ROOTS:raise RuntimeError('工作目录须为 /root/.vision-workbench 或 /root/autodl-tmp/vision-workbench。')
        root=Path(root_override)
    volume=root
    while not volume.exists() and volume!=volume.parent:volume=volume.parent
    free = shutil.disk_usage(volume).free
    blockers = []
    gpu = None
    try:
        raw = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid,name,driver_version,compute_cap', '--format=csv,noheader'], text=True, timeout=12)
        identity, name, driver, capability = [part.strip() for part in raw.strip().split(',')]
        gpu = dict(uuid=identity, name=name, driver=driver, capability=capability)
    except (OSError, ValueError, subprocess.SubprocessError):
        blockers.append('找不到可用的 NVIDIA GPU 或驱动，请选择带 GPU 的镜像并检查 nvidia-smi。')
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        blockers.append('训练执行器需要 Linux x86_64；Windows 本机请自行准备 WSL2 Ubuntu 并在 WSL 中安装训练环境。')
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        blockers.append('请使用 Python 3.10–3.12 的实例镜像。')
    recipe = 'cu121'
    if gpu:
        if float(gpu['capability']) >= 10 or (installed['torch'],installed['torchvision'])==RECIPES['cu128']:
            recipe = 'cu128'
        minimum = (570, 26) if recipe == 'cu128' else (525, 60, 13)
        if tuple(map(int, gpu['driver'].split('.'))) < minimum:
            blockers.append('NVIDIA驱动版本过低，请由用户或主机管理员升级到兼容所需CUDA的驱动，再重新校验。')
    torch, vision = RECIPES[recipe]
    desired = {'torch': torch, 'torchvision': vision, **DEPENDENCIES}
    reuse = installed['torch'] == torch and installed['torchvision'] == vision
    missing = [{'name': name, 'required': version, 'installed': installed[name]} for name, version in desired.items() if installed[name] != version]
    weight_dir = root / 'environment-v1' / 'weights'
    weight_missing = [name for name, sha in WEIGHTS.items() if not (weight_dir / name).is_file() or digest(weight_dir / name) != sha]
    if not root.is_dir() or not os.access(root,os.W_OK):blockers.append('请自行创建并授予训练账号写权限：'+str(root))
    if free < 2 * 1024**3:blockers.append('训练工作盘至少需要2GiB可用空间，实际训练需按数据和模型预留更多。请清理磁盘。')
    guidance=[]
    if missing:
        guidance.append(f'在所选环境手动执行：{sys.executable} -m pip install torch=={torch.split("+")[0]} torchvision=={vision.split("+")[0]} --index-url https://download.pytorch.org/whl/{recipe}')
        guidance.append(f'然后执行：{sys.executable} -m pip install '+ ' '.join(f'{name}=={version}' for name,version in DEPENDENCIES.items()))
    if weight_missing:guidance.append('用户下载对应模型并放入 '+str(weight_dir)+'。官方固定来源：'+WEIGHT_SOURCE+'。缺失的可选模型只影响对应模式。')
    return dict(gpu=gpu, python=platform.python_version(), executable=sys.executable, root=str(root), free_bytes=free,
                packages=installed, missing=missing, missing_weights=[name for name in weight_missing if name=='yolo26n.pt'],
                optional_missing_weights=[name for name in weight_missing if name!='yolo26n.pt'],available_weights=[name for name in WEIGHTS if name not in weight_missing],
                guidance=guidance,recipe=recipe, reuse_torch=reuse, blockers=blockers)


def validate_existing(plan):
    """Validate the user's existing environment without installations or training."""
    lock=None
    try:
        path=Path.home()/'.vision-workbench-locks/gpu-0.lock'
        if path.exists():
            import fcntl
            lock=path.open('r+b')
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:raise RuntimeError('GPU正由训练或推理任务占用，请任务结束后再进行计算能力校验。') from None
        return validate_runtime(plan)
    finally:
        if lock:lock.close()


def validate_runtime(plan):
    if plan['blockers'] or plan['missing'] or plan['missing_weights']:
        return dict(ready=False,plan=plan,validation_kind='environment')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='vision-environment-check-') as temporary:
        os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=temporary;os.environ['CUDA_VISIBLE_DEVICES']='0'
        import torch
        from torchvision.ops import nms
        from ultralytics import YOLO
        if not torch.cuda.is_available():raise RuntimeError('PyTorch 无法使用CUDA。请由用户安装与GPU及驱动匹配的CUDA版PyTorch/Torchvision后重新校验。')
        if torch.cuda.device_count()<1:raise RuntimeError('Python环境未检测到GPU0，请检查CUDA_VISIBLE_DEVICES和GPU分配。')
        torch.zeros((2,2),device='cuda:0').matmul(torch.ones((2,2),device='cuda:0'))
        nms(torch.tensor([[0.,0.,4.,4.]],device='cuda:0'),torch.ones(1,device='cuda:0'),.5)
        tests={}
        for filename,sha in WEIGHTS.items():
            if filename not in plan['available_weights']:continue
            model=YOLO(str(Path(plan['root'])/'environment-v1/weights'/filename))
            task='segment' if '-seg' in filename else 'detect'
            if model.task!=task:raise RuntimeError('权重任务类型不符：'+filename)
            with torch.inference_mode():model.model.to('cuda:0').float().eval()(torch.zeros((1,3,64,64),device='cuda:0'))
            torch.cuda.synchronize()
            tests[filename]=dict(run_id='environment-'+uuid.uuid4().hex,environment_checked=True,forward_passed=True,weights_sha256=sha,training_executed=False)
            del model;torch.cuda.empty_cache()
    return dict(ready=True,validation_kind='environment',tests=tests,python=sys.executable,
                runtime={**{name:plan['packages'][name] for name in ('torch','torchvision','ultralytics')},'python':plan['python']})


def main():
    command = json.loads(base64.b64decode(sys.argv[1]))
    try:
        if command['action'] not in ('inspect','validate'):raise RuntimeError('软件仅校验已有环境，请由用户自行搭建环境后重新连接。')
        plan = inspect(command.get('root'))
        if command['action']=='inspect':value=plan
        else:
            with contextlib.redirect_stdout(io.StringIO()):value=validate_existing(plan)
        print(json.dumps(dict(ok=True, value=value)))
    except Exception as error:
        message = f'{type(error).__name__}: {str(error)[:1000]}。请核对所选Python环境中的CUDA版torch/torchvision及权重依赖，处理后重新连接校验。'
        if isinstance(error,ModuleNotFoundError):
            package={'PIL':'Pillow','cv2':'opencv-python','yaml':'PyYAML','dill':'dill','ultralytics':'ultralytics==8.4.203','numpy':'numpy==1.26.4'}.get((error.name or '').split('.')[0])
            if package:message=f'缺少模块{error.name}。请由用户在所选环境执行：{sys.executable} -m pip install {package}，然后重新校验。'
        print(json.dumps(dict(ok=False, error=message)))


if __name__ == '__main__':
    main()
