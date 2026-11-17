"""Build the ONNX package on the production machine; no installs or downloads."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(folder):
    manifest = json.loads((folder/'checksums.json').read_text(encoding='utf-8'))
    for name, spec in manifest['files'].items():
        path = folder/name
        if Path(name).name != name or path.is_symlink() or not path.is_file() or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise RuntimeError('Package integrity failed: '+name)


def build(folder):
    verify(folder)
    import torch
    import tensorrt as trt
    config = json.loads((folder/'deployment.json').read_text(encoding='utf-8'))
    if not torch.version.cuda or int(torch.version.cuda.split('.')[0]) != (11 if config['cuda_profile']=='cuda11' else 12):
        raise RuntimeError('Configured CUDA profile does not match the installed PyTorch CUDA runtime. Use the preconfigured target environment, without automatic dependency replacement.')
    hardware = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=name,uuid,compute_cap,driver_version', '--format=csv,noheader'], text=True, timeout=15).strip()
    name, identity, capability, driver = [s.strip() for s in hardware.split(',')]
    if float(capability)>=10 and tuple(map(int,torch.version.cuda.split('.')[:2]))<(12,8):
        raise RuntimeError('RTX50 requires a Blackwell-compatible CUDA runtime (12.8 or newer) and driver; CUDA12 alone is insufficient.')
    major = int(trt.__version__.split('.')[0])
    if config['cuda_profile'] == 'cuda11' and ((major != 8 or tuple(map(int,trt.__version__.split('.')[:2]))<(8,6)) or float(capability) >= 10):
        raise RuntimeError('CUDA11 profile requires a compatible TensorRT 8.x environment on the RTX3050; RTX50 requires CUDA12/TensorRT10.9+.')
    if config['cuda_profile'] == 'cuda12' and tuple(map(int, trt.__version__.split('.')[:2])) < (10, 9):
        raise RuntimeError('CUDA12/RTX50 profile requires TensorRT 10.9 or newer and a compatible NVIDIA driver.')
    engine_path = folder/'model.engine'
    if engine_path.exists():
        raise RuntimeError('model.engine already exists. Keep it and build a fresh package directory to avoid overwriting an engine.')
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse((folder/'model.onnx').read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    settings = builder.create_builder_config()
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024**3)
    settings.set_flag(trt.BuilderFlag.FP16)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError('TensorRT engine build failed; inspect the TensorRT log. No precision or shape fallback was used.')
    metadata = (folder/'metadata.json').read_bytes()
    with engine_path.open('xb') as stream:
        stream.write(len(metadata).to_bytes(4, 'little'))
        stream.write(metadata)
        stream.write(bytes(plan))
    proof = dict(engine_sha256=digest(engine_path), source_onnx_sha256=digest(folder/'model.onnx'),
                 gpu=name, gpu_uuid=identity, compute_capability=capability, driver=driver,
                 os=platform.system(), architecture=platform.machine(), tensorrt=trt.__version__,
                 cuda_profile=config['cuda_profile'], cuda_runtime=torch.version.cuda, precision='FP16', batch=1, imgsz=config['imgsz'])
    (folder/'engine_environment.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print('Engine built on this machine. Run predict.py to validate production images and measure latency.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    if args.check_only:
        verify(directory)
        print('Package integrity OK; no GPU was used.')
    else:
        build(directory)
