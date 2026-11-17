"""TensorRT derived cache and production bundle, called under the inference GPU lock."""
import hashlib
import json
import platform
import shutil
import statistics
import time
import uuid
import zipfile
from pathlib import Path


def benchmark(model, image, parameters):
    import torch
    for _ in range(5):
        model.predict(image, **parameters, save=False, verbose=False)
    samples = []
    inference = []
    last = None
    for _ in range(20):
        torch.cuda.synchronize()
        started = time.perf_counter()
        last = model.predict(image, **parameters, save=False, verbose=False)[0]
        last.boxes.xyxy.cpu()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-started)*1000)
        inference.append(float(last.speed['inference']))
    return dict(iterations=20, warmup=5, processing_median_ms=statistics.median(samples),
                processing_p95_ms=sorted(samples)[18], inference_median_ms=statistics.median(inference)), last


def prepare(directory, request, native, image, support):
    import tensorrt as trt
    import onnx
    from ultralytics import YOLO
    snap = request['snapshot']
    expected = snap['acceleration']['runtime']
    if trt.__version__ != expected['tensorrt'] or onnx.__version__ != expected['onnx']:
        raise RuntimeError('TensorRT/ONNX versions changed; inspect the acceleration environment before retrying.')
    params = snap['requested_parameters']
    binding = dict(weights_sha256=request['weights_sha256'], gpu_uuid=request['expected_gpu_uuid'],
                   runtime=expected, ultralytics=snap['target']['runtime']['ultralytics'],
                   imgsz=params['imgsz'], batch=1, precision='FP16', os=platform.system(),
                   architecture=platform.machine(), opset=17, simplify=False)
    key = hashlib.sha256(support.encoded(binding)).hexdigest()
    workspace = Path(snap['target'].get('workspace_root', '/root/.vision-workbench'))
    if workspace not in (Path('/root/.vision-workbench'), Path('/root/autodl-tmp/vision-workbench')):
        raise RuntimeError('Unsupported acceleration cache root.')
    root = workspace/'environment-v1'/'engine-cache'
    root.mkdir(parents=True, exist_ok=True)
    cache = root/key
    reused = cache.exists()
    if not reused:
        stage = root/('building-'+uuid.uuid4().hex)
        stage.mkdir()
        shutil.copyfile(directory/'model.pt', stage/'model.pt')
        support.progress(directory, 'optimizing_engine', images_completed=0, images_requested=snap['sample_count'])
        exporter = YOLO(str(stage/'model.pt'))
        exporter.export(format='engine', imgsz=params['imgsz'], batch=1, dynamic=False,
                        quantize=16, device=0, simplify=False, opset=17, workspace=1, verbose=False)
        engine = stage/'model.engine'
        with engine.open('rb') as stream:
            length = int.from_bytes(stream.read(4), 'little')
            if not 0 < length < 1024**2:
                raise RuntimeError('Engine metadata missing.')
            raw_metadata = json.loads(stream.read(length))
        metadata = {k:raw_metadata[k] for k in ('stride','task','batch','imgsz','names','end2end','channels','version') if k in raw_metadata}
        (stage/'metadata.json').write_bytes(support.encoded(metadata))
        record = dict(binding=binding, engine_sha256=support.sha(engine), onnx_sha256=support.sha(stage/'model.onnx'))
        (stage/'cache.json').write_bytes(support.encoded(record))
        stage.rename(cache)
    record = json.loads((cache/'cache.json').read_text(encoding='utf-8'))
    if record['binding'] != binding or support.sha(cache/'model.engine') != record['engine_sha256'] or support.sha(cache/'model.onnx') != record['onnx_sha256']:
        raise RuntimeError('Acceleration cache integrity failed. Keep the cache for diagnosis; no rebuild or fallback was attempted.')
    import torch
    baseline, native_result = benchmark(native, image, {**params, 'half':False})
    engine = YOLO(str(cache/'model.engine'), task=snap.get('task_type','detect'))
    accelerated, engine_result = benchmark(engine, image, params)
    threshold = .25
    counts = dict(pytorch=sum(float(x)>=threshold for x in native_result.boxes.conf),
                  tensorrt=sum(float(x)>=threshold for x in engine_result.boxes.conf), threshold=threshold)
    acceleration = dict(backend='tensorrt', cache_key=key, cache_reused=reused,
                        binding=binding, engine_sha256=record['engine_sha256'],
                        onnx_sha256=record['onnx_sha256'], pytorch=baseline, tensorrt=accelerated,
                        prediction_counts=counts, business_accuracy_accepted=False,
                        scope='Same canonical RGB image, batch1, 5 warmups and 20 synchronized runs. Processing includes preprocessing, model and postprocessing plus box transfer; excludes exposure, camera transport, file I/O, SSH, engine build and PLC. Segmentation contour extraction is measured separately per image.')
    return engine, cache, acceleration


def deployment(directory, request, cache, acceleration, support):
    snap = request['snapshot']
    config = dict(format_version=1, cuda_profile=snap['acceleration']['deployment_profile'],
                  task_type=snap.get('task_type','detect'), imgsz=snap['requested_parameters']['imgsz'],
                  conf=snap['requested_parameters']['conf'], iou=snap['requested_parameters']['iou'],
                  batch=1, precision='FP16', source_weights_sha256=request['weights_sha256'],
                  source_model_id=snap['model_version_id'], source_onnx_sha256=acceleration['onnx_sha256'])
    files = {'model.onnx':(cache/'model.onnx').read_bytes(), 'metadata.json':(cache/'metadata.json').read_bytes(),
             'deployment.json':support.encoded(config), 'benchmark.json':support.encoded(acceleration),
             'classes.json':support.encoded(snap['classes']),
             'build_engine.py':(directory/'deployment_build.py').read_bytes(),
             'predict.py':(directory/'deployment_predict_trt.py').read_bytes()}
    files['README.md'] = ('''# 产线 TensorRT 模型包

本包包含 ONNX，不包含开发机 engine。先在实际产线 GPU/系统上构建一次引擎。
CUDA11 配置对应 RTX3050 / TensorRT8.6 兼容环境；CUDA12 配置对应 RTX5050 / CUDA12.8+ / TensorRT10.9+ 兼容环境。
CUDA 大版本本身不足以保证兼容；驱动、TensorRT、系统及 GPU 都须匹配。工具不安装依赖、不下载权重、不回退 CPU。
本机 RTX5060 的 benchmark.json 只是本机实测，不代表产线3050/5050速度或业务精度。

1. 在已配好的产线 Python 环境中运行 python build_engine.py --check-only 校验文件（不用GPU）。
2. 运行 python build_engine.py，在本机生成 model.engine 和 engine_environment.json。
3. python predict.py --source camera-test.png --output result.json，验证实际产线图片。
4. 连续检测时复用 predict.Detector 实例，初始化只执行一次；传入相机 BGR ndarray 或 PIL RGB 图像。
   detector = Detector(); result = detector.predict(frame)
   产线默认置信度0.25；可用 Detector(conf=0.5) 或命令行 --conf 0.5 调整，无需重建引擎。
   deployment.json中的conf是工作台保存预测的最低阈值0.001，用于前端即时筛选；benchmark沿用该阈值。

所需环境：TensorRT、ONNX、Ultralytics8.4.203、与所选CUDA匹配的PyTorch/Torchvision、Pillow、OpenCV和NumPy。
请使用工控机已验证的环境，不要让pip自动替换CUDA/PyTorch。CUDA11建议TensorRT8.6，CUDA12/RTX50需TensorRT10.9+及对应驱动。
FP16可能改变个别框、置信度和掩膜，请用有代表性的产线样本复核漏检、误检和边界后再验收。
processing_ms为预热后图像到结果（含结果提取）的耗时，不含曝光、取图、磁盘、PLC通信。
源模型、类别与ONNX SHA在deployment.json/classes.json。包内SHA用于完整性核对，并非数字签名。
''').encode('utf-8')
    files['checksums.json'] = support.encoded(dict(files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()}))
    with zipfile.ZipFile(directory/'deployment.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items():
            archive.writestr(name,data)
