"""Read-only, allowlisted Python model exports. No checkpoint deserialization here."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

from .storage import ProjectError
from .model_parameters import model_imgsz
from .versions import encoded


class DeliveryRepository:
    def __init__(self, models, evaluations):
        self.models = models
        self.evaluations = evaluations

    def build(self, pid, tid, mid):
        model = self.models.get(pid, tid, mid)
        if not model['available']:
            raise ProjectError(409, 'MODEL_DAMAGED', '模型完整性不通过，不能生成交付包。')
        manifest = model['manifest']
        if manifest['training_snapshot'].get('task_type')=='segment':raise ProjectError(422,'SEGMENT_DELIVERY_UNAVAILABLE','分割模型可下载.pt权重；当前检测专用Python包不适用。')
        weight_path, _, _ = self.models.artifact(pid, tid, mid, 'best.pt')
        weights = weight_path.read_bytes()
        digest = hashlib.sha256(weights).hexdigest()
        if digest != manifest['artifacts']['best.pt']['sha256']:
            raise ProjectError(409, 'MODEL_DAMAGED', '权重读取校验不符。')
        runtime = {**manifest['receipt']['report'].get('environment', {}).get('packages', {}),
                   **manifest['training_snapshot']['target'].get('runtime', {})}
        # Only export version identifiers; never export SSH/config paths or raw receipts.
        packages = {k: v for k, v in runtime.items() if v and k in {
            'torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow', 'opencv-python', 'dill'}}
        if packages.get('ultralytics') == '8.3.241':
            packages.setdefault('dill', '0.3.8')  # Verified checkpoint dependency in N-13a.
        environment = {'python': runtime.get('python'), 'packages': packages,
                       'scope': 'Reference versions from recorded training environment; not a bundled runtime.',
                       'compatibility': 'Linux x86_64/CUDA reference; other OS/GPU/CPU combinations require separate validation.'}
        metrics = {'status': 'NOT_EVALUATED', 'reason': '尚未成功独立评估；训练内置指标不作为独立评估。',
                   'latest_attempt': model['latest_evaluation'], 'business_accuracy_accepted': False}
        eid = model['latest_successful_evaluation_id']
        if eid:
            report = self.evaluations.report(pid, tid, eid)
            if report['snapshot']['weights_sha256'] != digest:
                raise ProjectError(409, 'EVALUATION_MODEL_MISMATCH', '评估权重与交付权重不一致。')
            metrics.update(status='EVALUATED', reason='独立评估结果；验证集参与过选模，不等同独立测试集或业务精度验收。',
                           evaluation_id=eid, subset=report['snapshot']['subset'],
                           parameters=report['snapshot']['requested_parameters'],
                           metrics=report.get('metrics'), per_class=report.get('per_class'))
        elif model['latest_evaluation']:
            metrics.update(status='NO_SUCCESSFUL_EVALUATION', reason='已有评估尝试，但没有成功的独立评估结果。')
        config = {'model_version_id': mid, 'weights_sha256': digest,
                  'parameters': {'imgsz': model_imgsz(model), 'batch': 1, 'workers': 0, 'half': False,
                                 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'augment': False, 'rect': False},
                  'preprocessing': 'EXIF transpose; alpha composited on white; RGB; Ultralytics letterbox/normalization.',
                  'postprocessing': 'Ultralytics NMS; original EXIF-oriented pixel xyxy, class index and confidence.',
                  'display_confidence_example': 0.25}
        source = {k: model[k] for k in ('id', 'number', 'training_run_id', 'task_data_version_id', 'manifest_sha256')}
        source.update(classes=manifest['classes'], selection_rule=manifest['selection_rule'],
                      requested_training_parameters=manifest['training_snapshot'].get('requested_parameters'),
                      actual_training_parameters=manifest['receipt']['report'].get('actual_parameters'),
                      package_validation='See validation.json for the exact tested model/script/runtime scope.')
        tools = Path(__file__).resolve().parents[1] / 'tools'
        predictor = (tools / 'delivery_predict.py').read_bytes()
        validation = {'status': 'NOT_RUNTIME_VALIDATED', 'reason': '此模型与脚本组合尚未完成独立导出包实机验证。'}
        evidence = tools / 'delivery_reference_validation.json'
        if evidence.is_file():
            record = json.loads(evidence.read_text(encoding='utf-8'))
            if (record['weights_sha256'] == digest and record['model_version_id'] == mid
                    and record['predict_sha256'] == hashlib.sha256(predictor).hexdigest()
                    and record['parameters'] == config['parameters'] | {'device': '0'}):
                validation = record
        readme = f'''# 视训工作台 Python 模型包 M{model['number']:03d}

模型 ID：{mid}；默认权重 best.pt。评估状态：{metrics['status']}，详见 metrics.json。
类别、来源版本及训练参数见 classes.json/source.json。校验清单覆盖除自身外所有包内文件。
包中不含训练原图、标注、SSH配置、口令或私钥；这不是项目备份。

## 使用
将 ZIP 解压到独立目录，保持文件不变。仅加载来源可信的 .pt（加载会反序列化 Python 对象）。
先执行：`python predict.py --check-only`，只检查文件和 SHA，不加载模型、不使用 GPU。
另建独立 Python 环境，按 environment.json/requirements.txt 准备依赖；不要装到工作台控制环境。
torch 的 CUDA wheel 来源与驱动须匹配；requirements.txt 不包含驱动或离线安装包。
示例（输出路径须在包目录之外，且不存在）：
`python predict.py --source /data/image.jpg /data/another.png --output /data/predictions.json --device 0`
脚本不安装依赖、不下载权重、不自动重试或回退 CPU。逐图失败写入 JSON，进程返回2；全成功返回0。
`--device cpu` 是显式可选路径，当前未完成该路径实机兼容验证。
输出 xyxy 对应 EXIF 方向统一后的原尺寸像素。空检测为成功；坏图为失败。
默认保存 conf>=0.001 的候选框；展示可另按0.25筛选，不是重新运行模型。
支持范围以 environment.json 为参考；其他系统、显卡和依赖组合尚未逐一验证。
本模型/脚本的独立运行验证状态：{validation['status']}。详见 validation.json；不能扩展到其他环境。
'''
        files = {'best.pt': weights, 'classes.json': encoded(manifest['classes']),
                 'inference_config.json': encoded(config), 'metrics.json': encoded(metrics),
                 'source.json': encoded(source), 'environment.json': encoded(environment),
                 'requirements.txt': ''.join(f'{k}=={v}\n' for k, v in sorted(packages.items())).encode(),
                 'predict.py': predictor, 'validation.json': encoded(validation),
                 'README.md': readme.encode('utf-8')}
        files['checksums.json'] = encoded({'format_version': 1, 'files': {
            k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k, v in files.items()}})
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, raw)
        return output.getvalue(), f'M{model["number"]:03d}-{mid[:8]}-python.zip'
