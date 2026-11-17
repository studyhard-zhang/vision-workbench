"""Reusable production predictor: load and warm once, then process camera frames."""
import argparse
import json
import time
from pathlib import Path


class Detector:
    def __init__(self, folder=None, *, conf=0.25):
        import math
        if type(conf) not in (int,float) or not math.isfinite(conf) or not 0.001<=conf<=0.95:
            raise ValueError('conf must be a finite confidence threshold between 0.001 and 0.95.')
        import torch
        import tensorrt as trt
        from ultralytics import YOLO
        from build_engine import verify, digest
        import platform
        import subprocess
        self.folder = Path(folder or Path(__file__).resolve().parent)
        verify(self.folder)
        proof = json.loads((self.folder/'engine_environment.json').read_text(encoding='utf-8'))
        gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).strip()
        if proof['engine_sha256'] != digest(self.folder/'model.engine') or proof['gpu_uuid'] != gpu_uuid or proof['os'] != platform.system() or proof['tensorrt'] != trt.__version__:
            raise RuntimeError('Engine hardware/runtime identity changed. Rebuild the package engine on this machine.')
        self.config = json.loads((self.folder/'deployment.json').read_text(encoding='utf-8'))
        self.model = YOLO(str(self.folder/'model.engine'), task=self.config['task_type'])
        self.parameters = dict(imgsz=self.config['imgsz'], device=0, batch=1, half=True,
                               conf=conf, iou=self.config['iou'], max_det=300,
                               rect=False, augment=False, verbose=False, retina_masks=True)
        import numpy as np
        sample = np.zeros((self.config['imgsz'], self.config['imgsz'], 3), dtype=np.uint8)
        for _ in range(5):
            self.model.predict(sample, **self.parameters)
        torch.cuda.synchronize()

    def predict(self, frame):
        """frame is a camera BGR ndarray or a PIL RGB image. No file I/O in timing."""
        import torch
        import cv2
        import numpy as np
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.model.predict(frame, **self.parameters)[0]
        predictions = []
        for i, box in enumerate(result.boxes):
            row = dict(class_index=int(box.cls.item()), class_name=self.model.names[int(box.cls.item())], confidence=float(box.conf.item()), xyxy=box.xyxy[0].cpu().tolist())
            if result.masks is not None:
                mask = result.masks.data[i].cpu().numpy().astype(np.uint8)
                if tuple(mask.shape) != tuple(result.orig_shape):
                    raise RuntimeError('Expected original-resolution segmentation masks.')
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                rings = []
                for index, contour in enumerate(contours):
                    if len(contour) < 3:
                        continue
                    parent, depth = hierarchy[0][index][3], 0
                    while parent >= 0:
                        depth += 1
                        parent = hierarchy[0][parent][3]
                    rings.append(dict(hole=bool(depth % 2), points=contour[:, 0, :].tolist()))
                row['segments'] = rings
            predictions.append(row)
        torch.cuda.synchronize()
        return dict(predictions=predictions, speed_ms=result.speed,
                    processing_ms=(time.perf_counter()-started)*1000,
                    timing_scope='Warm frame-to-result incl. preprocessing, inference, postprocessing and result extraction; excludes exposure, camera transfer, disk I/O and PLC communication.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    from PIL import Image, ImageOps
    with Image.open(args.source) as source:
        oriented = ImageOps.exif_transpose(source)
        rgba = oriented.convert('RGBA')
        white = Image.new('RGBA', rgba.size, 'white')
        white.alpha_composite(rgba)
        frame = white.convert('RGB')
    output = Detector(conf=args.conf).predict(frame)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, allow_nan=False)
