"""Standalone exported model runner. --check-only needs only Python's standard library."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def verify():
    manifest = read('checksums.json')
    expected = set(manifest['files']) | {'checksums.json'}
    if {p.name for p in ROOT.iterdir() if p.is_file()} != expected:
        raise ValueError('Package file list changed; extract into a dedicated directory.')
    for name, spec in manifest['files'].items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid manifest path')
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or linked package file: ' + name)
        raw = path.read_bytes()
        if len(raw) != spec['bytes'] or hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Package checksum mismatch: ' + name)
    return manifest


def predict(sources, output, device):
    # Never install dependencies or download a model automatically.
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    config = read('inference_config.json')
    environment = read('environment.json')
    for package, version in environment['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Runtime differs from reference: ' + package + ' expected ' + version)
    import torch
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    if device == '0' and not torch.cuda.is_available():
        raise ValueError('CUDA GPU unavailable; no automatic CPU fallback')
    model = YOLO(str(ROOT / 'best.pt'))
    classes = read('classes.json')
    if model.task != 'detect' or [model.names[i] for i in range(len(model.names))] != [c['name'] for c in classes]:
        raise ValueError('Model task/classes mismatch')
    params = config['parameters'] | {'device': device}
    rows = []
    for source in sources:
        row = {'filename': str(source), 'status': 'FAILED', 'predictions': []}
        try:
            with Image.open(source) as image:
                image.load()
                oriented = ImageOps.exif_transpose(image)
                rgba = oriented.convert('RGBA')
                background = Image.new('RGBA', rgba.size, 'white')
                background.alpha_composite(rgba)
                canonical = background.convert('RGB')
            result = model.predict(source=canonical, **params, save=False, verbose=False)[0]
            if tuple(result.orig_shape) != (canonical.height, canonical.width):
                raise ValueError('Coordinate dimensions mismatch')
            if device == '0' and model.predictor.device.type != 'cuda':
                raise ValueError('GPU not used')
            boxes = result.boxes
            row.update(status='SUCCEEDED', width=canonical.width, height=canonical.height,
                predictions=[{'class_index': int(c), 'class_name': classes[int(c)]['name'],
                              'confidence': float(score), 'xyxy': [float(x) for x in box]}
                             for c, score, box in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist())])
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
    value = {'model_version_id': config['model_version_id'], 'weights_sha256': config['weights_sha256'],
             'coordinate_system': 'exif-oriented-pixels', 'parameters': params, 'images': rows}
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Output must be outside the immutable model package')
    # Exclusive creation prevents accidental overwriting of an earlier result.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return 0 if all(r['status'] == 'SUCCEEDED' for r in rows) else 2


def main():
    parser = argparse.ArgumentParser(description='Standalone YOLO11 detection model delivery')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--source', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['0', 'cpu'], default='0')
    args = parser.parse_args()
    try:
        verify()
        if args.check_only:
            print('Package SHA-256/file list OK. Model was not loaded; GPU was not used.')
            return 0
        if not args.source or not args.output:
            parser.error('--source and --output are required for prediction')
        return predict(args.source, args.output, args.device)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
