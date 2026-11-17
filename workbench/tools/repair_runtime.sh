#!/usr/bin/env bash
# Run only in the observed AutoDL interpreter. No driver/CUDA/torch reinstall.
set -euo pipefail
cd -- "$(dirname -- "$0")"
python - <<'PY'
import importlib.metadata as m
expected={'torch':'2.3.0+cu121','torchvision':'0.18.0+cu121','numpy':'1.26.4','Pillow':'10.3.0'}
for name,value in expected.items():
    current=m.version(name)
    if current!=value:raise SystemExit(f'{name} changed: {current}, expected {value}. Stop and return the report before installing.')
print('Existing CUDA runtime versions verified. They are pinned by constraints.')
PY
python -m pip freeze > packages-before-r1.txt
if python -c 'from ultralytics import YOLO; import cv2; print("Runtime already importable:", cv2.__version__)'; then
  echo 'Reuse existing installed runtime.'
else
  python -m pip install --constraint runtime-constraints-autodl.txt 'ultralytics==8.3.241' 'opencv-python==4.11.0.86'
fi
python -m pip check
python -m pip freeze > packages-after-r1.txt
CUDA_VISIBLE_DEVICES=0 python compute_probe.py --workdir . --expected-gpu '2080 Ti' --output compute-env-r1.json
