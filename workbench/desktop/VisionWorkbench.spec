# Build from the project root using the isolated desktop build environment.
from pathlib import Path
root=Path(SPECPATH).parent
datas=[(str(root/'web'),'web')]
for name in ['polygon_geometry.py','verify_dataset.py','run_training_job.py','training_config.py','compute_probe.py','ssh_control.py','run_evaluation_job.py','run_inference_job.py','remote_environment.py','tensorrt_support.py','deployment_build.py','deployment_predict_trt.py','start_local_wsl.ps1']:
    datas.append((str(root/'tools'/name),'tools'))
for name in ['run.py','requirements.txt','RELEASE.md']:
    datas.append((str(root/name),'.'))
datas.append((str(root/'desktop/THIRD_PARTY_NOTICES.txt'),'.'))
datas.append((str(root/'desktop/使用说明.md'),'.'))
datas.append((str(root.parent/'NOTICE.md'),'.'))
a=Analysis([str(root/'desktop.py')],pathex=[str(root)],binaries=[],datas=datas,
    hiddenimports=['webview.platforms.edgechromium','webview.platforms.winforms','PIL.ImageFont','PIL.PngImagePlugin','PIL.JpegImagePlugin'],
    hookspath=[],hooksconfig={},runtime_hooks=[],excludes=['torch','torchvision','ultralytics','PyQt5','PyQt6','PySide2','PySide6','tkinter'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='VisionWorkbench',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False,icon=str(root/'desktop'/'app.ico'))
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='VisionWorkbench')
