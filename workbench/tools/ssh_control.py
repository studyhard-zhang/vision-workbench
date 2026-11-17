"""Fixed-purpose remote control, delivered over a pinned SSH connection. Standard library only."""
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import zipfile
import errno
from pathlib import Path

ROOT=Path('/root/autodl-tmp/vision-workbench')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):value.update(chunk)
    return value.hexdigest()


def read(path,default=None):return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def identity(pid,directory):
    try:
        stat=(Path('/proc')/str(pid)/'stat').read_text();parts=stat[stat.rfind(')')+2:].split()
        if parts[0]=='Z':return None
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
        if str(directory/'run_training_job.py') not in cmd or str(directory) not in cmd:return None
        return parts[19]  # proc starttime, after the parenthesized comm field
    except (OSError,ValueError,IndexError):return None


def resource_state(directory):
    """Nonblocking cross-process checks. Do not infer release from a result file or PID alone."""
    import fcntl
    paths=[directory/'.job.lock',Path.home()/'.vision-workbench-locks/gpu-0.lock']
    handles=[]
    try:
        for path in paths:
            f=path.open('r+b');handles.append(f)
            try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES,errno.EAGAIN):return {'checked':True,'free':False,'reason':'managed_lock_held'}
                raise
        return {'checked':True,'free':True,'reason':'managed_locks_released'}
    except OSError:
        return {'checked':False,'free':False,'reason':'managed_locks_unavailable'}
    finally:
        for f in handles:f.close()


def status(directory):
    launch=read(directory/'ssh-launch.json',{})
    pid=launch.get('pid');tick=identity(pid,directory) if type(pid) is int else None
    result=directory/'return-result.zip'
    log=directory/'console.log'
    if not log.is_file():log=directory/'ssh-launcher.log'
    tail=''
    if log.is_file():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-24000));tail=f.read().decode('utf-8',errors='replace')
    try:
        volume=directory
        while not volume.exists() and volume!=volume.parent:volume=volume.parent
        storage={'checked':True,'free_bytes':shutil.disk_usage(volume).free}
    except OSError:storage={'checked':False,'free_bytes':None}
    return {'directory_exists':directory.is_dir(),'launch':launch,'alive':bool(tick and tick==launch.get('start_tick')),'storage':storage,
            'started':(directory/'started.json').is_file(),'result_ready':result.is_file(),
            'result_sha256':digest(result) if result.is_file() else None,
            'progress':read(directory/'progress.json',{}),'log_tail':tail,
            'resources':resource_state(directory) if result.is_file() else {'checked':False,'free':False,'reason':'result_not_ready'}}


def handle(command):
    action=command['action']
    if action=='probe':
        smi=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=12)
        return {'python':sys.version.split()[0],'python_executable':sys.executable,'nvidia_smi':smi.stdout.strip(),'nvidia_smi_ok':smi.returncode==0}
    smi=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=12)
    if smi.returncode or command.get('expected_gpu_uuid') not in smi.stdout.splitlines():raise RuntimeError('GPU identity changed; no control action performed')
    job_id=command['job_id']
    if not isinstance(job_id,str) or not re.fullmatch('[0-9a-f]{32}',job_id):raise ValueError('Invalid job identity')
    workspace=Path(command.get('workspace_root',ROOT))
    if workspace not in (ROOT,Path('/root/autodl-tmp/vision-workbench'),Path('/root/.vision-workbench')):raise ValueError('Remote workspace not supported')
    directory=workspace/job_id
    if directory.resolve().parent!=workspace.resolve():raise ValueError('Remote path ownership mismatch')
    if action=='status':return status(directory)
    if action=='verify-release':
        state=status(directory)
        state['resources']=resource_state(directory)
        return state
    if action=='prepare':directory.mkdir(parents=True,exist_ok=True);return status(directory)
    if action in ('install','launch'):
        if (directory/'ssh-launch.json').exists():return status(directory)
        source=directory/'execution.zip.upload'
        if not source.is_file() or digest(source)!=command['package_sha256']:raise ValueError('Execution package SHA mismatch')
        source.replace(directory/'execution.zip')
        allowed={'dataset.zip','request.json','README.txt','run_training_job.py','compute_probe.py','verify_dataset.py'}
        with zipfile.ZipFile(directory/'execution.zip') as archive:
            names=archive.namelist()
            evaluation={'dataset.zip','request.json','README.txt','run_training_job.py','training_support.py','compute_probe.py','verify_dataset.py','model.pt'}
            if len(names)!=len(set(names)) or set(names) not in (allowed,allowed|{'training_config.py'},allowed|{'training_config.py','polygon_geometry.py'},allowed|{'training_config.py','polygon_geometry.py','initial.pt'},evaluation,evaluation|{'inference_support.py'},evaluation|{'inference_support.py','tensorrt_support.py','deployment_build.py','deployment_predict_trt.py'}):raise ValueError('Invalid execution package members')
            archive.extractall(directory)
        request=read(directory/'request.json')
        if request['job_id']!=job_id:raise ValueError('Package job mismatch')
        if action=='install':return status(directory)
        with (directory/'ssh-launch.json').open('x',encoding='utf-8') as f:json.dump({'claimed':True},f);f.flush();os.fsync(f.fileno())
        with (directory/'ssh-launcher.log').open('ab') as log:
            process=subprocess.Popen([sys.executable,'-u',str(directory/'run_training_job.py'),'--directory',str(directory)],cwd=directory,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        tick=None
        for _ in range(10):
            tick=identity(process.pid,directory)
            if tick:break
            time.sleep(.05)
        temporary=directory/'ssh-launch.json.tmp';temporary.write_text(json.dumps({'pid':process.pid,'start_tick':tick}),encoding='utf-8');temporary.replace(directory/'ssh-launch.json')
        return status(directory)
    if action=='stop':
        before=status(directory)
        if before['result_ready']:return before
        launch=before['launch'];pid=launch.get('pid')
        if type(pid) is int:
            tick=identity(pid,directory)
            if tick and tick!=launch.get('start_tick'):raise RuntimeError('Process identity changed; no signal sent')
            if tick:
                if pid<=1 or os.getpgid(pid)!=pid:raise RuntimeError('Process group mismatch; no signal sent')
                os.killpg(pid,signal.SIGTERM)
                for _ in range(100):
                    if not identity(pid,directory):break
                    time.sleep(.1)
                if identity(pid,directory):os.killpg(pid,signal.SIGKILL)
        # Data-loader children can briefly retain inherited locks after the main
        # runner exits. Wait for their release before attempting CPU-only closure.
        deadline=time.monotonic()+15
        while (directory/'.job.lock').is_file():
            resources=resource_state(directory)
            if not resources['checked'] or resources['free']:break
            if time.monotonic()>=deadline:return status(directory)
            time.sleep(.1)
        # The runner obtains both flock locks, verifies interprocess exclusion and never trains in this mode.
        result=subprocess.run([sys.executable,str(directory/'run_training_job.py'),'--directory',str(directory),'--recover-interrupted'],capture_output=True,text=True,timeout=25)
        state=status(directory)
        if not state['result_ready']:raise RuntimeError('Remote stop not confirmed: '+(result.stdout+result.stderr)[-1200:])
        return state
    raise ValueError('Unsupported control action')


if __name__=='__main__':
    import base64
    try:print(json.dumps({'ok':True,'value':handle(json.loads(base64.b64decode(sys.argv[1])))},ensure_ascii=False))
    except Exception as exc:print(json.dumps({'ok':False,'error':f'{type(exc).__name__}: {exc}'}));sys.exit(2)
