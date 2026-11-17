"""Remote-only, explicitly authorized missing-wheel preparation. Never imports GPU libraries.

Default: resolve and save a reviewable plan. Apply requires its exact SHA256 and an
unchanged environment. Existing distributions are constrained and never replaced.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

BASELINE={'torch':'2.3.0+cu121','torchvision':'0.18.0+cu121','numpy':'1.26.4','pillow':'10.3.0'}
REQUESTED={'ultralytics':'8.3.241','opencv-python':'4.11.0.86','dill':'0.3.8'}


def normalize(name):return re.sub(r'[-_.]+','-',name).lower()


def inventory():
    found={}
    for dist in importlib.metadata.distributions():
        name=normalize(dist.metadata['Name']);version=dist.version
        if name in found and found[name]!=version:raise ValueError('Ambiguous installed distribution: '+name)
        found[name]=version
    return found


def encoded(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2).encode('utf-8')


def check_baseline(installed):
    if any(installed.get(k)!=v for k,v in BASELINE.items()):raise ValueError('Existing torch/torchvision/numpy/Pillow baseline differs; stop without installation')
    for name,version in installed.items():
        if not re.fullmatch(r'[a-z0-9-]+',name) or not re.fullmatch(r'[a-zA-Z0-9.+!_-]+',version):raise ValueError('Cannot safely constrain distribution: '+name)


def build_plan(report,installed):
    check_baseline(installed);wheels=[];seen=set()
    for item in report.get('install',[]):
        name=normalize(item['metadata']['name']);version=item['metadata']['version'];download=item['download_info']
        sha=download.get('archive_info',{}).get('hashes',{}).get('sha256','')
        if name in installed:raise ValueError('Resolver would replace an existing distribution: '+name)
        if name in seen or not re.fullmatch(r'[a-z0-9-]+',name) or not re.fullmatch(r'[a-zA-Z0-9.+!_-]+',version):raise ValueError('Invalid planned distribution')
        if not urlparse(download['url']).path.lower().endswith('.whl') or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('Only hashed binary wheels may be installed')
        seen.add(name);wheels.append({'name':name,'version':version,'sha256':sha})
    resolved={**installed,**{w['name']:w['version'] for w in wheels}}
    if any(resolved.get(k)!=v for k,v in REQUESTED.items()):raise ValueError('Requested dependency versions not resolved')
    return {'format_version':1,'installed_before':installed,'wheels':wheels,'requested':REQUESTED,'gpu_used':False}


def apply_requirements(raw,expected_sha,installed):
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Reviewed plan SHA256 mismatch')
    plan=json.loads(raw);check_baseline(installed)
    if plan['installed_before']!=installed:raise ValueError('Environment changed since resolution; create and review a new plan')
    lines=[]
    for w in plan['wheels']:
        if w['name'] in installed or not re.fullmatch(r'[a-z0-9-]+',w['name']) or not re.fullmatch(r'[a-zA-Z0-9.+!_-]+',w['version']) or not re.fullmatch('[0-9a-f]{64}',w['sha256']):raise ValueError('Unsafe install plan')
        lines.append(f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}")
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--apply-plan-sha256')
    args=parser.parse_args()
    if platform.system()!='Linux' or platform.python_version()!='3.12.3':raise ValueError('Run only on the reviewed Linux/Python3.12.3 compute instance')
    installed=inventory();check_baseline(installed)
    folder=args.directory.resolve();folder.mkdir(parents=True,exist_ok=True);path=folder/'missing-wheels-plan.json'
    if args.apply_plan_sha256:
        requirements=apply_requirements(path.read_bytes(),args.apply_plan_sha256,installed)
        required=folder/'reviewed-wheels.txt';required.write_text(requirements,encoding='utf-8')
        if requirements.strip():subprocess.run([sys.executable,'-m','pip','install','--no-deps','--require-hashes','--only-binary=:all:','-r',str(required)],check=True)
        after=inventory()
        if any(after.get(k)!=v for k,v in installed.items()):raise RuntimeError('Existing distribution changed unexpectedly; preserve the environment and logs')
        if any(after.get(k)!=v for k,v in REQUESTED.items()):raise RuntimeError('Preparation incomplete; no GPU operation authorized by this script')
        (folder/'prepared.json').write_bytes(encoded({'installed_after':after,'existing_distributions_preserved':True,'gpu_used':False}))
        print('Missing wheels installed; existing versions preserved. GPU validation has not run.')
    else:
        constraints=folder/'existing-constraints.txt';constraints.write_text('\n'.join(k+'=='+v for k,v in sorted(installed.items()))+'\n',encoding='utf-8')
        report=folder/'pip-dry-run.json'
        subprocess.run([sys.executable,'-m','pip','install','--dry-run','--only-binary=:all:','--report',str(report),'-c',str(constraints),*[k+'=='+v for k,v in REQUESTED.items()]],check=True)
        raw=encoded(build_plan(json.loads(report.read_text(encoding='utf-8')),installed));path.write_bytes(raw)
        print('Review missing-wheels-plan.json; apply only after checking the wheel list. SHA256='+hashlib.sha256(raw).hexdigest())
    return 0


if __name__=='__main__':raise SystemExit(main())
