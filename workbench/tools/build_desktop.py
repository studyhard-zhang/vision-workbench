"""Reproducible desktop/installer build; never copies workspace data or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.version import VERSION,SCHEMA

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--iscc',type=Path,required=True);parser.add_argument('--skip-freeze',action='store_true');args=parser.parse_args()
    if not args.skip_freeze:
        subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--clean','--distpath',str(ROOT/'.runtime/desktop-dist'),'--workpath',str(ROOT/'.runtime/desktop-pyinstaller'),str(ROOT/'desktop/VisionWorkbench.spec')],cwd=ROOT,check=True)
    bundle=ROOT/'.runtime/desktop-dist/VisionWorkbench'
    if not (bundle/'VisionWorkbench.exe').is_file():raise RuntimeError('Desktop executable is missing')
    forbidden=[p for p in bundle.rglob('*') if p.is_file() and (p.suffix.lower() in {'.db','.sqlite','.sqlite3','.pt','.onnx','.engine','.pem','.key','.dpapi'} or any(part in {'var','tests','.runtime','compute-evidence'} for part in p.relative_to(bundle).parts) or p.name in {'ssh-target.json','ssh-candidate.json','active-compute.json','compute-target.json','segment-training.json','delivery_reference_validation.json','id_ed25519','id_rsa'})]
    if forbidden:raise RuntimeError('Workspace data or credentials found in desktop bundle')
    subprocess.run([str(args.iscc),'/DAppVersion='+VERSION,'/DSourceDir='+str(bundle),'/DOutputDir='+str(ROOT/'deliverables'),str(ROOT/'desktop/installer.iss')],check=True,cwd=ROOT)
    setup=ROOT/f'deliverables/VisionWorkbench-Setup-{VERSION}-x64.exe'
    proof={'version':VERSION,'schema':SCHEMA,'setup':setup.name,'bytes':setup.stat().st_size,'sha256':hashlib.file_digest(setup.open('rb'),'sha256').hexdigest(),'signed':False,'business_data_included':False,'files':{p.relative_to(bundle).as_posix():hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in bundle.rglob('*') if p.is_file()}}
    path=setup.with_suffix('.manifest.json');path.write_text(json.dumps(proof,ensure_ascii=False,indent=2),'utf-8');print(setup)

if __name__=='__main__':main()
