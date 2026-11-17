"""Build a source-only release archive, excluding workspace data and credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.version import VERSION,SCHEMA


def build(destination):
    root=Path(__file__).resolve().parents[1]
    files={}
    for folder,extensions in [('app',{'.py'}),('web',{'.html','.css','.js','.mjs','.svg'}),('tools',{'.py','.json'})]:
        for p in (root/folder).rglob('*'):
            if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts and not p.is_symlink():
                files[p.relative_to(root).as_posix()]=p.read_bytes()
    for name in ('run.py','requirements.txt','RELEASE.md'):
        files[name]=(root/name).read_bytes()
    files['NOTICE.md']=(root.parent/'NOTICE.md').read_bytes()
    checks={name:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} for name,raw in files.items()}
    files['release-manifest.json']=(json.dumps({'format_version':1,'version':VERSION,'schema':SCHEMA,
        'files':checks,'excluded':['var','restored-workspaces','.runtime','.vendor','SSH configuration/keys','business images/weights']},indent=2)+'\n').encode()
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(files.items()):z.writestr(name,raw)
    return {'path':str(destination),'files':len(files),'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output),indent=2))
