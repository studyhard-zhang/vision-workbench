"""Standard-library integrity check for an extracted release, without loading models."""
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest=json.loads((root/'release-manifest.json').read_text(encoding='utf-8'))
    for name,spec in manifest['files'].items():
        path=root/name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():raise ValueError('Missing/unsafe file: '+name)
        raw=path.read_bytes()
        if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Checksum mismatch: '+name)
    return manifest


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    m=verify(root)
    print(f"Release {m['version']} / schema{m['schema']}: {len(m['files'])} files verified; no model loaded.")
