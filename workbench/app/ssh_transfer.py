"""Bounded, resumable SFTP result transport; GPU execution is never retried here."""
import concurrent.futures,hashlib,json
from pathlib import Path
from .storage import ProjectError

def download(client_factory,local,remote,identity):
    client=client_factory()
    try:
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(120);info=sftp.stat(remote);size=info.st_size
            if not 0<size<=128*1024**2:raise ProjectError(413,'SSH_RESULT_TOO_LARGE','远程结果包大小无效或超过128MiB，保留原任务。')
            if size<=8*1024**2:
                sftp.get(remote,str(local),prefetch=False);return
            signature=hashlib.sha256(json.dumps([identity,remote,size,info.st_mtime]).encode()).hexdigest()[:24]
    finally:client.close()
    local=Path(local);folder=local.parent/('.sftp-parts-'+signature);folder.mkdir(exist_ok=True);step=2*1024**2
    def part(offset):
        length=min(step,size-offset);destination=folder/(str(offset)+'.part')
        if destination.is_file() and destination.stat().st_size==length:return destination
        connection=client_factory()
        try:
            with connection.open_sftp() as sftp:
                sftp.get_channel().settimeout(120)
                with sftp.open(remote,'rb') as source,destination.with_suffix('.partial').open('wb') as output:
                    source.seek(offset);remaining=length
                    while remaining:
                        chunk=source.read(min(32768,remaining))
                        if not chunk:raise ProjectError(503,'SSH_TRANSFER_FAILED','结果包分段传输不完整，保留原任务和已回传分段。')
                        output.write(chunk);remaining-=len(chunk)
                destination.with_suffix('.partial').replace(destination)
                return destination
        finally:connection.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(part,range(0,size,step)))
    with local.open('wb') as output:
        for path in paths:
            with path.open('rb') as source:
                while chunk:=source.read(65536):output.write(chunk)
    # Only this download's derived temporary parts are removed; source/result directories remain.
    for path in paths:path.unlink()
    folder.rmdir()
