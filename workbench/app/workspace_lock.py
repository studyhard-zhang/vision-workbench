"""One HTTP service per workspace, across source and packaged launches."""
import os
from pathlib import Path

class WorkspaceLease:
    def __init__(self, database_path):
        root=Path(database_path).resolve().parent;root.mkdir(parents=True,exist_ok=True)
        self.file=(root/'.workbench-service.lock').open('a+b')
        try:
            if os.name=='nt':
                import msvcrt
                if self.file.seek(0,2)==0:self.file.write(b'0');self.file.flush()
                self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();raise OSError('该工作区已有服务在运行；请使用原服务，不要同时启动另一个写入进程。')
    def close(self):
        if not self.file.closed:self.file.close()
