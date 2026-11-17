"""Keep the explicitly configured local executor awake; never launch a GPU job."""
import atexit
import logging
import os
import socket
import subprocess
import threading
import time


def is_local(profile):
    return (os.name == 'nt' and profile.get('local_wsl') == 'Ubuntu-24.04'
            and profile.get('host') == '127.0.0.1' and profile.get('port') == 22226
            and profile.get('user') == 'root')


class LocalWSL:
    def __init__(self):
        self.process = None
        self.guard = threading.Lock()
        self.stopped = threading.Event()
        self.worker = None

    def ensure(self, profile):
        if not is_local(profile):
            return
        from .storage import ProjectError
        with self.guard:
            if self.process is None or self.process.poll() is not None:
                # An open stdin keeps a foreground Linux process alive. Closing it
                # releases only our keeper, without stopping SSH or training jobs.
                try:
                    self.process = subprocess.Popen(
                        ['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'bash', '-lc',
                         'systemctl start vision-workbench-ssh.service && exec cat >/dev/null'],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except OSError as error:
                    raise ProjectError(503, 'LOCAL_EXECUTOR_UNAVAILABLE',
                                       '无法启动本机 WSL 训练环境，请检查 Ubuntu-24.04 安装；原任务保持。') from error
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    break
                try:
                    with socket.create_connection(('127.0.0.1', 22226), timeout=1) as sock:
                        sock.settimeout(1)
                        if sock.recv(256).startswith(b'SSH-'):
                            return
                except OSError:
                    pass
                time.sleep(.25)
            raise ProjectError(503, 'LOCAL_EXECUTOR_UNAVAILABLE',
                               '本机训练环境未就绪，请检查 Ubuntu-24.04 和本机训练服务；原任务保持，不会自动重跑。')

    def start(self, profile):
        if not is_local(profile) or self.worker and self.worker.is_alive():
            return
        self.stopped.clear()

        def watch():
            while not self.stopped.is_set():
                try:
                    self.ensure(profile)
                except Exception as error:
                    logging.warning('Local WSL executor is not ready: %s', error)
                if self.stopped.wait(15):
                    return
        self.worker = threading.Thread(target=watch, name='local-wsl-keeper', daemon=True)
        self.worker.start()

    def close(self):
        self.stopped.set()
        with self.guard:
            if self.process and self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()


manager = LocalWSL()
atexit.register(manager.close)
