"""Windows desktop entry point. Runs the existing service and UI, never a second project."""
import argparse
import configparser
import ctypes
import errno
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import threading
import urllib.error
import urllib.request

from app.version import VERSION,SCHEMA

def profile_root():
    return Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'VisionWorkbench'

def workspace_from_profile(profile):
    config=configparser.ConfigParser(interpolation=None)
    config.read(profile/'desktop.ini',encoding='utf-8-sig')
    value=config.get('workspace','directory',fallback='').strip()
    if not value:raise ValueError('尚未选择工作区。请运行安装向导，创建新工作区或选择已有工作区。')
    return Path(value)

def initialize_workspace(value):
    """Create an explicitly selected empty workspace; never seed demo projects."""
    path=Path(value).expanduser().resolve()
    if getattr(sys,'frozen',False) and path.is_relative_to(Path(sys.executable).parent.resolve()):
        raise ValueError('工作区必须位于程序安装目录以外。')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('新工作区须为空目录。已有数据请选择“使用已有工作区”，不会覆盖或清空。')
    path.mkdir(parents=True,exist_ok=True)
    database=path/'platform.db'
    with database.open('xb'):pass
    from app.storage import Storage
    Storage(database)
    return validate_workspace(path)

def validate_workspace(value):
    path=Path(value).expanduser().resolve()
    if not (path/'platform.db').is_file():raise ValueError('所选目录没有platform.db。请在安装向导中选择原var目录；不会新建空库或替换真实数据。')
    if getattr(sys,'frozen',False) and path.is_relative_to(Path(sys.executable).parent.resolve()):raise ValueError('工作区必须位于程序安装目录以外，以便升级和卸载时保留数据。')
    db=sqlite3.connect((path/'platform.db').as_uri()+'?mode=ro',uri=True)
    try:version=db.execute('SELECT version FROM schema_version').fetchone()[0]
    finally:db.close()
    if version not in (15,SCHEMA):raise ValueError(f'当前客户端需要schema{SCHEMA}，所选工作区为schema{version}。请先使用对应版本完成备份/升级。')
    return path

def runtime_info(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/runtime',timeout=3) as response:
            return json.loads(response.read(16384))
    except (OSError,ValueError):return None

class Session:
    def __init__(self,workspace,port):self.workspace=Path(workspace).resolve();self.port=port;self.server=None;self.thread=None;self.reused=False
    def start(self):
        info=runtime_info(self.port)
        matching=isinstance(info,dict) and info.get('product')=='vision-workbench' and isinstance(info.get('data_dir'),str) and Path(info['data_dir']).resolve()==self.workspace
        if matching:
            if info.get('version')!=VERSION or info.get('schema')!=SCHEMA:
                raise ValueError(f'此工作区已由其他版本的工作台服务打开，请先退出该工作区的旧版服务再启动客户端。\n工作区：{self.workspace}')
            self.reused=True
        else:
            from app.server import create_server
            preferred=self.port
            try:
                try:self.server=create_server(0 if info else preferred,self.workspace/'platform.db')
                except OSError as exc:
                    # Binding may race another process, or Windows may reserve the
                    # preferred port. Workspace-lock errors must never start a second writer.
                    if info or (exc.errno not in (errno.EADDRINUSE,errno.EACCES) and getattr(exc,'winerror',None) not in (10048,10013)):raise
                    self.server=create_server(0,self.workspace/'platform.db')
            except OSError as exc:
                raise ValueError(f'无法打开所选工作区，请检查是否已有服务使用该目录，以及目录权限。\n工作区：{self.workspace}') from exc
            self.port=self.server.server_port
            if self.port!=preferred:logging.info('Preferred desktop port %s unavailable; selected %s for workspace=%s',preferred,self.port,self.workspace)
            self.thread=threading.Thread(target=self.server.serve_forever,name='desktop-workbench',daemon=True);self.thread.start()
        return f'http://127.0.0.1:{self.port}/'
    def close(self):
        if self.server:
            self.server.shutdown();self.server.server_close();self.thread.join(timeout=5);self.server=None

class DesktopInstance:
    def __init__(self,workspace):
        self.handle=None
        if os.name=='nt':
            kernel=ctypes.windll.kernel32;kernel.CreateMutexW.restype=ctypes.c_void_p
            self.handle=kernel.CreateMutexW(None,False,'Local\\VisionWorkbenchDesktop-'+hashlib.sha256(str(workspace).casefold().encode()).hexdigest()[:24])
            if not self.handle:raise OSError('无法创建客户端实例标识。')
            if kernel.GetLastError()==183:
                self.close();raise ValueError('该工作区的桌面客户端已经打开，请使用已有窗口。')
    def close(self):
        if self.handle:
            ctypes.windll.kernel32.CloseHandle.argtypes=[ctypes.c_void_p];ctypes.windll.kernel32.CloseHandle(self.handle);self.handle=None

def notify(message):
    if os.name=='nt':ctypes.windll.user32.MessageBoxW(None,str(message),'视训工作台',0x10)
    else:print(message,file=sys.stderr)

def main():
    parser=argparse.ArgumentParser(description='视训工作台桌面客户端')
    parser.add_argument('--data-dir',type=Path);parser.add_argument('--profile-dir',type=Path,default=profile_root());parser.add_argument('--port',type=int,default=8765,help='首选本地端口；被其他工作区或程序占用时自动选择空闲端口')
    parser.add_argument('--smoke-report',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--initialize-workspace',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();profile=args.profile_dir.resolve();profile.mkdir(parents=True,exist_ok=True)
    if getattr(sys,'frozen',False):
        # A windowed EXE has no valid console streams. HTTP and runtime diagnostics
        # must not fail while responding simply because stderr is unavailable.
        sys.stdout=(profile/'console.log').open('a',encoding='utf-8',buffering=1)
        sys.stderr=sys.stdout
    logging.basicConfig(filename=profile/'desktop.log',encoding='utf-8',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    session=None;instance=None
    try:
        if not 1024<=args.port<=65535:raise ValueError('本地端口须为1024至65535。')
        if args.initialize_workspace:
            if args.data_dir is None:raise ValueError('创建工作区必须明确指定目录。')
            initialize_workspace(args.data_dir)
            return 0
        workspace=validate_workspace(args.data_dir or workspace_from_profile(profile));instance=DesktopInstance(workspace)
        session=Session(workspace,args.port);url=session.start()
        import webview
        webview.settings['ALLOW_DOWNLOADS']=True
        webview.settings['ALLOW_FILE_URLS']=False
        webview.settings['OPEN_EXTERNAL_LINKS_IN_BROWSER']=True
        window=webview.create_window('视训工作台 · 桌面版 '+VERSION,url,width=1440,height=980,min_size=(1000,700),confirm_close=not bool(args.smoke_report),text_select=True)
        if args.smoke_report:
            def loaded():
                info=runtime_info(session.port)
                if not info or info.get('version')!=VERSION:raise RuntimeError('Desktop page loaded without matching workspace service')
                args.smoke_report.parent.mkdir(parents=True,exist_ok=True)
                args.smoke_report.write_text(json.dumps({'version':VERSION,'workspace':str(workspace),'runtime':info,'reused_service':session.reused,'window_loaded':True,'url':window.get_current_url()},ensure_ascii=False,indent=2),encoding='utf-8')
                threading.Timer(2,window.destroy).start()
            window.events.loaded+=loaded
        logging.info('Desktop opened; workspace=%s port=%s reused=%s',workspace,session.port,session.reused)
        webview.start(gui='edgechromium',private_mode=False,storage_path=str(profile/'webview'),localization={'global.quitConfirmation':'退出客户端？请先保存当前编辑。关闭窗口不会中止远端GPU任务；下次打开可继续核对进度。','global.ok':'退出','global.cancel':'返回工作台','global.saveFile':'保存文件','global.openFile':'选择文件','global.allFiles':'所有文件'})
        return 0
    except Exception as exc:
        logging.exception('Desktop launch failed')
        if args.smoke_report:print(str(exc),file=sys.stderr)
        else:notify(f'{exc}\n\n日志：{profile / "desktop.log"}')
        return 1
    finally:
        if session:session.close()
        if instance:instance.close()

if __name__=='__main__':raise SystemExit(main())
