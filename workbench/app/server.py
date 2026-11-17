from __future__ import annotations

import json
import logging
import re
import sqlite3
import shutil
import uuid
import time
import os
import socket
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .version import VERSION, SCHEMA
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, unquote, quote

from .storage import ProjectError, Storage
from .images import ImageRepository, MAX_FILE_BYTES
from .tasks import TaskRepository
from .annotations import AnnotationRepository
from .preparation import PreparationRepository
from .versions import VersionRepository
from .training import TrainingRepository
from .executions import ExecutionRepository,MAX_RESULT_BYTES
from .remote import RemoteRepository
from .models import ModelRepository
from .delivery import DeliveryRepository
from .maintenance import MaintenanceRepository, MAX_ARCHIVE
from .training_reports import TrainingReportRepository
from .augmentation import AugmentationRepository
from .evaluations import EvaluationRepository,EvaluationExecution
from .inferences import InferenceInputRepository,InferenceRepository,InferenceExecution

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
STATIC_FILES = {
    '/editor-view.mjs': ('editor-view.mjs','text/javascript; charset=utf-8'),
    '/ssh-connection.js': ('ssh-connection.js','text/javascript; charset=utf-8'),
    '/roi-editor.js': ('roi-editor.js','text/javascript; charset=utf-8'),
    "/inference-studio.js": ("inference-studio.js", "text/javascript; charset=utf-8"),
    "/inference-studio.css": ("inference-studio.css", "text/css; charset=utf-8"),
    "/task-navigation.js": ("task-navigation.js", "text/javascript; charset=utf-8"),
    "/maintenance.js": ("maintenance.js", "text/javascript; charset=utf-8"),
    "/inferences.js": ("inferences.js", "text/javascript; charset=utf-8"),
    "/evaluations.js": ("evaluations.js", "text/javascript; charset=utf-8"),
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/datasets.css": ("datasets.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/api.js": ("api.js", "text/javascript; charset=utf-8"),
    "/datasets.js": ("datasets.js", "text/javascript; charset=utf-8"),
    "/images.js": ("images.js", "text/javascript; charset=utf-8"),
    "/images.css": ("images.css", "text/css; charset=utf-8"),
    "/tasks.js": ("tasks.js", "text/javascript; charset=utf-8"),
    "/tasks.css": ("tasks.css", "text/css; charset=utf-8"),
    "/annotations.js": ("annotations.js", "text/javascript; charset=utf-8"),
    "/annotations.css": ("annotations.css", "text/css; charset=utf-8"),
    "/box-editor.js": ("box-editor.js", "text/javascript; charset=utf-8"),
    "/editor-flow.mjs": ("editor-flow.mjs", "text/javascript; charset=utf-8"),
    "/box-model.mjs": ("box-model.mjs", "text/javascript; charset=utf-8"),
    "/polygon-model.mjs": ("polygon-model.mjs", "text/javascript; charset=utf-8"),
    "/box-editor.css": ("box-editor.css", "text/css; charset=utf-8"),
    "/preparation.js": ("preparation.js", "text/javascript; charset=utf-8"),
    "/preparation.css": ("preparation.css", "text/css; charset=utf-8"),
    "/versions.js": ("versions.js", "text/javascript; charset=utf-8"),
    "/versions.css": ("versions.css", "text/css; charset=utf-8"),
    "/training.js": ("training.js", "text/javascript; charset=utf-8"),
    "/training.css": ("training.css", "text/css; charset=utf-8"),
    "/models.js": ("models.js", "text/javascript; charset=utf-8"),
    "/training-report.js": ("training-report.js", "text/javascript; charset=utf-8"),
    "/training-chart.mjs": ("training-chart.mjs", "text/javascript; charset=utf-8"),
    "/training-config.js": ("training-config.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
INFERENCE_INPUT_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/inference-inputs(?:/([0-9a-f]{32})/(view|thumbnail|original))?$")
INFERENCE_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/inferences(?:/(options|[0-9a-f]{32})(?:/(cancel|remote-run|remote-stop|remote-pause|report|export|artifacts/(?:report\.json|console\.log|deployment\.zip)))?)?$")
INFERENCE_IMAGE_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/inferences/([0-9a-f]{32})/images/([0-9a-f]{32})/(view|overlay)$")
EVALUATION_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/evaluations(?:/(options|[0-9a-f]{32})(?:/(cancel|remote-run|remote-stop|remote-pause|report|artifacts/(?:report\.json|console\.log|[A-Za-z0-9_]+\.png)))?)?$")
PROJECT_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})$")
DATASETS_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/datasets$")
DATASET_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/datasets/([0-9a-f]{32})$")
IMAGES_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/datasets/([0-9a-f]{32})/images$")
IMAGE_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/datasets/([0-9a-f]{32})/images/([0-9a-f]{32})(?:/(thumbnail|view|original))?$")
TASKS_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks$")
TASK_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})(?:/(images|sync-images))?$")
CLASSES_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/classes(?:/([0-9a-f]{32}))?$")
ANNOTATION_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/images/([0-9a-f]{32})/annotation$")
PREPARATION_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/(preparation|split)$")
VERSIONS_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/versions$")
TRAINING_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/training(?:/([0-9a-f]{32})(?:/(cancel|snapshot|dispatch|package|result|remote-run|remote-stop|remote-pause|remote-abandon|remote-stop-and-trash|register-model|report|restore))?)?$")
TRAINING_VISIBILITY_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/training/(trash|restore)$")
MODEL_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/models(?:/([0-9a-f]{32})(?:/artifacts/(best\.pt|last\.pt|args\.yaml|results\.csv|console\.log|manifest\.json))?)?$")
DELIVERY_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/models/([0-9a-f]{32})/delivery$")
TRAINING_ARTIFACT_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/training/([0-9a-f]{32})/artifacts/(console\.log|best\.pt|last\.pt|args\.yaml|results\.csv)$")
VERSION_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/versions/([0-9a-f]{32})(?:/(archive|readiness|images/([0-9a-f]{32})))?$")
PREVIEW_PATH = re.compile(r"^/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/training-preview$")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != 'nt'

    def server_bind(self):
        if os.name=='nt' and hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

    def __init__(self, address, database_path):
        self.storage = Storage(database_path)
        self.mutation_lock = threading.Lock()
        self.maintenance = MaintenanceRepository(self.storage)
        self.images = ImageRepository(self.storage)
        self.tasks = TaskRepository(self.storage)
        self.annotations = AnnotationRepository(self.storage)
        self.preparation = PreparationRepository(self.storage)
        self.versions = VersionRepository(self.storage)
        self.training = TrainingRepository(self.storage,self.versions)
        self.executions = ExecutionRepository(self.storage,self.training)
        self.models = ModelRepository(self.storage,self.executions)
        self.executions.on_success=self.models.automatic
        self.models.sync_successful()
        self.training_reports = TrainingReportRepository(self.executions)
        self.augmentation = AugmentationRepository(self.versions)
        self.remote = RemoteRepository(self.storage,self.training,self.executions)
        from .compute_instances import ComputeInstances
        self.compute_instances = ComputeInstances(self.remote)
        from .local_wsl import manager
        try: manager.start(self.remote.transport().profile)
        except ProjectError: pass
        self.evaluations = EvaluationRepository(self.storage,self.models)
        self.evaluation_executions = EvaluationExecution(self.storage,self.evaluations)
        self.delivery = DeliveryRepository(self.models,self.evaluation_executions)
        self.evaluation_remote = RemoteRepository(self.storage,self.evaluations,self.evaluation_executions,kind='evaluation')
        self.inference_inputs = InferenceInputRepository(self.storage,self.tasks)
        self.inferences = InferenceRepository(self.storage,self.models,self.inference_inputs)
        self.inference_executions = InferenceExecution(self.storage,self.inferences)
        self.inference_remote = RemoteRepository(self.storage,self.inferences,self.inference_executions,kind='inference')
        super().__init__(address, Handler)
        if self.remote.info()['configured']:
            self.remote.start_worker()
            self.evaluation_remote.start_worker()
            self.inference_remote.start_worker()

    def server_close(self):
        from .local_wsl import manager
        manager.close()
        self.remote.close()
        self.evaluation_remote.close()
        self.inference_remote.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logging.info('%s - %s',self.address_string(),format % args)
    server_version = 'VisionWorkspace/'+VERSION
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def send_content(self, status, body, content_type, download_name=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if download_name:self.send_header('Content-Disposition',f'attachment; filename="{download_name}"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, status, data):
        self.send_content(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def read_body(self, limit=32768):
        if self.headers.get_content_type() != "application/json":
            raise ProjectError(415, "JSON_REQUIRED", "请求内容必须为 JSON。")
        if self.headers.get("Transfer-Encoding"):
            raise ProjectError(400, "INVALID_BODY", "不支持此请求传输方式。")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ProjectError(400, "INVALID_BODY", "请求长度无效。") from exc
        if not 0 < size <= limit:
            raise ProjectError(413, "BODY_LIMIT", "请求内容为空或超过大小限制。")
        try:
            raw = self.rfile.read(size)
            self._body_bytes_read += len(raw)
            if len(raw) != size:
                raise ValueError("Incomplete body")
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise ProjectError(400, "INVALID_JSON", "请求内容不是有效的 JSON。") from exc
        if not isinstance(value, dict):
            raise ProjectError(400, "INVALID_JSON", "请求内容必须为对象。")
        return value

    def read_training_result(self):
        if self.headers.get_content_type()!='application/zip' or self.headers.get('Transfer-Encoding'):
            raise ProjectError(415,'ZIP_REQUIRED','请上传实例生成的return-result.zip。')
        try:length=int(self.headers.get('Content-Length','0'))
        except ValueError as exc:raise ProjectError(400,'INVALID_BODY','结果包长度无效。') from exc
        if not 0<length<=MAX_RESULT_BYTES:raise ProjectError(413,'RESULT_TOO_LARGE','结果ZIP不能为空，最多128MiB。')
        raw=self.rfile.read(length);self._body_bytes_read+=len(raw)
        if len(raw)!=length:raise ProjectError(400,'INCOMPLETE_RESULT','结果包传输不完整，请重新上传。')
        return raw

    def send_training_file(self,value):
        path,name,mime=value
        with path.open('rb') as stream:
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(path.stat().st_size))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Disposition',f'attachment; filename="{name}"');self.end_headers()
            if self.command!='HEAD':shutil.copyfileobj(stream,self.wfile)

    def check_access(self):
        host = self.headers.get("Host", "")
        if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
            raise ProjectError(403, "HOST_NOT_ALLOWED", "请通过本机地址访问工作台。")
        if self.command in {"POST", "PUT", "DELETE"}:
            origin = self.headers.get("Origin")
            if origin and origin not in {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }:
                raise ProjectError(403, "ORIGIN_NOT_ALLOWED", "不允许来自其他网站的修改请求。")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise ProjectError(403, "ORIGIN_NOT_ALLOWED", "不允许来自其他网站的修改请求。")

    def read_image_body(self):
        if self.headers.get('Transfer-Encoding'):
            raise ProjectError(400,'INVALID_BODY','不支持此请求传输方式。')
        try:
            length=int(self.headers.get('Content-Length','0'))
        except ValueError as exc:
            raise ProjectError(400,'INVALID_BODY','文件长度无效。') from exc
        if not 0 < length <= MAX_FILE_BYTES:
            raise ProjectError(413,'IMAGE_SIZE_LIMIT','图片不能为空，单张最多 20MB。')
        try:
            filename=unquote(self.headers.get('X-File-Name',''),encoding='utf-8',errors='strict')
            group=unquote(self.headers.get('X-Group-Id',''),encoding='utf-8',errors='strict')
        except UnicodeError as exc:
            raise ProjectError(422,'INVALID_FILENAME','文件名或采集组编码无效。') from exc
        raw=self.rfile.read(length)
        self._body_bytes_read += len(raw)
        if len(raw)!=length:
            raise ProjectError(400,'INCOMPLETE_IMAGE','图片传输不完整，请重新导入。')
        return filename,raw,group

    def send_image_file(self,project_id,dataset_id,image_id,kind):
        path,mime,record=self.server.images.file(project_id,dataset_id,image_id,kind)
        with path.open('rb') as stream:
            size=path.stat().st_size
            self.send_response(200)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(size))
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','private, max-age=3600' if kind!='original' else 'no-store')
            if kind=='original':
                self.send_header('Content-Disposition',"attachment; filename*=UTF-8''"+quote(record['filename'],safe=''))
            self.end_headers()
            if self.command!='HEAD': shutil.copyfileobj(stream,self.wfile)

    def dispatch(self):
        self.request_id = uuid.uuid4().hex[:12]
        self._body_bytes_read = 0
        mutation_locked = False
        try:
            self.check_access()
            if self.command in ('POST','PUT','DELETE'):
                mutation_locked=self.server.mutation_lock.acquire(blocking=False)
                if not mutation_locked:raise ProjectError(409,'WORKSPACE_BUSY','工作区正在保存、备份或恢复，请稍后重试；当前输入保留。')
            path = urlsplit(self.path).path
            if self.command in {"GET", "HEAD"}:
                if path=='/api/maintenance/release':
                    package=self.server.storage.path.parent/'releases/vision-workbench-0.20.0.zip'
                    if not package.is_file():raise ProjectError(404,'RELEASE_NOT_READY','发布包尚未生成。')
                    return self.send_training_file((package,package.name,'application/zip'))
                if path=='/api/maintenance':return self.send_json(200,{'usage':self.server.maintenance.usage(),'backups':self.server.maintenance.list(),'restores':self.server.maintenance.restored_list()})
                match=re.fullmatch(r'/api/maintenance/backups/([0-9a-f]{32})',path)
                if match:return self.send_training_file(self.server.maintenance.archive(match[1]))
                match=re.fullmatch(r'/api/projects/([0-9a-f]{32})/delete-impact',path)
                if match:return self.send_json(200,self.server.maintenance.impact(match[1]))
                if path == "/api/health":
                    self.server.storage.list_projects()
                    return self.send_json(200, {"status": "ok", "version": VERSION, "node": "N-31", "schema": SCHEMA})
                if path=='/api/runtime':
                    return self.send_json(200,{'product':'vision-workbench','version':VERSION,'schema':SCHEMA,'data_dir':str(self.server.storage.path.parent.resolve())})
                match=INFERENCE_INPUT_PATH.fullmatch(path)
                if match:
                    p,t,i,kind=match.groups()
                    if i:
                        file,mime,_=self.server.inference_inputs.file(p,t,i,kind)
                        return self.send_training_file((file,f'infer-input-{i[:8]}-{kind}'+file.suffix,mime))
                    inputs=self.server.inference_inputs.list_inputs(p,t)
                    return self.send_json(200,{'inputs':inputs,'total':len(inputs)})
                match=INFERENCE_IMAGE_PATH.fullmatch(path)
                if match:
                    p,t,j,i,kind=match.groups();job=self.server.inferences.get(p,t,j)
                    if i not in {item['id'] for item in job['snapshot']['inputs']}:raise ProjectError(404,'INFERENCE_INPUT_NOT_FOUND','图片不属于当前推理。')
                    if kind=='view':
                        file,mime,_=self.server.inference_inputs.file(p,t,i,'view')
                        return self.send_training_file((file,f'infer-input-{i[:8]}'+file.suffix,mime))
                    conf=parse_qs(urlsplit(self.path).query).get('conf',[max(0.25,job['snapshot']['requested_parameters']['conf'])])[0]
                    return self.send_content(200,self.server.inference_executions.overlay(p,t,j,i,conf),'image/png',f'infer-{j[:8]}-{i[:8]}.png')
                match=INFERENCE_PATH.fullmatch(path)
                if match:
                    p,t,j,action=match.groups()
                    if j=='options':return self.send_json(200,self.server.inferences.options(p,t,parse_qs(urlsplit(self.path).query).get('model_id',[''])[0]))
                    if action=='report':return self.send_json(200,self.server.inference_executions.report(p,t,j))
                    if action=='export':
                        job=self.server.inferences.get(p,t,j);conf=parse_qs(urlsplit(self.path).query).get('conf',[max(0.25,job['snapshot']['requested_parameters']['conf'])])[0]
                        data=self.server.inference_executions.export(p,t,j,conf)
                        return self.send_content(200,(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'),'application/json',f'infer-{j[:8]}-filtered.json')
                    if action and action.startswith('artifacts/'):return self.send_training_file(self.server.inference_executions.artifact(p,t,j,action.split('/')[1]))
                    if j and not action:return self.send_json(200,{'job':self.server.inferences.get(p,t,j)})
                    if not j:
                        jobs=self.server.inferences.list(p,t)
                        return self.send_json(200,{'jobs':jobs,'total':len(jobs)})
                    raise ProjectError(404,'NOT_FOUND','推理接口不存在。')
                match=EVALUATION_PATH.fullmatch(path)
                if match:
                    p,t,j,action=match.groups()
                    if j=='options':return self.send_json(200,self.server.evaluations.options(p,t,parse_qs(urlsplit(self.path).query).get('model_id',[''])[0]))
                    if action=='report':return self.send_json(200,self.server.evaluation_executions.report(p,t,j))
                    if action and action.startswith('artifacts/'):return self.send_training_file(self.server.evaluation_executions.artifact(p,t,j,action.split('/')[1]))
                    if j and not action:return self.send_json(200,{'job':self.server.evaluations.get(p,t,j)})
                    if not j:
                        jobs=self.server.evaluations.list(p,t)
                        return self.send_json(200,{'jobs':jobs,'total':len(jobs)})
                    raise ProjectError(404,'NOT_FOUND','评估接口不存在。')
                match=DELIVERY_PATH.fullmatch(path)
                if match:
                    raw,name=self.server.delivery.build(*match.groups())
                    return self.send_content(200,raw,'application/zip',name)
                match=MODEL_PATH.fullmatch(path)
                if match:
                    if match[4]:return self.send_training_file(self.server.models.artifact(match[1],match[2],match[3],match[4]))
                    if match[3]:return self.send_json(200,{'model':self.server.models.get(match[1],match[2],match[3])})
                    models=self.server.models.list(match[1],match[2])
                    return self.send_json(200,{'models':models,'total':len(models)})
                if path=='/api/training-options':
                    query=parse_qs(urlsplit(self.path).query)
                    return self.send_json(200,self.server.training.options(query.get('project_id',[None])[0],query.get('task_id',[None])[0],query.get('version_id',[None])[0],query.get('training_mode',['standard'])[0]))
                if path=='/api/remote':return self.send_json(200,self.server.remote.info())
                if path=='/api/remote/instances':return self.send_json(200,self.server.compute_instances.status())
                match=TRAINING_ARTIFACT_PATH.fullmatch(path)
                if match:return self.send_training_file(self.server.executions.artifact(match[1],match[2],match[3],match[4]))
                match=TRAINING_PATH.fullmatch(path)
                if match:
                    if match[3]:
                        if match[4]=='report':return self.send_json(200,self.server.training_reports.get(match[1],match[2],match[3]))
                        if match[4]=='package':return self.send_training_file(self.server.executions.package(match[1],match[2],match[3]))
                        job=self.server.training.get(match[1],match[2],match[3])
                        if match[4]=='snapshot':
                            return self.send_content(200,(json.dumps({'name':job['name'],'snapshot_sha256':job['snapshot_sha256'],'snapshot':job['snapshot']},ensure_ascii=False,indent=2)+'\n').encode('utf-8'),'application/json; charset=utf-8',f'training-{job["id"][:8]}.json')
                        if match[4]:raise ProjectError(404,'NOT_FOUND','接口不存在。')
                        return self.send_json(200,{'job':job})
                    query=parse_qs(urlsplit(self.path).query)
                    jobs=self.server.training.list(match[1],match[2],query.get('view',['active'])[0])
                    return self.send_json(200,{'jobs':jobs,'total':len(jobs),'executor_connected':False})
                if path=='/tools/verify-dataset.py':
                    return self.send_content(200,(WEB_ROOT.parent/'tools'/'verify_dataset.py').read_bytes(),'application/octet-stream','verify_dataset.py')
                if path == "/api/projects":
                    projects = self.server.storage.list_projects()
                    return self.send_json(200, {"projects": projects, "total": len(projects)})
                match=VERSIONS_PATH.fullmatch(path)
                if match:
                    versions=self.server.versions.list(match[1],match[2])
                    return self.send_json(200,{'versions':versions,'total':len(versions)})
                match=VERSION_PATH.fullmatch(path)
                if match:
                    if match[4]=='readiness':
                        return self.send_json(200,self.server.versions.readiness(match[1],match[2],match[3]))
                    if match[4]=='archive':
                        with self.server.versions.archive(match[1],match[2],match[3]) as (stream,size,name):
                            self.send_response(200);self.send_header('Content-Type','application/zip')
                            self.send_header('Content-Length',str(size));self.send_header('Cache-Control','no-store')
                            self.send_header('X-Content-Type-Options','nosniff')
                            self.send_header('Content-Disposition',f'attachment; filename="{name}"');self.end_headers()
                            if self.command!='HEAD':shutil.copyfileobj(stream,self.wfile)
                        return
                    if match[5]:
                        raw,mime=self.server.versions.image(match[1],match[2],match[3],match[5])
                        return self.send_content(200,raw,mime)
                    return self.send_json(200,self.server.versions.read(match[1],match[2],match[3]))
                match=TASKS_PATH.fullmatch(path)
                if match:
                    tasks=self.server.tasks.list(match[1])
                    return self.send_json(200,{'tasks':tasks,'total':len(tasks)})
                match=PREPARATION_PATH.fullmatch(path)
                if match and match[3]=='preparation':
                    return self.send_json(200,self.server.preparation.read(match[1],match[2]))
                match=CLASSES_PATH.fullmatch(path)
                if match and not match[3]:
                    return self.send_json(200,{'classes':self.server.annotations.classes(match[1],match[2])})
                match=ANNOTATION_PATH.fullmatch(path)
                if match:
                    return self.send_json(200,{'annotation':self.server.annotations.get(match[1],match[2],match[3])})
                match=TASK_PATH.fullmatch(path)
                if match:
                    if not match[3]:return self.send_json(200,{'task':self.server.tasks.get(match[1],match[2])})
                    if match[3]=='images':
                        query=parse_qs(urlsplit(self.path).query)
                        try:page=int(query.get('page',['1'])[0]);size=int(query.get('page_size',['24'])[0])
                        except ValueError as exc:raise ProjectError(422,'INVALID_QUERY','分页参数无效。') from exc
                        return self.send_json(200,self.server.tasks.images(match[1],match[2],page,size,query.get('q',[''])[0],query.get('state',[''])[0]))
                match=IMAGES_PATH.fullmatch(path)
                if match:
                    query=parse_qs(urlsplit(self.path).query)
                    try:
                        page=int(query.get('page',['1'])[0]);page_size=int(query.get('page_size',['48'])[0])
                    except ValueError as exc:
                        raise ProjectError(422,'INVALID_QUERY','分页参数无效。') from exc
                    return self.send_json(200,self.server.images.list(match[1],match[2],page,page_size,
                        query.get('q',[''])[0],query.get('sort',['newest'])[0]))
                match=IMAGE_PATH.fullmatch(path)
                if match:
                    if match[4]: return self.send_image_file(match[1],match[2],match[3],match[4])
                    from .roi import image_info
                    record=self.server.images.get(match[1],match[2],match[3])
                    with self.server.storage.connect() as db:record=image_info(db,record)
                    return self.send_json(200,{'image':record})
                roi_match=re.fullmatch(r'/api/projects/([0-9a-f]{32})/datasets/([0-9a-f]{32})/roi',path)
                if roi_match:
                    self.server.storage.get_dataset(*roi_match.groups())
                    from .roi import setting
                    with self.server.storage.connect() as db:return self.send_json(200,setting(db,roi_match[2]))
                match = DATASETS_PATH.fullmatch(path)
                if match:
                    datasets = self.server.storage.list_datasets(match[1])
                    return self.send_json(200, {"datasets": datasets, "total": len(datasets)})
                match = DATASET_PATH.fullmatch(path)
                if match:
                    return self.send_json(200, {"dataset": self.server.storage.get_dataset(match[1], match[2])})
                match = PROJECT_PATH.fullmatch(path)
                if match:
                    return self.send_json(200, {"project": self.server.storage.get(match[1])})
                if path in STATIC_FILES:
                    filename, content_type = STATIC_FILES[path]
                    return self.send_content(200, (WEB_ROOT / filename).read_bytes(), content_type)
            elif self.command == "POST":
                model_sync=re.fullmatch(r'/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/models/sync',path)
                if model_sync:
                    self.check_fields(self.read_body(),set())
                    results=self.server.models.sync_successful(*model_sync.groups())
                    return self.send_json(200,{'repaired':sum('model' in r for r in results),'errors':[r['model_error'] for r in results if 'model_error' in r]})
                source_match=re.fullmatch(r'/api/projects/([0-9a-f]{32})/tasks/([0-9a-f]{32})/inference-inputs/from-dataset',path)
                if source_match:
                    values=self.read_body();self.check_fields(values,{'dataset_id','image_id'})
                    return self.send_json(201,self.server.inference_inputs.from_dataset(source_match[1],source_match[2],values.get('dataset_id'),values.get('image_id')))
                match=re.fullmatch(r'/api/projects/([0-9a-f]{32})/backup',path)
                if match:
                    self.check_fields(self.read_body(),set())
                    return self.send_json(201,{'backup':self.server.maintenance.backup(match[1])})
                if path=='/api/maintenance/cleanup':
                    values=self.read_body();self.check_fields(values,{'confirmation'})
                    if values.get('confirmation')!='清理过期临时文件':raise ProjectError(422,'CONFIRMATION_REQUIRED','请确认仅清理超过24小时的暂存文件。')
                    return self.send_json(200,self.server.maintenance.cleanup())
                if path=='/api/maintenance/restore':
                    if self.headers.get_content_type()!='application/zip' or self.headers.get('Transfer-Encoding'):raise ProjectError(415,'ZIP_REQUIRED','请上传项目备份ZIP。')
                    try:length=int(self.headers.get('Content-Length','0'))
                    except ValueError:raise ProjectError(400,'INVALID_BODY','长度无效。')
                    if not 0<length<=MAX_ARCHIVE:raise ProjectError(413,'BACKUP_CAPACITY','备份ZIP最多2GiB。')
                    self.server.maintenance.space(self.server.maintenance.restores,length*2)
                    with tempfile.TemporaryDirectory(dir=self.server.maintenance.restores) as temp:
                        source=Path(temp)/'upload.zip'
                        with source.open('xb') as stream:
                            remaining=length
                            while remaining:
                                chunk=self.rfile.read(min(1024**2,remaining))
                                if not chunk:raise ProjectError(400,'INCOMPLETE_BACKUP','备份上传不完整。')
                                self._body_bytes_read+=len(chunk);remaining-=len(chunk);stream.write(chunk)
                        return self.send_json(201,{'restore':self.server.maintenance.restore(source)})
                match=INFERENCE_INPUT_PATH.fullmatch(path)
                if match and not match[3]:
                    self.server.tasks.get(match[1],match[2]);filename,raw,_=self.read_image_body()
                    return self.send_json(200,self.server.inference_inputs.upload(match[1],match[2],filename,raw))
                match=INFERENCE_PATH.fullmatch(path)
                if match:
                    p,t,j,action=match.groups();values=self.read_body()
                    if not j:
                        result=self.server.inferences.create(p,t,values)
                        return self.send_json(200 if result['reused'] else 201,result)
                    self.check_fields(values,set())
                    if action=='cancel':return self.send_json(200,{'job':self.server.inferences.cancel(p,t,j)})
                    if action=='remote-run':return self.send_json(200,{'job':self.server.inference_remote.submit(p,t,j)})
                    if action=='remote-stop':return self.send_json(200,{'job':self.server.inference_remote.cancel(p,t,j)})
                    if action=='remote-pause':return self.send_json(200,{'job':self.server.inference_remote.pause(p,t,j)})
                    raise ProjectError(404,'NOT_FOUND','推理操作不存在。')

                match=EVALUATION_PATH.fullmatch(path)
                if match:
                    p,t,j,action=match.groups();values=self.read_body()
                    if not j:
                        result=self.server.evaluations.create(p,t,values)
                        return self.send_json(200 if result['reused'] else 201,result)
                    self.check_fields(values,set())
                    if action=='cancel':return self.send_json(200,{'job':self.server.evaluations.cancel(p,t,j)})
                    if action=='remote-run':return self.send_json(200,{'job':self.server.evaluation_remote.submit(p,t,j)})
                    if action=='remote-stop':return self.send_json(200,{'job':self.server.evaluation_remote.cancel(p,t,j)})
                    if action=='remote-pause':return self.send_json(200,{'job':self.server.evaluation_remote.pause(p,t,j)})
                    raise ProjectError(404,'NOT_FOUND','评估操作不存在。')

                match=TRAINING_VISIBILITY_PATH.fullmatch(path)
                if match:
                    values=self.read_body(262144);self.check_fields(values,{'records'})
                    result=self.server.training.change_visibility(match[1],match[2],values.get('records'),match[3]=='restore')
                    if match[3]=='restore':self.server.models.sync_successful(match[1],match[2])
                    return self.send_json(200,result)
                match=PREVIEW_PATH.fullmatch(path)
                if match:return self.send_json(200,self.server.augmentation.preview(match[1],match[2],self.read_body()))
                if path=='/api/remote/host-key':
                    from .ssh_setup import host_key
                    body=self.read_body()
                    if set(body)!={'command'}:raise ProjectError(422,'UNKNOWN_FIELDS','请只提交SSH登录指令。')
                    config=self.server.remote.config
                    return self.send_json(200,host_key(body['command'],json.loads(config.read_text('utf-8')) if config.is_file() else None))
                if path=='/api/remote/configure':
                    from .ssh_setup import configure
                    return self.send_json(200,configure(self.server.remote,self.read_body()))
                if path=='/api/remote/instances/connect':
                    return self.send_json(200,self.server.compute_instances.connect(self.read_body()))
                if path=='/api/remote/instances/local':
                    values=self.read_body();self.check_fields(values,{'python','root'})
                    return self.send_json(200,self.server.compute_instances.connect_local(values))
                if path in ('/api/remote/instances/prepare','/api/remote/instances/validate','/api/remote/instances/activate'):
                    body=self.read_body();self.check_fields(body,{'id'})
                    instances=self.server.compute_instances
                    operation=instances.activate if path.endswith('/activate') else instances.prepare
                    return self.send_json(200,operation(body.get('id')))
                if path=='/api/remote/check':
                    values=self.read_body();self.check_fields(values,set())
                    return self.send_json(200,self.server.remote.check())
                if path=='/api/remote/recover':
                    values=self.read_body();self.check_fields(values,set())
                    return self.send_json(200,self.server.remote.recover_device())
                match=TRAINING_PATH.fullmatch(path)
                if match:
                    if match[3] and match[4]=='result':
                        self.server.training.get(match[1],match[2],match[3])
                        return self.send_json(200,self.server.executions.receive(match[1],match[2],match[3],self.read_training_result()))
                    values=self.read_body()
                    if match[3] and match[4]=='restore':
                        self.check_fields(values,{'confirmation_name','visibility_revision'})
                        result=self.server.training.change_visibility(match[1],match[2],[{'id':match[3],**values}],True)
                        self.server.models.sync_successful(match[1],match[2]);return self.send_json(200,result)
                    if match[3] and match[4]=='register-model':
                        self.check_fields(values,set())
                        result=self.server.models.register(match[1],match[2],match[3])
                        try:result['initial_evaluation']=self.server.evaluations.ensure_initial(match[1],match[2],result['model'])
                        except ProjectError as exc:result['evaluation_error']={'code':exc.code,'message':exc.message}
                        return self.send_json(200 if result['reused'] else 201,result)
                    if not match[3]:
                        result=self.server.training.create(match[1],match[2],values)
                        return self.send_json(200 if result['reused'] else 201,result)
                    if match[4]=='remote-abandon':
                        return self.send_json(200,self.server.remote.abandon(match[1],match[2],match[3],values))
                    if match[4]=='remote-stop-and-trash':
                        return self.send_json(200,self.server.remote.stop_and_trash(match[1],match[2],match[3],values))
                    if match[4]=='cancel':
                        self.check_fields(values,set())
                        return self.send_json(200,{'job':self.server.training.cancel(match[1],match[2],match[3])})
                    if match[4]=='dispatch':
                        self.check_fields(values,set())
                        result=self.server.executions.dispatch(match[1],match[2],match[3])
                        return self.send_json(200 if result['reused'] else 201,result)
                    if match[4] in ('remote-run','remote-stop','remote-pause'):
                        self.check_fields(values,set())
                        action={'remote-run':self.server.remote.submit,'remote-stop':self.server.remote.cancel,'remote-pause':self.server.remote.pause}[match[4]]
                        return self.send_json(202,{'job':action(match[1],match[2],match[3])})
                    raise ProjectError(404,'NOT_FOUND','接口不存在。')
                match=VERSIONS_PATH.fullmatch(path)
                if match:
                    values=self.read_body();self.check_fields(values,{'fingerprint','split_revision'})
                    result=self.server.versions.publish(match[1],match[2],values)
                    return self.send_json(200 if result['reused'] else 201,result)
                match=PREPARATION_PATH.fullmatch(path)
                if match and match[3]=='split':
                    values=self.read_body();self.check_fields(values,{'train_percent','val_percent','test_percent','seed','fingerprint','revision','strategy'})
                    result=self.server.preparation.generate(match[1],match[2],values)
                    try:result['auto_version']=self.server.versions.publish(match[1],match[2],{'fingerprint':result['report']['fingerprint'],'split_revision':result['split']['revision']})
                    except ProjectError as exc:raise ProjectError(exc.status,'SPLIT_VERSION_FAILED','划分已保存，但数据版本生成未完成：'+exc.message+' 可重新点击划分重试，不会重复生成版本。') from exc
                    return self.send_json(200,result)
                match=CLASSES_PATH.fullmatch(path)
                if match and not match[3]:
                    values=self.read_body();self.check_fields(values,{'name'})
                    return self.send_json(201,{'class':self.server.annotations.change_class(match[1],match[2],values)})
                match=TASKS_PATH.fullmatch(path)
                if match:
                    values=self.read_body();self.check_fields(values,{'name','description','task_type','dataset_id'})
                    return self.send_json(201,{'task':self.server.tasks.create(match[1],values)})
                match=TASK_PATH.fullmatch(path)
                if match and match[3]=='sync-images':
                    values=self.read_body();self.check_fields(values,{'revision'})
                    return self.send_json(200,self.server.tasks.sync(match[1],match[2],values))
                match=IMAGES_PATH.fullmatch(path)
                if match:
                    self.server.storage.get_dataset(match[1],match[2])
                    filename,raw,group=self.read_image_body()
                    result=self.server.images.import_image(match[1],match[2],filename,raw,group)
                    return self.send_json(201 if result['status']=='imported' else 200,result)
                if path == "/api/projects":
                    values = self.read_body()
                    self.check_fields(values, {"name", "description"})
                    return self.send_json(201, {"project": self.server.storage.create(values)})
                match = DATASETS_PATH.fullmatch(path)
                if match:
                    values = self.read_body()
                    self.check_fields(values, {"name", "description"})
                    return self.send_json(201, {"dataset": self.server.storage.create_dataset(match[1], values)})
            elif self.command in {"PUT", "DELETE"}:
                roi_match=re.fullmatch(r'/api/projects/([0-9a-f]{32})/datasets/([0-9a-f]{32})/roi',path)
                if roi_match and self.command=='PUT':
                    from .roi import save
                    return self.send_json(200,save(self.server.storage,*roi_match.groups(),self.read_body()))

                match=TRAINING_PATH.fullmatch(path)
                if match and match[3] and not match[4] and self.command=='DELETE':
                    values=self.read_body();self.check_fields(values,{'confirmation_name','visibility_revision'})
                    return self.send_json(200,self.server.training.change_visibility(match[1],match[2],[{'id':match[3],**values}]))
                match=CLASSES_PATH.fullmatch(path)
                if match and match[3]:
                    values=self.read_body()
                    self.check_fields(values,{'revision','confirmation_name'} if self.command=='DELETE' else {'name','revision'})
                    return self.send_json(200,{'class':self.server.annotations.change_class(match[1],match[2],values,match[3],self.command=='DELETE')})
                match=ANNOTATION_PATH.fullmatch(path)
                if match and self.command=='PUT':
                    values=self.read_body(2*1024**2);self.check_fields(values,{'revision','state','boxes'})
                    return self.send_json(200,{'annotation':self.server.annotations.save(match[1],match[2],match[3],values)})
                match=TASK_PATH.fullmatch(path)
                if match and not match[3]:
                    values=self.read_body()
                    if self.command=='PUT':
                        self.check_fields(values,{'name','description','revision'})
                        return self.send_json(200,{'task':self.server.tasks.update(match[1],match[2],values)})
                    self.check_fields(values,{'revision','confirmation_name'})
                    task=self.server.tasks.delete(match[1],match[2],values)
                    return self.send_json(200,{'deleted_id':task['id']})
                match=IMAGE_PATH.fullmatch(path)
                if match and self.command=='PUT' and not match[4]:
                    values=self.read_body()
                    self.check_fields(values,{'group_id','revision'})
                    return self.send_json(200,{'image':self.server.images.update_group(match[1],match[2],match[3],values)})
                match = DATASET_PATH.fullmatch(path)
                if match:
                    values = self.read_body()
                    if self.command == "PUT":
                        self.check_fields(values, {"name", "description", "revision"})
                        dataset = self.server.storage.update_dataset(match[1], match[2], values)
                        return self.send_json(200, {"dataset": dataset})
                    self.check_fields(values, {"revision", "confirmation_name"})
                    dataset = self.server.storage.delete_dataset(match[1], match[2], values)
                    return self.send_json(200, {"deleted_id": dataset["id"], "project_id": match[1]})
                match = PROJECT_PATH.fullmatch(path)
                if match:
                    values = self.read_body()
                    if self.command == "PUT":
                        self.check_fields(values, {"name", "description", "revision"})
                        project = self.server.storage.update(match[1], values)
                        return self.send_json(200, {"project": project})
                    self.check_fields(values, {"revision", "confirmation_name"})
                    project = self.server.storage.delete(match[1], values)
                    return self.send_json(200, {"deleted_id": project["id"]})
            raise ProjectError(404, "NOT_FOUND", "请求的页面或功能不存在。")
        except ProjectError as exc:
            self.discard_rejected_body()
            self.send_json(exc.status, {"error": {"code": exc.code, "message": exc.message, "fields": exc.fields}, "request_id": self.request_id})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logging.info("Client disconnected: %s", self.request_id)
        except OSError as exc:
            status=507 if exc.errno==28 else 503
            logging.exception("File operation failed: %s",self.request_id)
            self.send_json(status,{'error':{'code':'STORAGE_FULL' if status==507 else 'FILE_UNAVAILABLE','message':'文件操作未完成，请检查可用空间和权限；保留原数据后再重试。'},'request_id':self.request_id})
        except sqlite3.OperationalError:
            logging.exception("Storage operation failed: %s", self.request_id)
            self.send_json(503, {"error": {"code": "STORAGE_UNAVAILABLE", "message": "暂时无法保存或读取数据，请稍后重试。"}, "request_id": self.request_id})
        except Exception:
            logging.exception("Request failed: %s", self.request_id)
            self.send_json(500, {"error": {"code": "INTERNAL_ERROR", "message": "服务暂时出现问题，请重试。"}, "request_id": self.request_id})
        finally:
            if mutation_locked:self.server.mutation_lock.release()

    @staticmethod
    def check_fields(values, allowed):
        if set(values) - allowed:
            raise ProjectError(422, "UNKNOWN_FIELD", "请求包含不支持的字段。")

    def discard_rejected_body(self):
        # Closing with unread bytes can reset TCP on Windows before the JSON error
        # is received. Discard only bounded, declared bytes; never parse/import them.
        self.close_connection=True
        if self.headers.get('Transfer-Encoding'):return
        try:size=int(self.headers.get('Content-Length','0'))
        except ValueError:return
        remaining=size-self._body_bytes_read
        if not 0<remaining<=MAX_FILE_BYTES:return
        timeout=self.connection.gettimeout();deadline=time.monotonic()+0.25
        try:
            while remaining>0 and time.monotonic()<deadline:
                self.connection.settimeout(max(0.001,deadline-time.monotonic()))
                raw=self.rfile.read1(min(65536,remaining))
                if not raw:break
                remaining-=len(raw)
        except OSError:pass
        finally:self.connection.settimeout(timeout)

    do_GET = dispatch
    do_HEAD = dispatch
    do_POST = dispatch
    do_PUT = dispatch
    do_DELETE = dispatch


def create_server(port, database_path):
    from .workspace_lock import WorkspaceLease
    lease=WorkspaceLease(database_path)
    try:server=Server(("127.0.0.1", port), database_path)
    except BaseException:
        lease.close();raise
    original_close=server.server_close
    def close():
        try:original_close()
        finally:lease.close()
    server.server_close=close
    return server

