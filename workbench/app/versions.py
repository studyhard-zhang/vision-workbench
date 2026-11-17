"""Immutable task input snapshots and portable YOLO bundles; no training imports."""
import hashlib
import json
import shutil
import tempfile
import threading
import uuid
import zipfile
import time
from contextlib import contextmanager

from .images import MIN_FREE_BYTES, comparable_path
from .preparation import PreparationRepository, SUBSETS, grouped_images
from .storage import ProjectError, timestamp
from tools.verify_dataset import audit


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def sha_file(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def yolo_label(row, class_indices):
    lines=[]
    for box in row['boxes']:
        coords=tuple(v for x,y in box['points'] for v in (x/row['width'],y/row['height'])) if 'points' in box else ((box['x']+box['width']/2)/row['width'],(box['y']+box['height']/2)/row['height'],box['width']/row['width'],box['height']/row['height'])
        lines.append(str(class_indices[box['class_id']])+' '+' '.join(format(v,'.12g') for v in coords))
    return ('\n'.join(lines)+'\n' if lines else '').encode('utf-8')


class VersionRepository:
    def __init__(self, storage):
        self.storage=storage;self.prep=PreparationRepository(storage);self.tasks=self.prep.tasks;self.assets=self.prep.assets
        self.root=(storage.path.parent/'versions').resolve();self.root.mkdir(exist_ok=True)
        self.staging=self.root/'.staging';self.staging.mkdir(exist_ok=True)
        self.slot=threading.BoundedSemaphore(1)

    def owned(self, *parts):
        path=self.root.joinpath(*parts).resolve()
        key,root=comparable_path(path),comparable_path(self.root)
        if key==root or not key.is_relative_to(root):raise ProjectError(400,'INVALID_PATH','版本文件路径无效。')
        return path

    def space(self, required):
        if shutil.disk_usage(self.root).free < required+MIN_FREE_BYTES:
            raise ProjectError(507,'STORAGE_FULL','可用空间不足，需保留至少 256MiB；本次未发布或导出，请释放空间后重试。')

    @contextmanager
    def operation(self):
        if not self.slot.acquire(blocking=False):raise ProjectError(409,'VERSION_BUSY','正在发布或打包数据版本，请稍后重试。')
        try:yield
        except OSError as exc:
            if isinstance(exc,(BrokenPipeError,ConnectionResetError)):raise
            raise ProjectError(507 if exc.errno==28 else 503,'VERSION_WRITE_FAILED','无法完成版本文件操作，本次未完成；请检查空间和文件权限后重试。') from exc
        finally:self.slot.release()

    def row(self, db, project_id, task_id, version_id):
        self.tasks.get(project_id,task_id,db)
        row=db.execute('SELECT * FROM task_data_versions WHERE id=? AND task_id=?',(version_id,task_id)).fetchone()
        if row is None:raise ProjectError(404,'VERSION_NOT_FOUND','数据版本不存在或不属于当前任务。')
        return dict(row)

    @staticmethod
    def summary(row):
        manifest=json.loads(row['manifest_json'])
        return {key:row[key] for key in ('id','number','fingerprint','split_revision','created_at')} | {
            'image_count':len(manifest['images']),'class_count':len(manifest['classes']),
            'distribution':manifest['distribution'],'warnings':manifest['warnings']}

    def list(self, project_id, task_id):
        with self.storage.connect() as db:
            self.tasks.get(project_id,task_id,db)
            return [self.summary(dict(row)) for row in db.execute('SELECT * FROM task_data_versions WHERE task_id=? ORDER BY number DESC',(task_id,))]

    def read(self, project_id, task_id, version_id):
        with self.storage.connect() as db:row=self.row(db,project_id,task_id,version_id)
        manifest=json.loads(row['manifest_json'])
        return {'version':self.summary(row),'manifest':manifest,'integrity':self.integrity(row,manifest)}

    def source_path(self, manifest, image, kind):
        if kind=='view' and image.get('roi'):return self.owned(manifest['version_id'],image['image_path'])
        suffix='jpg' if image.get('source_format',image['format'])=='JPEG' else 'png'
        return self.assets.owned_path(manifest['dataset_id'],image['id'],('original.' if kind=='original' else 'display.')+suffix)

    def integrity(self, row, manifest):
        issues=[]
        def check(path, expected, label):
            try:valid=sha_file(path)==expected
            except OSError:valid=False
            if not valid:issues.append(label)
        check(self.owned(row['id'],'manifest.json'),row['manifest_sha256'],'版本清单缺失或校验不符')
        for relative,expected in manifest['files'].items():
            check(self.owned(row['id'],relative),expected,'版本文件缺失或校验不符：'+relative)
        for image in manifest['images']:
            for kind,field,label in [('original','original_sha256','原图'),('view','display_sha256','训练图')]:
                check(self.source_path(manifest,image,kind),image[field],label+'缺失或校验不符：'+image['filename'])
        return {'ok':not issues,'issue_count':len(issues),'issues':issues[:50]}

    @staticmethod
    def require_integrity(result):
        if not result['ok']:raise ProjectError(409,'VERSION_DAMAGED','版本文件或引用图片缺失/变化，不能导出。请保留版本并检查存储；不会从当前工作区重建旧版本。')

    @staticmethod
    def validate_members(split, rows):
        mapping={r['image_id']:r['subset'] for r in split['images']}
        eligible={r['id'] for r in rows if r['state'] in ('annotated','no_target')}
        valid=set(mapping)==eligible and all(s in SUBSETS for s in mapping.values())
        for subset in SUBSETS:
            members=[r for r in rows if mapping.get(r['id'])==subset]
            if split[subset+'_percent']:
                valid=valid and bool(members) and (subset=='test' or any(r['state']=='annotated' for r in members))
            else:valid=valid and not members
        present={b['class_id'] for r in rows if r['id'] in eligible for b in r['boxes']}
        train={b['class_id'] for r in rows if mapping.get(r['id'])=='train' for b in r['boxes']}
        valid=valid and train==present
        for group in grouped_images(rows):valid=valid and len({mapping.get(r['id']) for r in group})==1
        if not valid:raise ProjectError(409,'SPLIT_INVALID','划分成员或集合约束不完整，请重新生成划分后发布。')
        return mapping

    def publish(self, project_id, task_id, values):
        fingerprint=values.get('fingerprint');revision=values.get('split_revision')
        if not isinstance(fingerprint,str) or len(fingerprint)!=64 or type(revision) is not int or revision<1:
            raise ProjectError(422,'INVALID_VERSION_INPUT','请先检查并保存有效划分，再发布数据版本。')
        version_id=uuid.uuid4().hex;stage=self.owned('.staging',version_id);destination=self.owned(version_id)
        committed=False;published=False
        with self.operation():
            try:
                with self.storage.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    self.tasks.get(project_id,task_id,db)
                    # A retry after a lost response resolves to the original immutable version.
                    existing=db.execute('SELECT * FROM task_data_versions WHERE task_id=? AND fingerprint=? AND split_revision=?',(task_id,fingerprint,revision)).fetchone()
                    if existing:
                        row=dict(existing);manifest=json.loads(row['manifest_json']);self.require_integrity(self.integrity(row,manifest))
                        return {'version':self.summary(row),'reused':True}
                    report,rows,classes=self.prep.inspect(project_id,task_id,db)
                    split=self.prep.saved(db,task_id,report,rows,classes)
                    if not split:raise ProjectError(409,'SPLIT_REQUIRED','请先生成并保存数据划分。')
                    if split['stale'] or split['fingerprint']!=fingerprint or split['revision']!=revision:
                        raise ProjectError(409,'SPLIT_STALE','划分或标注已变化，请重新检查并保存有效划分后发布。')
                    if report['error_count']:raise ProjectError(422,'DATA_NOT_READY','当前有效数据存在阻断项，请先处理。')
                    mapping=self.validate_members(split,rows)
                    task=self.tasks.get(project_id,task_id,db)
                    number=db.execute('SELECT COALESCE(MAX(number),0)+1 FROM task_data_versions WHERE task_id=?',(task_id,)).fetchone()[0]
                    classes=[dict(c,index=i) for i,c in enumerate(classes)];indices={c['id']:c['index'] for c in classes}
                    files={};images=[]
                    for source in rows:
                        if source['id'] not in mapping:continue
                        subset=mapping[source['id']];suffix='jpg' if source['format']=='JPEG' else 'png'
                        image={k:source[k] for k in ('id','filename','format','width','height','group_id','state','boxes','original_sha256','display_sha256')}
                        image['group_id']=''
                        image.update(subset=subset,image_path=f'images/{subset}/{source["id"]}.{suffix}',label_path=f'labels/{subset}/{source["id"]}.txt')
                        if source.get('roi'):
                            image['roi']=source['roi'];image['source_format']=source['format'];image['format']='PNG';image['image_path']=f'images/{subset}/{source["id"]}.png'
                            path,_,_=self.assets.file(project_id,task['dataset_id'],source['id'],'view');raw=path.read_bytes();files[image['image_path']]=raw;image['display_sha256']=hashlib.sha256(raw).hexdigest()
                        files[image['label_path']]=yolo_label(source,indices);images.append(image)
                    config={'train':'images/train','val':'images/val','names':[c['name'] for c in classes]}
                    if split['test_percent']:config['test']='images/test'
                    # JSON is valid YAML; no interpolated names, paths, or executable download key.
                    files['data.yaml']=encoded(config)
                    manifest={'format_version':1,'version_id':version_id,'number':number,'created_at':timestamp(),
                        'project_id':project_id,'task_id':task_id,'task_name':task['name'],'dataset_id':task['dataset_id'],'dataset_name':task['dataset_name'],
                        'fingerprint':fingerprint,'coordinate_system':'canonical-exif-pixels','label_format':'yolo-segment-polygon-normalized-v1' if task['task_type']=='segment' else 'yolo-detect-xywh-normalized-v1',
                        'split':{k:split[k] for k in ('revision','seed','algorithm','train_percent','val_percent','test_percent')},
                        'classes':classes,'images':images,'distribution':split['distribution'],
                        'warnings':[i['message'] for i in report['issues'] if i['severity']=='warning']+split['warnings'],
                        'files':{name:hashlib.sha256(value).hexdigest() for name,value in files.items() if not name.startswith('images/')}}
                    payload=encoded(manifest);files['manifest.json']=payload
                    self.space(sum(map(len,files.values()))+len(files)*4096)
                    stage.mkdir()
                    for relative,value in files.items():
                        path=self.owned('.staging',version_id,relative);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
                    self.space(0)
                    stage.rename(destination);published=True
                    db.execute('INSERT INTO task_data_versions VALUES(?,?,?,?,?,?,?,?)',
                        (version_id,task_id,number,fingerprint,revision,manifest['created_at'],payload.decode('utf-8'),hashlib.sha256(payload).hexdigest()))
                    db.executemany('INSERT INTO task_data_version_images VALUES(?,?)',[(version_id,image['id']) for image in images])
                    db.execute('UPDATE vision_tasks SET revision=revision+1,updated_at=? WHERE id=?',(timestamp(),task_id))
                    result=self.summary(dict(db.execute('SELECT * FROM task_data_versions WHERE id=?',(version_id,)).fetchone()))
                committed=True
                return {'version':result,'reused':False}
            finally:
                if stage.exists():shutil.rmtree(self.owned('.staging',version_id))
                if published and not committed and destination.exists():shutil.rmtree(self.owned(version_id))

    def image(self, project_id, task_id, version_id, image_id):
        with self.storage.connect() as db:row=self.row(db,project_id,task_id,version_id)
        manifest=json.loads(row['manifest_json']);image=next((r for r in manifest['images'] if r['id']==image_id),None)
        if image is None:raise ProjectError(404,'VERSION_IMAGE_NOT_FOUND','图片不属于该版本。')
        path=self.source_path(manifest,image,'view')
        try:raw=path.read_bytes()
        except OSError:raise ProjectError(409,'VERSION_DAMAGED','版本引用的训练图片缺失。')
        if hashlib.sha256(raw).hexdigest()!=image['display_sha256']:raise ProjectError(409,'VERSION_DAMAGED','版本引用的训练图片校验不符。')
        return raw,'image/jpeg' if image['format']=='JPEG' else 'image/png'

    def readiness(self, project_id, task_id, version_id):
        started=time.perf_counter()
        with self.archive(project_id,task_id,version_id) as (stream,size,_):
            report=audit(stream)
        report.update(source_integrity_checked=True,archive_bytes=size,total_elapsed_seconds=round(time.perf_counter()-started,4))
        return report

    @contextmanager
    def archive(self, project_id, task_id, version_id):
        with self.operation():
            with self.storage.connect() as db:row=self.row(db,project_id,task_id,version_id)
            manifest=json.loads(row['manifest_json']);self.require_integrity(self.integrity(row,manifest))
            size=sum(self.source_path(manifest,image,'view').stat().st_size for image in manifest['images'])
            size+=sum(self.owned(version_id,path).stat().st_size for path in ['manifest.json',*manifest['files']])
            self.space(size+4096*(len(manifest['images'])+len(manifest['files'])+1))
            with tempfile.TemporaryFile(dir=self.staging,suffix='.zip') as stream:
                with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
                    for relative,expected in {'manifest.json':row['manifest_sha256'],**manifest['files']}.items():
                        raw=self.owned(version_id,relative).read_bytes()
                        if hashlib.sha256(raw).hexdigest()!=expected:raise ProjectError(409,'VERSION_DAMAGED','版本文件在打包时发生变化，请检查存储。')
                        archive.writestr(relative,raw)
                    for image in manifest['images']:
                        if image['image_path'] in manifest['files']:continue
                        actual=hashlib.sha256()
                        with self.source_path(manifest,image,'view').open('rb') as source,archive.open(image['image_path'],'w',force_zip64=True) as target:
                            for chunk in iter(lambda:source.read(1024*1024),b''):actual.update(chunk);target.write(chunk)
                        if actual.hexdigest()!=image['display_sha256']:raise ProjectError(409,'VERSION_DAMAGED','训练图片在打包时发生变化，请检查存储。')
                size=stream.tell();stream.seek(0)
                yield stream,size,f'data-v{row["number"]:03d}-{version_id[:8]}.zip'
