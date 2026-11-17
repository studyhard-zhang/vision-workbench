"""Project archives restored into a new namespace. Never execute SQL from an archive."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tempfile
import time
import uuid
import zipfile

from .storage import Storage, ProjectError, timestamp
from .images import MIN_FREE_BYTES
from .versions import encoded, sha_file

MAX_ARCHIVE = 2 * 1024**3
MAX_EXPANDED = 4 * 1024**3
ROOTS = ('assets', 'versions', 'training', 'models', 'evaluations', 'inferences', 'inference-inputs')
BUSINESS_TABLES = {'schema_version','projects','datasets','images','vision_tasks','task_images','task_classes',
    'task_annotations','segmentation_polygons','detection_boxes','task_splits','task_split_images','task_data_versions','task_data_version_images',
    'training_jobs','training_dispatches','training_remote_sessions','model_versions','model_events',
    'training_record_visibility','evaluation_jobs','evaluation_dispatches','evaluation_remote_sessions',
    'inference_jobs','inference_dispatches','inference_remote_sessions','inference_inputs',
    'inference_upload_failures','inference_job_inputs','dataset_rois','roi_annotations'}


class MaintenanceRepository:
    def __init__(self, storage):
        self.storage = storage
        self.root = storage.path.parent.resolve()
        self.archives = self.root / 'project-backups'
        self.archives.mkdir(exist_ok=True)
        self.restores = self.root.parent / 'restored-workspaces'
        self.restores.mkdir(exist_ok=True)

    def space(self, parent, size):
        if shutil.disk_usage(parent).free < size + MIN_FREE_BYTES:
            raise ProjectError(507, 'STORAGE_FULL', '可用空间不足；需要容纳本次文件并保留256MiB，原数据未改变。')

    def idle(self, db, pid=None):
        for kind in ('training','evaluation','inference'):
            sql = f"SELECT count(*) FROM {kind}_jobs j JOIN vision_tasks t ON t.id=j.task_id WHERE j.state IN ('QUEUED','DISPATCHED','RUNNING')"
            args = ()
            if pid: sql += ' AND t.project_id=?'; args=(pid,)
            if db.execute(sql,args).fetchone()[0]:
                raise ProjectError(409, 'MAINTENANCE_ACTIVE_JOB', '存在等待或活动计算任务；请先取消等待或完成原任务，再备份/清理。')
        for kind in ('training','evaluation','inference'):
            if db.execute(f"SELECT 1 FROM {kind}_dispatches WHERE state='AWAITING_RESULT'").fetchone():
                raise ProjectError(409, 'MAINTENANCE_ACTIVE_JOB', '仍有计算结果待收尾，暂不备份或清理。')

    def scoped_rows(self, db, pid):
        self.storage.get(pid,db)
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        if tables != BUSINESS_TABLES: raise ProjectError(409,'BACKUP_SCHEMA','此数据库结构需要更新备份适配器。')
        selected={'projects':[dict(r) for r in db.execute('SELECT * FROM projects WHERE id=?',(pid,))],
                  'schema_version':[dict(r) for r in db.execute('SELECT * FROM schema_version')]}
        while len(selected)<len(tables):
            count=len(selected)
            for table in sorted(tables-set(selected)):
                fks={}
                for r in db.execute(f'PRAGMA foreign_key_list("{table}")'):
                    fks.setdefault(r['id'],[]).append(dict(r))
                if not fks or any(parts[0]['table'] not in selected for parts in fks.values()):continue
                rows=[]
                for row in db.execute(f'SELECT * FROM "{table}"'):
                    keep=True
                    for parts in fks.values():
                        if any(row[p['from']] is None for p in parts):continue
                        if not any(all(row[p['from']]==parent[p['to']] for p in parts) for parent in selected[parts[0]['table']]):keep=False;break
                    if keep:rows.append(dict(row))
                selected[table]=rows
            if len(selected)==count:raise ProjectError(409,'BACKUP_SCHEMA','无法解析项目引用关系。')
        return selected

    def project_files(self, rows):
        directories=[]
        for table,root in [('task_data_versions','versions'),('training_jobs','training'),('model_versions','models'),
                           ('evaluation_jobs','evaluations'),('inference_jobs','inferences')]:
            directories += [self.root/root/r['id'] for r in rows[table]]
        directories += [self.root/'assets'/r['dataset_id']/r['id'] for r in rows['images']]
        directories += [self.root/'inference-inputs'/r['task_id']/r['id'] for r in rows['inference_inputs']]
        files={}
        for directory in directories:
            if not directory.resolve().is_relative_to(self.root) or directory.is_symlink():raise ProjectError(409,'BACKUP_PATH','数据目录含越界引用。')
            for p in directory.rglob('*') if directory.exists() else ():
                if p.is_symlink() or not p.resolve().is_relative_to(self.root):raise ProjectError(409,'BACKUP_PATH','数据中含链接，未打包。')
                if p.is_file() and '.staging' not in p.parts:files[p.relative_to(self.root).as_posix()]=p
        return files

    @staticmethod
    def verify_records(root,rows):
        from tools.polygon_geometry import polygon
        tasks={r['id']:r for r in rows['vision_tasks']};images={r['id']:r for r in rows['images']}
        for row in rows.get('segmentation_polygons',[]):
            if tasks[row['task_id']]['task_type']!='segment':raise ValueError('分割标注归属检测任务')
            image=images[row['image_id']];polygon(json.loads(row['points_json']),image['width'],image['height'])
        for row in rows.get('dataset_rois',[]):
            area=json.loads(row['region_json'])
            if area is not None and (not isinstance(area,list) or len(area)!=4 or any(type(v) not in (int,float) for v in area) or not (0<=area[0]<area[2]<=1 and 0<=area[1]<area[3]<=1)):raise ValueError('ROI设置无效')
        class_ids={(r['task_id'],r['id']) for r in rows['task_classes']}
        for row in rows.get('roi_annotations',[]):
            if row['state'] not in ('annotated','unannotated','no_target','excluded'):raise ValueError('ROI标注状态无效')
            for shape in json.loads(row['boxes_json']):
                if (row['task_id'],shape['class_id']) not in class_ids:raise ValueError('ROI标注类别归属无效')
        def check(path,digest):
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha_file(path)!=digest:
                raise ValueError('业务文件缺失或SHA不符：'+path.name)
        for table,category,parent in [('images','assets','dataset_id'),('inference_inputs','inference-inputs','task_id')]:
            for row in rows[table]:
                suffix='jpg' if row['format']=='JPEG' else 'png'
                folder=root/category/row[parent]/row['id']
                check(folder/('original.'+suffix),row['original_sha256'])
                check(folder/('display.'+suffix),row['display_sha256'])
                if not (folder/'thumbnail.jpg').is_file():raise ValueError('缩略图缺失')
        for kind,category in [('training','training'),('evaluation','evaluations'),('inference','inferences')]:
            for row in rows[kind+'_dispatches']:
                if not row['result_json']:continue
                receipt=json.loads(row['result_json']);folder=root/category/row['job_id']/'returned'
                check(folder/'result.zip',row['result_sha256'])
                for name,spec in receipt['artifacts'].items():
                    if PurePosixPath(name).name!=name:raise ValueError('产物文件名非法')
                    check(folder/name,spec['sha256'])

    def impact(self,pid):
        with self.storage.connect() as db:
            rows=self.scoped_rows(db,pid)
        counts={k:len(v) for k,v in rows.items() if k!='schema_version'}
        return {'counts':counts,'permanent_delete_allowed':not rows['datasets'] and not rows['vision_tasks'],
                'notice':'含数据集、任务、版本或模型的项目保持引用保护，不能级联删除。训练记录回收站不删除权重。'}

    def usage(self):
        groups={}
        for name in (*ROOTS,'backups','project-backups'):
            folder=self.root/name
            paths=[p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()] if folder.exists() else []
            groups[name]={'bytes':sum(p.stat().st_size for p in paths),'files':len(paths)}
            if name=='assets':
                groups[name]['detail']={prefix:sum(p.stat().st_size for p in paths if p.name.startswith(prefix)) for prefix in ('original.','display.','thumbnail.')}
        groups['database']={'bytes':sum(p.stat().st_size for p in self.root.glob('platform.db*') if p.is_file()),'files':len(list(self.root.glob('platform.db*')))}
        groups['releases']={'bytes':sum(p.stat().st_size for p in (self.root/'releases').glob('*.zip')),'files':len(list((self.root/'releases').glob('*.zip')))}
        volume=shutil.disk_usage(self.root)
        log=self.root/'maintenance-events.jsonl'
        events=[]
        if log.exists():
            for line in log.read_text(encoding='utf-8').splitlines()[-30:]:
                try:events.append(json.loads(line))
                except ValueError:pass
        return {'data_directory':str(self.root),'groups':groups,'free_bytes':volume.free,'reserve_bytes':MIN_FREE_BYTES,
                'restore_root':str(self.restores),'cleanup_candidates':len(self.candidates()),'events':events,
                'limits':{'archive_bytes':MAX_ARCHIVE,'expanded_bytes':MAX_EXPANDED}}

    def event(self,kind,**values):
        with (self.root/'maintenance-events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'time':timestamp(),'kind':kind,**values},ensure_ascii=False)+'\n')

    def candidates(self):
        files=[]
        for name in ROOTS:
            stage=self.root/name/'.staging'
            if not stage.exists() or stage.is_symlink():continue
            for p in stage.rglob('*'):
                if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(stage.resolve()) and p.stat().st_mtime<time.time()-86400:files.append(p)
        return files

    def cleanup(self):
        released=0;count=0
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE');self.idle(db)
            for p in self.candidates():
                size=p.stat().st_size;p.unlink();released+=size;count+=1
        self.event('STALE_STAGING_CLEANUP',files=count,released_bytes=released)
        return {'files':count,'released_bytes':released}

    def backup(self,pid):
        bid=uuid.uuid4().hex;temporary=self.archives/(bid+'.partial');destination=self.archives/(bid+'.zip')
        try:
            with self.storage.connect() as db:
                db.execute('BEGIN IMMEDIATE');self.idle(db,pid)
                rows=self.scoped_rows(db,pid);files=self.project_files(rows)
                try:self.verify_records(self.root,rows)
                except ValueError as exc:raise ProjectError(409,'BACKUP_SOURCE_DAMAGED',str(exc)) from exc
                size=sum(p.stat().st_size for p in files.values())
                if size>MAX_EXPANDED:raise ProjectError(413,'BACKUP_CAPACITY','项目超过当前4GiB解压容量，请保留数据并扩容适配。')
                self.space(self.archives,size*2+4*1024**2)
                snapshot=encoded({'format_version':1,'schema':16,'project_id':pid,'tables':rows})
                specs={'snapshot.json':{'bytes':len(snapshot),'sha256':hashlib.sha256(snapshot).hexdigest()}}
                for name,p in files.items():specs['data/'+name]={'bytes':p.stat().st_size,'sha256':sha_file(p)}
                manifest={'format':'vision-workbench-project','version':1,'schema':16,'created_at':timestamp(),
                    'project_id':pid,'project_name':rows['projects'][0]['name'],'files':specs,
                    'restore_policy':'NEW_DIRECTORY_PRESERVE_IDS_NO_SSH_NO_QUEUE_REPLAY',
                    'excluded':['SSH configuration and keys','other projects','backup history','staging files'],
                    'notice':'包含该项目业务原图、标注、历史日志、权重和运行元数据；请自行妥善保管。'}
                with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
                    z.writestr('manifest.json',encoded(manifest));z.writestr('snapshot.json',snapshot)
                    for name,p in files.items():z.write(p,'data/'+name)
                if temporary.stat().st_size>MAX_ARCHIVE:raise ProjectError(413,'BACKUP_CAPACITY','压缩包超过当前2GiB上限，未发布。')
                self.space(self.archives,0)
                # Recheck bytes before publishing: never publish a changed file snapshot.
                with zipfile.ZipFile(temporary) as z:
                    for name,spec in specs.items():
                        with z.open(name) as stream:
                            h=hashlib.file_digest(stream,'sha256').hexdigest()
                        if h!=spec['sha256']:raise ProjectError(409,'BACKUP_CHANGED','数据在备份期间发生变化，未发布备份。')
                temporary.rename(destination)
            self.event('PROJECT_BACKUP',backup_id=bid,project_id=pid,bytes=destination.stat().st_size)
            return {'id':bid,'project_id':pid,'project_name':manifest['project_name'],'bytes':destination.stat().st_size,
                    'sha256':sha_file(destination),'created_at':manifest['created_at'],'files':len(specs)}
        except OSError as exc:
            raise ProjectError(507 if exc.errno==28 else 503,'BACKUP_WRITE_FAILED','备份写入失败，未完成的包不发布，原数据保留。') from exc
        finally:
            if temporary.exists():temporary.unlink()

    def archive(self,bid):
        if not re.fullmatch('[0-9a-f]{32}',bid):raise ProjectError(404,'BACKUP_NOT_FOUND','备份不存在。')
        path=self.archives/(bid+'.zip')
        if not path.is_file() or path.is_symlink():raise ProjectError(404,'BACKUP_NOT_FOUND','备份不存在。')
        return path,'project-'+bid[:8]+'.zip','application/zip'

    def list(self):
        values=[]
        for p in sorted(self.archives.glob('*.zip'),key=lambda p:p.stat().st_mtime,reverse=True):
            if not re.fullmatch('[0-9a-f]{32}',p.stem) or p.is_symlink():continue
            with zipfile.ZipFile(p) as z:m=json.loads(z.read('manifest.json'))
            values.append({'id':p.stem,'project_id':m['project_id'],'project_name':m['project_name'],'created_at':m['created_at'],'bytes':p.stat().st_size})
        return values

    def restored_list(self):
        values=[]
        for p in self.restores.iterdir():
            if not p.is_dir() or p.is_symlink() or not re.fullmatch('[0-9a-f]{32}',p.name):continue
            provenance=p/'restore-provenance.json'
            if not provenance.is_file():continue
            value=json.loads(provenance.read_text(encoding='utf-8'))
            values.append({'id':p.name,'directory':str(p),'data_directory':str(p/'var'),
                'project_id':value['source_project_id'],'restored_at':value['restored_at'],'archive_sha256':value['archive_sha256']})
        return sorted(values,key=lambda r:r['restored_at'],reverse=True)

    def restore(self,archive):
        rid=uuid.uuid4().hex;destination=self.restores/rid
        try:
            with zipfile.ZipFile(archive) as z:
                infos=z.infolist();names=[i.filename for i in infos]
                if len(names)>20000 or len(names)!=len(set(names)):raise ValueError('文件数量超限或重名')
                if 'manifest.json' not in names or z.getinfo('manifest.json').file_size>8*1024**2:raise ValueError('清单缺失或过大')
                for i in infos:
                    p=PurePosixPath(i.filename)
                    if i.flag_bits&1 or stat.S_ISLNK(i.external_attr>>16) or i.is_dir() or p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename:raise ValueError('非法路径/链接/加密文件')
                    if p.as_posix()!=i.filename or any(part.endswith(('.', ' ')) for part in p.parts):raise ValueError('非规范路径')
                m=json.loads(z.read('manifest.json'))
                if m.get('format')!='vision-workbench-project' or m.get('version')!=1 or m.get('schema') not in (14,15,16):raise ValueError('备份格式不支持')
                if set(names)!={'manifest.json',*m['files']}:raise ValueError('文件清单不一致')
                total=sum(i.file_size for i in infos)
                if total>MAX_EXPANDED:raise ValueError('解压容量超限')
                self.space(self.restores,total*2+8*1024**2)
                with tempfile.TemporaryDirectory(prefix='.restore-',dir=self.restores) as tmp:
                    stage=Path(tmp);data=stage/'var';data.mkdir()
                    snapshot=None
                    for name,spec in m['files'].items():
                        i=z.getinfo(name)
                        if i.file_size!=spec['bytes'] or type(spec['bytes']) is not int or spec['bytes']<0:raise ValueError('文件长度不符')
                        if name=='snapshot.json':
                            if i.file_size>32*1024**2:raise ValueError('快照过大')
                            raw=z.read(name)
                            if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('快照哈希不符')
                            snapshot=json.loads(raw);continue
                        path=PurePosixPath(name)
                        if len(path.parts)<3 or path.parts[0]!='data' or path.parts[1] not in ROOTS or '.staging' in path.parts:raise ValueError('不允许的文件类型')
                        target=data.joinpath(*path.parts[1:])
                        if not target.resolve().is_relative_to(data.resolve()):raise ValueError('越界路径')
                        target.parent.mkdir(parents=True,exist_ok=True)
                        with z.open(name) as source,target.open('xb') as out:shutil.copyfileobj(source,out,1024**2)
                        if sha_file(target)!=spec['sha256']:raise ValueError('文件哈希不符')
                    if not snapshot or snapshot.get('schema')!=m['schema'] or snapshot.get('format_version')!=1 or set(snapshot['tables'])!=(BUSINESS_TABLES-({'segmentation_polygons','dataset_rois','roi_annotations'} if m['schema']==14 else {'dataset_rois','roi_annotations'} if m['schema']==15 else set())):raise ValueError('项目快照不完整')
                    rows=snapshot['tables']
                    if rows['schema_version']!=[{'version':m['schema']}]:raise ValueError('数据库格式不符')
                    if len(rows['projects'])!=1 or rows['projects'][0]['id']!=m['project_id'] or snapshot['project_id']!=m['project_id']:raise ValueError('项目身份不符')
                    for kind in ('training','evaluation','inference'):
                        if any(r['state'] in ('QUEUED','DISPATCHED','RUNNING') for r in rows[kind+'_jobs']) or any(r['state']=='AWAITING_RESULT' for r in rows[kind+'_dispatches']):raise ValueError('备份含活动/等待任务，不能恢复原进程或队列')
                    restored=Storage(data/'platform.db')
                    with restored.connect() as db:
                        db.execute('PRAGMA foreign_keys=OFF');db.execute('BEGIN IMMEDIATE')
                        for table,records in rows.items():
                            if table=='schema_version':continue
                            columns=[r['name'] for r in db.execute(f'PRAGMA table_info("{table}")')]
                            for row in records:
                                if set(row)!=set(columns):raise ValueError('字段不匹配')
                                db.execute(f'INSERT INTO "{table}" ({",".join(columns)}) VALUES ({",".join("?" for _ in columns)})',[row[c] for c in columns])
                        if db.execute('PRAGMA foreign_key_check').fetchall():raise ValueError('项目引用不完整')
                        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('数据库检查失败')
                    # Validate trusted app references using the restored files, before publication.
                    from .versions import VersionRepository
                    from .training import TrainingRepository
                    from .executions import ExecutionRepository
                    from .models import ModelRepository
                    versions=VersionRepository(restored)
                    models=ModelRepository(restored,ExecutionRepository(restored,TrainingRepository(restored,versions)))
                    self.verify_records(data,rows)
                    for r in rows['task_data_versions']:
                        if not versions.read(m['project_id'],r['task_id'],r['id'])['integrity']['ok']:raise ValueError('数据版本完整性不通过')
                    for r in rows['model_versions']:
                        if not models.get(m['project_id'],r['task_id'],r['id'])['available']:raise ValueError('模型完整性不通过')
                    (stage/'restore-provenance.json').write_bytes(encoded({'source_project_id':m['project_id'],'restored_at':timestamp(),
                        'archive_sha256':sha_file(Path(archive)),'id_policy':'preserved_in_new_directory','ssh_config_restored':False,'queue_replay':False}))
                    (stage/'RESTORE.md').write_text('恢复目录只含业务数据。启动同版本工作台时使用 --data-dir 指向此目录下var，并选择空闲端口。没有SSH密钥/配置，不会自动恢复远程进程或重跑队列。\n',encoding='utf-8')
                    self.space(self.restores,0)
                    # tempfile owns only the temporary wrapper; move the validated contents into a new destination.
                    destination.mkdir()
                    try:
                        for p in stage.iterdir():p.rename(destination/p.name)
                    except Exception:
                        if destination.resolve().parent==self.restores.resolve():shutil.rmtree(destination)
                        raise
            self.event('PROJECT_RESTORE',restore_id=rid,project_id=m['project_id'])
            return {'id':rid,'project_id':m['project_id'],'directory':str(destination),'data_directory':str(destination/'var'),
                    'project_name':rows['projects'][0]['name'],'id_policy':'preserved_in_new_directory','tables':{k:len(v) for k,v in rows.items()}}
        except ProjectError:raise
        except (ValueError,KeyError,TypeError,zipfile.BadZipFile,RuntimeError,sqlite3.IntegrityError) as exc:
            raise ProjectError(422,'INVALID_PROJECT_BACKUP','恢复校验失败：'+str(exc)[:180]+'；当前项目未改变。') from exc
        except OSError as exc:
            raise ProjectError(507 if exc.errno==28 else 503,'RESTORE_WRITE_FAILED','恢复写入失败，临时目录回滚，当前项目未改变。') from exc
