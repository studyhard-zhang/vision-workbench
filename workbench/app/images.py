"""Validated image import and canonical EXIF-oriented browsing assets."""
from __future__ import annotations

import hashlib
import io
import math
import os
import re
import shutil
import threading
import unicodedata
import uuid
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .storage import ProjectError, timestamp, validate_revision

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
MIN_FREE_BYTES = 256 * 1024 * 1024
DECODE_SLOTS = threading.BoundedSemaphore(2)


def comparable_path(path):
    """Compare resolved Windows paths with/without the equivalent extended prefix.

    Windows/Python 3.11 may retain this prefix when a parent is concurrently created.
    Keep the resolved path for I/O; normalize only the ownership comparison.
    """
    value=str(path)
    if os.name=='nt':
        if value.startswith('\\\\?\\UNC\\'):value='\\\\'+value[8:]
        elif value.startswith('\\\\?\\'):value=value[4:]
    return Path(value)


def validate_group(value):
    if not isinstance(value, str) or len(value.strip()) > 80 or any(unicodedata.category(c) == 'Cc' for c in value):
        raise ProjectError(422, 'INVALID_GROUP', '采集组最多 80 个字符，不能含控制字符。', {'group_id': '请检查采集组名称。'})
    return value.strip()


class ImageRepository:
    def __init__(self, storage):
        self.storage = storage
        self.root = (storage.path.parent / 'assets').resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / '.staging').mkdir(exist_ok=True)

    def check_scope(self,project_id,dataset_id,db=None):
        return self.storage.get_dataset(project_id,dataset_id,db)

    def execute(self,db,sql,parameters=()):
        return db.execute(sql,parameters)

    def owned_path(self, *parts):
        result = self.root.joinpath(*parts).resolve()
        key,root=comparable_path(result),comparable_path(self.root)
        if key == root or not key.is_relative_to(root):
            raise ProjectError(400, 'INVALID_PATH', '文件路径无效。')
        return result

    def cleanup(self, path):
        checked = self.owned_path(path)
        if checked.exists():
            shutil.rmtree(checked)

    @staticmethod
    def public(row):
        return dict(row)

    def canonical_image(self,oriented,fmt):
        return oriented.convert('RGBA' if fmt=='PNG' and ('A' in oriented.getbands() or 'transparency' in oriented.info) else 'RGB')

    def get(self, project_id, dataset_id, image_id, db=None):
        if db is None:
            with self.storage.connect() as connection:
                return self.get(project_id, dataset_id, image_id, connection)
        self.check_scope(project_id, dataset_id, db)
        row = self.execute(db,'SELECT * FROM images WHERE id=? AND dataset_id=?', (image_id, dataset_id)).fetchone()
        if row is None:
            raise ProjectError(404, 'IMAGE_NOT_FOUND', '图片不存在或不属于当前数据集。')
        return self.public(row)

    def list(self, project_id, dataset_id, page=1, page_size=48, query='', sort='newest'):
        if not 1 <= page <= 1_000_000 or not 1 <= page_size <= 100 or len(query) > 255 or sort not in {'newest', 'name'}:
            raise ProjectError(422, 'INVALID_QUERY', '图片查询参数无效。')
        with self.storage.connect() as db:
            self.check_scope(project_id, dataset_id, db)
            query = query.strip()
            where, args = 'dataset_id=?', [dataset_id]
            if query:
                needle = '%' + query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
                where += " AND (filename LIKE ? ESCAPE '\\' OR group_id LIKE ? ESCAPE '\\')"
                args += [needle, needle]
            total = self.execute(db,'SELECT count(*) FROM images WHERE ' + where, args).fetchone()[0]
            order = 'created_at DESC,id DESC' if sort == 'newest' else 'filename COLLATE NOCASE,id'
            rows = self.execute(db,'SELECT * FROM images WHERE ' + where + ' ORDER BY ' + order + ' LIMIT ? OFFSET ?',
                              [*args, page_size, (page - 1) * page_size]).fetchall()
            summary = dict(self.execute(db,
                "SELECT count(*) AS image_count,coalesce(sum(byte_size),0) AS original_bytes,"
                "count(DISTINCT nullif(group_id,'')) AS group_count FROM images WHERE dataset_id=?", (dataset_id,)
            ).fetchone())
            return {'images': [__import__('app.roi',fromlist=['image_info']).image_info(db,self.public(row)) for row in rows], 'total': total, 'page': page,
                    'page_size': page_size, 'pages': max(1, math.ceil(total/page_size)), 'summary': summary}

    def import_image(self, project_id, dataset_id, filename, raw, group_id=''):
        self.check_scope(project_id, dataset_id)
        if not isinstance(filename, str) or not 1 <= len(filename) <= 255 or any(c in filename for c in '/\\') or any(unicodedata.category(c) == 'Cc' for c in filename):
            raise ProjectError(422, 'INVALID_FILENAME', '图片文件名无效。')
        extension = Path(filename).suffix.lower()
        if extension not in {'.jpg', '.jpeg', '.png'}:
            raise ProjectError(422, 'UNSUPPORTED_FORMAT', '仅支持 JPG/JPEG、PNG 图片。')
        if not raw or len(raw) > MAX_FILE_BYTES:
            raise ProjectError(413, 'IMAGE_SIZE_LIMIT', '图片不能为空，单张最多 20MB。')
        group_id = validate_group(group_id)
        digest = hashlib.sha256(raw).hexdigest()
        with self.storage.connect() as db:
            existing = self.execute(db,'SELECT * FROM images WHERE dataset_id=? AND original_sha256=?', (dataset_id,digest)).fetchone()
            if existing:
                if (existing['format'] == 'JPEG') != (extension in {'.jpg','.jpeg'}):
                    raise ProjectError(422,'FORMAT_MISMATCH','文件扩展名与真实图片格式不一致。')
                return {'status': 'duplicate', 'image': self.public(existing), 'message': '相同内容已存在，未重复导入；原采集组保持不变。'}
        if not DECODE_SLOTS.acquire(blocking=False):
            raise ProjectError(429, 'IMAGE_BUSY', '图片处理繁忙，请稍后重试。')
        image_id = uuid.uuid4().hex
        stage = self.owned_path('.staging', image_id)
        destination = self.owned_path(dataset_id, image_id)
        published = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as check:
                    fmt = check.format
                    if fmt not in {'JPEG', 'PNG'} or (fmt == 'JPEG') != (extension in {'.jpg','.jpeg'}):
                        raise ProjectError(422, 'FORMAT_MISMATCH', '文件扩展名与真实图片格式不一致。')
                    source_width, source_height = check.size
                    if source_width * source_height > MAX_PIXELS:
                        raise ProjectError(422, 'PIXEL_LIMIT', '单张图片最多 4000 万像素，请先缩小图片。')
                    if getattr(check, 'n_frames', 1) != 1:
                        raise ProjectError(422, 'ANIMATED_IMAGE', '首版只支持单帧图片，请先导出静态图片。')
                    check.verify()
                required = MIN_FREE_BYTES + len(raw) + source_width*source_height*4
                if shutil.disk_usage(self.root).free < required:
                    raise ProjectError(507, 'STORAGE_FULL', '存储空间不足，请释放空间后重试。')
                with Image.open(io.BytesIO(raw)) as source:
                    source.load()
                    oriented = ImageOps.exif_transpose(source)
                    canonical = self.canonical_image(oriented,fmt)
                try:
                    stage.mkdir()
                    suffix = 'jpg' if fmt == 'JPEG' else 'png'
                    (stage / f'original.{suffix}').write_bytes(raw)
                    display = stage / f'display.{suffix}'
                    if fmt == 'JPEG':
                        canonical.save(display, 'JPEG', quality=95, subsampling=0, exif=b'')
                    else:
                        canonical.save(display, 'PNG')
                    thumbnail = canonical.copy()
                    try:
                        thumbnail.thumbnail((360,240), Image.Resampling.LANCZOS)
                        background = Image.new('RGB', thumbnail.size, '#f1f4f2')
                        try:
                            background.paste(thumbnail, mask=thumbnail.getchannel('A') if thumbnail.mode == 'RGBA' else None)
                            background.save(stage/'thumbnail.jpg', 'JPEG', quality=85)
                        finally:
                            background.close()
                    finally:
                        thumbnail.close()
                    display_digest = hashlib.sha256(display.read_bytes()).hexdigest()
                    width, height = canonical.size
                finally:
                    canonical.close()
                if shutil.disk_usage(self.root).free < MIN_FREE_BYTES:
                    raise ProjectError(507, 'STORAGE_FULL', '生成预览后可用空间不足，本张图片未发布。')
                now = timestamp()
                with self.storage.connect() as db:
                    self.execute(db,'BEGIN IMMEDIATE')
                    self.check_scope(project_id, dataset_id, db)
                    existing = self.execute(db,'SELECT * FROM images WHERE dataset_id=? AND original_sha256=?', (dataset_id,digest)).fetchone()
                    if existing:
                        return {'status':'duplicate','image':self.public(existing),'message':'相同内容已存在，未重复导入。'}
                    destination.parent.mkdir(exist_ok=True)
                    os.replace(stage,destination)
                    published = True
                    self.execute(db,
                        'INSERT INTO images(id,dataset_id,filename,original_sha256,display_sha256,byte_size,width,height,'
                        'source_width,source_height,format,group_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (image_id,dataset_id,filename,digest,display_digest,len(raw),width,height,
                         source_width,source_height,fmt,group_id,now,now)
                    )
                    record = self.get(project_id,dataset_id,image_id,db)
                published = False
                return {'status':'imported','image':record,'message':'导入成功。'}
        except ProjectError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ProjectError(422,'PIXEL_LIMIT','图片解码尺寸超出限制。') from exc
        except (UnidentifiedImageError, SyntaxError, ValueError) as exc:
            raise ProjectError(422,'INVALID_IMAGE','图片无法完整解码，可能已损坏。') from exc
        except OSError as exc:
            if exc.errno == 28:
                raise ProjectError(507,'STORAGE_FULL','写入时存储空间不足，本张图片未发布。') from exc
            if stage.exists() or published:
                raise ProjectError(503,'IMAGE_WRITE_FAILED','无法保存图片，本张图片未发布，请重试。') from exc
            raise ProjectError(422,'INVALID_IMAGE','图片无法完整解码，可能已损坏。') from exc
        finally:
            try:
                if stage.exists(): self.cleanup(stage)
                if published and destination.exists(): self.cleanup(destination)
            finally:
                DECODE_SLOTS.release()

    def update_group(self, project_id,dataset_id,image_id,values):
        group_id = validate_group(values.get('group_id'))
        revision = validate_revision(values.get('revision'),'图片')
        with self.storage.connect() as db:
            self.execute(db,'BEGIN IMMEDIATE')
            current = self.get(project_id,dataset_id,image_id,db)
            if current['revision'] != revision:
                raise ProjectError(409,'REVISION_CONFLICT','图片信息已在其他页面更新，请保留输入并重新打开图片后重试。')
            self.execute(db,'UPDATE images SET group_id=?,revision=revision+1,updated_at=? WHERE id=?', (group_id,timestamp(),image_id))
            return self.get(project_id,dataset_id,image_id,db)

    def file(self, project_id,dataset_id,image_id,kind):
        record = self.get(project_id,dataset_id,image_id)
        suffix = 'jpg' if record['format'] == 'JPEG' else 'png'
        names = {'thumbnail': 'thumbnail.jpg', 'view': f'display.{suffix}', 'original': f'original.{suffix}'}
        if kind not in names: raise ProjectError(404,'NOT_FOUND','文件不存在。')
        path = self.owned_path(dataset_id,image_id,names[kind])
        if kind in ('view','thumbnail'):
            from .roi import bounds,cropped
            with self.storage.connect() as db:area=bounds(db,record)
            if area:
                path=self.owned_path(dataset_id,image_id,'roi-'+area['key']+'.png')
                original=self.owned_path(dataset_id,image_id,names['view'])
                if hashlib.sha256(original.read_bytes()).hexdigest()!=record['display_sha256']:raise ProjectError(409,'ASSET_CHANGED','原预览图校验不符，未使用ROI缓存，请检查原文件。')
                checksum=path.with_suffix('.sha256')
                if not path.is_file() or not checksum.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=checksum.read_text('ascii'):
                    raw=cropped(original,area)
                    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');temporary.write_bytes(raw);temporary.replace(path)
                    checksum.write_text(hashlib.sha256(raw).hexdigest(),'ascii')
                return path,'image/png',record

        if not path.is_file(): raise ProjectError(404,'IMAGE_FILE_MISSING','图片文件缺失，请保留记录并检查存储。')
        mime = 'image/jpeg' if kind=='thumbnail' or suffix=='jpg' else 'image/png'
        return path,mime,record
