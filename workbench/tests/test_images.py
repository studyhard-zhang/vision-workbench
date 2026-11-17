import hashlib
import io
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.server import create_server
from app.storage import Storage, ProjectError
from app.images import ImageRepository, MAX_FILE_BYTES
from PIL import Image


def picture(fmt='PNG', color='red', size=(80,40), orientation=None, mode='RGB'):
    with Image.new(mode,size,color) as image:
        target=io.BytesIO()
        options={}
        if orientation:
            exif=Image.Exif();exif[274]=orientation;options['exif']=exif
        image.save(target,fmt,**options)
        return target.getvalue()


class ImageBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=Storage(Path(self.temp.name)/'workspace.db')
        self.project=self.store.create({'name':'图片项目'})
        self.dataset=self.store.create_dataset(self.project['id'],{'name':'原图 A'})
        self.other=self.store.create_dataset(self.project['id'],{'name':'原图 B'})
        self.repo=ImageRepository(self.store)
        self.pid=self.project['id'];self.did=self.dataset['id']

    def tearDown(self): self.temp.cleanup()

    def test_png_and_jpeg_have_thumbnail_canonical_preview_and_unchanged_original(self):
        for fmt,suffix in [('PNG','png'),('JPEG','jpg')]:
            raw=picture(fmt)
            record=self.repo.import_image(self.pid,self.did,'图片.'+suffix,raw,'相机 A')['image']
            path,mime,_=self.repo.file(self.pid,self.did,record['id'],'original')
            self.assertEqual(path.read_bytes(),raw)
            self.assertEqual(record['original_sha256'],hashlib.sha256(raw).hexdigest())
            path,_,_=self.repo.file(self.pid,self.did,record['id'],'thumbnail')
            with Image.open(path) as thumb:
                self.assertLessEqual(thumb.width,360);self.assertLessEqual(thumb.height,240)
            path,_,_=self.repo.file(self.pid,self.did,record['id'],'view')
            with Image.open(path) as preview: self.assertEqual(preview.size,(80,40))
        self.assertEqual(self.store.get_dataset(self.pid,self.did)['image_count'],2)

    def test_exif_orientation_is_consistent_and_original_is_preserved(self):
        raw=picture('JPEG',orientation=6)
        record=self.repo.import_image(self.pid,self.did,'方向.jpg',raw)['image']
        self.assertEqual((record['source_width'],record['source_height']),(80,40))
        self.assertEqual((record['width'],record['height']),(40,80))
        path,_,_=self.repo.file(self.pid,self.did,record['id'],'view')
        with Image.open(path) as preview:
            self.assertEqual(preview.size,(40,80));self.assertNotIn(274,preview.getexif())
        path,_,_=self.repo.file(self.pid,self.did,record['id'],'original')
        self.assertEqual(path.read_bytes(),raw)

    def test_transparent_png_keeps_alpha_in_full_preview(self):
        record=self.repo.import_image(self.pid,self.did,'透明.png',picture(color=(255,0,0,100),mode='RGBA'))['image']
        path,_,_=self.repo.file(self.pid,self.did,record['id'],'view')
        with Image.open(path) as preview: self.assertEqual(preview.getpixel((0,0)),(255,0,0,100))

    def test_content_dedup_preserves_existing_group_but_same_name_different_content_is_allowed(self):
        first=self.repo.import_image(self.pid,self.did,'相机.png',picture(), '日间')
        duplicate=self.repo.import_image(self.pid,self.did,'改名.png',picture(), '夜间')
        second=self.repo.import_image(self.pid,self.did,'相机.png',picture(color='blue'))
        self.assertEqual(duplicate['status'],'duplicate')
        self.assertEqual(duplicate['image'],first['image'])
        with self.assertRaises(ProjectError) as error:
            self.repo.import_image(self.pid,self.did,'伪装.jpg',picture())
        self.assertEqual(error.exception.code,'FORMAT_MISMATCH')
        self.assertNotEqual(second['image']['id'],first['image']['id'])
        self.assertEqual(self.repo.list(self.pid,self.did)['summary']['image_count'],2)
        other=self.repo.import_image(self.pid,self.other['id'],'相机.png',picture(),'另一个来源')
        self.assertEqual(other['status'],'imported')
        self.assertNotEqual(other['image']['id'],first['image']['id'])
        self.assertEqual(self.repo.list(self.pid,self.other['id'])['total'],1)

    def test_simultaneous_duplicate_imports_publish_one_asset(self):
        raw=picture()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:self.repo.import_image(self.pid,self.did,'并发.png',raw),range(2)))
        self.assertEqual(sorted(item['status'] for item in results),['duplicate','imported'])
        self.assertEqual(len(list((self.repo.root/self.did).iterdir())),1)

    @unittest.skipUnless(os.name=='nt','Windows extended-path alias regression')
    def test_extended_windows_path_alias_preserves_ownership_checks(self):
        inside=Path('\\\\?\\'+str(self.repo.root/self.did/'image'))
        with patch.object(Path,'resolve',return_value=inside):
            self.assertEqual(self.repo.owned_path(self.did,'image'),inside)
        for candidate in [self.repo.root,self.repo.root.parent/'outside']:
            extended=Path('\\\\?\\'+str(candidate))
            with patch.object(Path,'resolve',return_value=extended),self.assertRaises(ProjectError):self.repo.owned_path(self.did,'image')

    def test_owned_path_still_rejects_parent_traversal(self):
        with self.assertRaises(ProjectError):self.repo.owned_path('..','outside')

    def test_corrupt_mismatched_unsupported_oversized_and_bad_paths_do_not_create_records(self):
        cases=[('损坏.png',b'not an image'),('伪装.jpg',picture()),('文件.gif',picture()),
               ('../越界.png',picture()),('空.png',b''),('超限.png',b'x'*(MAX_FILE_BYTES+1))]
        for name,raw in cases:
            with self.subTest(name=name),self.assertRaises(ProjectError):
                self.repo.import_image(self.pid,self.did,name,raw)
        with self.assertRaises(ProjectError): self.repo.import_image(self.pid,self.did,'截断.jpg',picture('JPEG')[:100])
        self.assertEqual(self.repo.list(self.pid,self.did)['total'],0)

    def test_pixel_limit_and_space_shortage_have_no_visible_partial_assets(self):
        with patch('app.images.MAX_PIXELS',100),self.assertRaises(ProjectError) as result:
            self.repo.import_image(self.pid,self.did,'像素.png',picture())
        self.assertEqual(result.exception.code,'PIXEL_LIMIT')
        with patch('app.images.shutil.disk_usage',return_value=SimpleNamespace(free=0)),self.assertRaises(ProjectError) as result:
            self.repo.import_image(self.pid,self.did,'空间.png',picture())
        self.assertEqual(result.exception.code,'STORAGE_FULL')
        self.assertEqual(self.repo.list(self.pid,self.did)['total'],0)
        self.assertEqual(list((self.repo.root/'.staging').iterdir()),[])

    def test_database_insert_failure_rolls_back_published_files(self):
        with self.store.connect() as db:
            db.execute("CREATE TRIGGER fail_image BEFORE INSERT ON images BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError): self.repo.import_image(self.pid,self.did,'失败.png',picture())
        self.assertEqual(self.repo.list(self.pid,self.did)['total'],0)
        self.assertEqual(list((self.repo.root/'.staging').iterdir()),[])
        self.assertEqual(list((self.repo.root/self.did).iterdir()),[])

    def test_group_revision_search_pagination_and_image_ownership(self):
        first=self.repo.import_image(self.pid,self.did,'百分比%_图.png',picture())['image']
        self.repo.import_image(self.pid,self.did,'其他.png',picture(color='blue'))
        updated=self.repo.update_group(self.pid,self.did,first['id'],{'group_id':'采集组 A','revision':1})
        self.assertEqual(updated['revision'],2)
        with self.assertRaises(ProjectError): self.repo.update_group(self.pid,self.did,first['id'],{'group_id':'旧改动','revision':1})
        self.assertEqual(self.repo.list(self.pid,self.did,query='%')['total'],1)
        self.assertEqual(self.repo.list(self.pid,self.did,query='采集组 A')['total'],1)
        self.assertEqual(self.repo.list(self.pid,self.did,page_size=1)['pages'],2)
        with self.assertRaises(ProjectError): self.repo.file(self.pid,self.other['id'],first['id'],'original')
        with self.assertRaises(ProjectError): self.store.delete_dataset(self.pid,self.did,{'revision':1,'confirmation_name':self.dataset['name']})
        restored=ImageRepository(Storage(self.store.path))
        self.assertEqual(restored.get(self.pid,self.did,first['id'])['group_id'],'采集组 A')


class ImageHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.server=create_server(0,Path(cls.temp.name)/'http.db')
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'
        cls.project=cls.server.storage.create({'name':'接口图片'})
        cls.dataset=cls.server.storage.create_dataset(cls.project['id'],{'name':'批次'})
        cls.collection=f"/api/projects/{cls.project['id']}/datasets/{cls.dataset['id']}/images"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(3);cls.temp.cleanup()

    def call(self,path,method='GET',body=None,headers=None):
        request=Request(self.base+path,method=method,data=body,headers=headers or {})
        try: response=urlopen(request,timeout=5)
        except HTTPError as error: response=error
        with response:return response.status,response.headers,response.read()

    def test_upload_list_group_preview_original_and_duplicate_roundtrip(self):
        raw=picture('JPEG',color='green')
        headers={'X-File-Name':quote('中文 原图.jpg'),'X-Group-Id':quote('批次 A'),'Content-Type':'image/jpeg'}
        status,_,body=self.call(self.collection,'POST',raw,headers)
        self.assertEqual(status,201);record=json.loads(body)['image'];path=self.collection+'/'+record['id']
        self.assertEqual(self.call(self.collection,'POST',raw,headers)[0],200)
        status,_,body=self.call(self.collection+'?q='+quote('批次 A'))
        self.assertEqual(json.loads(body)['total'],1)
        status,headers,body=self.call(path+'/original')
        self.assertEqual(body,raw);self.assertIn('attachment',headers['Content-Disposition'])
        status,headers,body=self.call(path+'/thumbnail');self.assertEqual(status,200)
        with Image.open(io.BytesIO(body)) as image:self.assertEqual(image.format,'JPEG')
        status,_,_=self.call(path,'PUT',json.dumps({'group_id':'批次 B','revision':1}).encode(),{'Content-Type':'application/json'})
        self.assertEqual(status,200)

    def test_invalid_query_bad_files_and_cross_dataset_are_rejected(self):
        self.assertEqual(self.call(self.collection+'?page=bad')[0],422)
        self.assertEqual(self.call(self.collection+'?page_size=1000')[0],422)
        self.assertEqual(self.call(self.collection,'POST',b'bad',{'X-File-Name':'bad.png'})[0],422)
        status,_,body=self.call(self.collection,'POST',picture(color='yellow'),{'X-File-Name':'unique.png'})
        image_id=json.loads(body)['image']['id']
        other=self.server.storage.create_dataset(self.project['id'],{'name':'另一集合'})
        wrong=self.collection.replace(self.dataset['id'],other['id'])+'/'+image_id+'/original'
        self.assertEqual(self.call(wrong)[0],404)

    def test_cross_site_upload_is_rejected_before_image_import(self):
        status,_,_=self.call(self.collection,'POST',picture(),{'X-File-Name':'cross.png','Sec-Fetch-Site':'cross-site'})
        self.assertEqual(status,403)


class ImageMigrationTests(unittest.TestCase):
    def test_v3_upgrade_preserves_project_and_dataset_and_can_roll_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'v3.db'
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE schema_version(version INTEGER PRIMARY KEY)');db.execute('INSERT INTO schema_version VALUES(3)')
                Storage.create_project_table(db,'projects');Storage.create_dataset_table(db)
                db.execute('INSERT INTO projects VALUES(?,?,?,?,?,?,?)',('a'*32,'原项目','原项目','说明',3,'2026-10-27','2026-10-27'))
                db.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?)',('b'*32,'a'*32,'原数据集','原数据集','原说明',5,'2026-10-27','2026-10-27'));db.commit()
            class FailingStorage(Storage):
                @staticmethod
                def create_image_table(db):
                    Storage.create_image_table(db);raise RuntimeError('Injected migration failure')
            with self.assertRaises(RuntimeError):FailingStorage(path)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],3)
                self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='images'").fetchone())
            store=Storage(path);record=store.get_dataset('a'*32,'b'*32)
            self.assertEqual((record['name'],record['description'],record['revision'],record['image_count']),('原数据集','原说明',5,0))
            self.assertTrue(list((Path(directory)/'backups').glob('schema-v3-*.db')))


if __name__=='__main__':unittest.main()
