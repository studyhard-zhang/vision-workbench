import io
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from app.storage import Storage,ProjectError
from app.tasks import TaskRepository
from app.images import ImageRepository
from app.server import create_server
from PIL import Image


def sample(color='red'):
    with Image.new('RGB',(32,20),color) as image:
        stream=io.BytesIO();image.save(stream,'PNG');return stream.getvalue()


class TaskBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'work.db'
        self.store=Storage(self.path);self.tasks=TaskRepository(self.store);self.images=ImageRepository(self.store)
        self.project=self.store.create({'name':'项目 A'});self.other_project=self.store.create({'name':'项目 B'})
        self.pid=self.project['id'];self.dataset=self.store.create_dataset(self.pid,{'name':'原图'})
        self.did=self.dataset['id'];self.other_dataset=self.store.create_dataset(self.other_project['id'],{'name':'另一项目原图'})

    def tearDown(self):self.temp.cleanup()

    def create(self,name='检测任务',dataset_id=None,project_id=None):
        return self.tasks.create(project_id or self.pid,{'name':name,'dataset_id':dataset_id or self.did,'task_type':'detect'})

    def test_same_dataset_can_feed_independent_task_image_lists_without_copying_images(self):
        image=self.images.import_image(self.pid,self.did,'原图.png',sample())['image']
        files={str(path.relative_to(self.images.root)):path.read_bytes() for path in self.images.root.rglob('*') if path.is_file()}
        first=self.create('任务 A');second=self.create('任务 B')
        self.assertEqual((first['image_count'],second['image_count']),(1,1))
        self.assertEqual(self.tasks.images(self.pid,first['id'])['images'][0]['id'],image['id'])
        self.assertEqual(self.tasks.images(self.pid,second['id'])['images'][0]['id'],image['id'])
        self.assertEqual(self.store.get(self.pid)['task_count'],2)
        self.assertEqual(self.store.get_dataset(self.pid,self.did)['task_count'],2)
        self.assertEqual(files,{str(path.relative_to(self.images.root)):path.read_bytes() for path in self.images.root.rglob('*') if path.is_file()})

    def test_empty_dataset_can_create_draft_and_nonempty_source_is_added_only_on_explicit_sync(self):
        first=self.create('任务 A');second=self.create('任务 B')
        self.assertEqual(first['image_count'],0)
        self.images.import_image(self.pid,self.did,'新增.png',sample())
        pending=self.tasks.get(self.pid,first['id']);self.assertEqual((pending['image_count'],pending['pending_image_count']),(0,1))
        result=self.tasks.sync(self.pid,first['id'],{'revision':1})
        self.assertEqual((result['added_count'],result['task']['image_count'],result['task']['revision']),(1,1,2))
        self.assertEqual(self.tasks.sync(self.pid,first['id'],{'revision':2})['added_count'],0)
        other=self.tasks.get(self.pid,second['id']);self.assertEqual((other['image_count'],other['pending_image_count']),(0,1))
        with self.assertRaises(ProjectError):self.tasks.sync(self.pid,first['id'],{'revision':1})

    def test_task_names_are_unique_within_project_but_not_across_projects(self):
        self.create('Camera A')
        with self.assertRaises(ProjectError) as result:self.create(' ＣＡＭＥＲＡ A ')
        self.assertEqual(result.exception.code,'TASK_NAME_EXISTS')
        other=self.create('Camera A',self.other_dataset['id'],self.other_project['id'])
        self.assertEqual(other['project_id'],self.other_project['id'])
        for name in ['', ' ', 'x'*61, '名字\n换行']:
            with self.assertRaises(ProjectError):self.create(name)

    def test_task_type_and_cross_project_sources_are_rejected(self):
        for task_type in ['classify','semantic',None]:
            with self.assertRaises(ProjectError) as result:self.tasks.create(self.pid,{'name':'不支持','dataset_id':self.did,'task_type':task_type})
            self.assertEqual(result.exception.code,'UNSUPPORTED_TASK_TYPE')
        with self.assertRaises(ProjectError):self.create('跨项目来源',self.other_dataset['id'])
        self.assertEqual(self.tasks.list(self.pid),[])

    def test_update_conflict_and_rename_failure_preserve_task_source(self):
        first=self.create('任务 A');self.create('任务 B')
        updated=self.tasks.update(self.pid,first['id'],{'name':'任务 A 新','description':'说明\n第二行','revision':1})
        self.assertEqual((updated['dataset_id'],updated['task_type']),(self.did,'detect'))
        with self.assertRaises(ProjectError):self.tasks.update(self.pid,first['id'],{'name':'旧页面','revision':1})
        with self.assertRaises(ProjectError):self.tasks.update(self.pid,first['id'],{'name':'任务 B','revision':2})
        self.assertEqual(self.tasks.get(self.pid,first['id']),updated)

    def test_cross_project_ids_do_not_allow_task_reads_updates_syncs_or_deletes(self):
        task=self.create();other=self.other_project['id']
        operations=[lambda:self.tasks.get(other,task['id']),lambda:self.tasks.images(other,task['id']),
            lambda:self.tasks.update(other,task['id'],{'name':'坏改动','revision':1}),
            lambda:self.tasks.sync(other,task['id'],{'revision':1}),
            lambda:self.tasks.delete(other,task['id'],{'revision':1,'confirmation_name':task['name']})]
        for operation in operations:
            with self.assertRaises(ProjectError) as result:operation()
            self.assertEqual(result.exception.code,'TASK_NOT_FOUND')
        self.assertEqual(self.tasks.get(self.pid,task['id']),task)

    def test_deleting_one_task_keeps_source_images_and_other_tasks_and_protects_dataset(self):
        image=self.images.import_image(self.pid,self.did,'保留.png',sample())['image']
        first=self.create('任务 A');second=self.create('任务 B')
        with self.assertRaises(ProjectError):self.tasks.delete(self.pid,first['id'],{'revision':1,'confirmation_name':'错误'})
        with self.assertRaises(ProjectError) as result:self.store.delete_dataset(self.pid,self.did,{'revision':1,'confirmation_name':self.dataset['name']})
        self.assertEqual(result.exception.code,'DATASET_IN_USE')
        self.tasks.delete(self.pid,first['id'],{'revision':1,'confirmation_name':first['name']})
        self.assertEqual(self.tasks.get(self.pid,second['id'])['image_count'],1)
        path,_,_=self.images.file(self.pid,self.did,image['id'],'original');self.assertEqual(path.read_bytes(),sample())
        self.assertEqual(self.store.get(self.pid)['task_count'],1)

    def test_database_foreign_keys_prevent_wrong_source_mapping_and_empty_dataset_deletion(self):
        empty_task=self.create()
        with self.assertRaises(ProjectError):self.store.delete_dataset(self.pid,self.did,{'revision':1,'confirmation_name':self.dataset['name']})
        other_image=self.images.import_image(self.other_project['id'],self.other_dataset['id'],'B.png',sample())['image']
        for dataset_id in [self.did,self.other_dataset['id']]:
            with self.assertRaises(sqlite3.IntegrityError):
                with self.store.connect() as db:db.execute('INSERT INTO task_images VALUES(?,?,?,?)',(empty_task['id'],other_image['id'],dataset_id,'now'))

    def test_failed_task_creation_rolls_back_task_and_links(self):
        self.images.import_image(self.pid,self.did,'A.png',sample())
        with self.store.connect() as db:db.execute("CREATE TRIGGER fail_task BEFORE INSERT ON task_images BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.create()
        self.assertEqual(self.tasks.list(self.pid),[])
        self.assertEqual(self.store.get(self.pid)['task_count'],0)

    def test_task_and_image_binding_survive_reopen_and_paginate(self):
        for index,color in enumerate(['red','blue','green']):self.images.import_image(self.pid,self.did,f'{index}.png',sample(color))
        task=self.create();restored=TaskRepository(Storage(self.path))
        self.assertEqual(restored.get(self.pid,task['id']),task)
        page=restored.images(self.pid,task['id'],2,1);self.assertEqual((len(page['images']),page['total'],page['pages']),(1,3,3))


class TaskHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.server=create_server(0,Path(cls.temp.name)/'http.db')
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'
        cls.project=cls.server.storage.create({'name':'接口任务项目'})
        cls.dataset=cls.server.storage.create_dataset(cls.project['id'],{'name':'原图'})
        cls.collection=f"/api/projects/{cls.project['id']}/tasks"

    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join(3);cls.temp.cleanup()

    def call(self,path,method='GET',body=None):
        request=Request(self.base+path,method=method,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
        try:response=urlopen(request,timeout=5)
        except HTTPError as error:response=error
        with response:return response.status,json.loads(response.read())

    def test_http_task_lifecycle_and_source_cannot_be_rebound(self):
        status,data=self.call(self.collection,'POST',{'name':'HTTP 检测','task_type':'detect','dataset_id':self.dataset['id']})
        self.assertEqual(status,201);task=data['task'];path=self.collection+'/'+task['id']
        self.assertEqual(self.call(path)[1]['task'],task)
        self.assertEqual(self.call(path,'PUT',{'name':'不能改源','revision':1,'dataset_id':'0'*32})[0],422)
        self.assertEqual(self.call(path,'PUT',{'name':'不能改类型','revision':1,'task_type':'segment'})[0],422)
        self.assertEqual(self.call(path+'/images')[1]['total'],0)
        self.assertEqual(self.call(path+'/images?page=bad')[0],422)
        self.assertEqual(self.call(path+'/sync-images','POST',{'revision':1})[1]['added_count'],0)
        self.assertEqual(self.call(path,'PUT',{'name':'HTTP 修改','description':'已保存','revision':1})[0],200)
        self.assertEqual(self.call(path,'DELETE',{'revision':2,'confirmation_name':'HTTP 修改'})[0],200)
        self.assertEqual(self.call(path)[0],404)

    def test_http_unsupported_type_or_missing_source_does_not_create_task(self):
        self.assertEqual(self.call(self.collection,'POST',{'name':'不支持','task_type':'classify','dataset_id':self.dataset['id']})[0],422)
        self.assertEqual(self.call(self.collection,'POST',{'name':'无源','task_type':'detect'})[0],422)


class TaskMigrationTests(unittest.TestCase):
    def test_v4_upgrade_preserves_all_existing_rows_and_rolls_back_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'v4.db'
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE schema_version(version INTEGER PRIMARY KEY)');db.execute('INSERT INTO schema_version VALUES(4)')
                Storage.create_project_table(db,'projects');Storage.create_dataset_table(db);Storage.create_image_table(db)
                db.execute('INSERT INTO projects VALUES(?,?,?,?,?,?,?)',('a'*32,'项目','项目','说明',3,'date','date'))
                db.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?)',('b'*32,'a'*32,'数据集','数据集','说明',5,'date','date'))
                db.commit();before={table:db.execute('SELECT * FROM '+table).fetchall() for table in ['projects','datasets','images']}
            class FailingStorage(Storage):
                @staticmethod
                def create_task_tables(db):Storage.create_task_tables(db);raise RuntimeError('Injected task migration failure')
            with self.assertRaises(RuntimeError):FailingStorage(path)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],4)
                self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='vision_tasks'").fetchone())
            store=Storage(path)
            with store.connect() as db:
                for table,records in before.items():self.assertEqual([tuple(row) for row in db.execute('SELECT * FROM '+table)],records)
            self.assertEqual(store.get('a'*32)['task_count'],0)
            self.assertTrue(list((Path(directory)/'backups').glob('schema-v4-*.db')))


if __name__=='__main__':unittest.main()
