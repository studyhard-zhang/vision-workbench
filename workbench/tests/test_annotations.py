import io
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.storage import Storage, ProjectError
from app.tasks import TaskRepository
from app.annotations import AnnotationRepository
from app.images import ImageRepository
from app.server import create_server
from PIL import Image


class AnnotationFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'work.db'
        self.store = Storage(self.path); self.repo = AnnotationRepository(self.store); self.tasks = TaskRepository(self.store)
        self.pid = self.store.create({'name':'项目'})['id']
        self.did = self.store.create_dataset(self.pid,{'name':'原图'})['id']
        stream=io.BytesIO(); Image.new('RGB',(40,20),'red').save(stream,'PNG')
        self.image = ImageRepository(self.store).import_image(self.pid,self.did,'A.png',stream.getvalue())['image']['id']
        self.a=self.tasks.create(self.pid,{'name':'A','task_type':'detect','dataset_id':self.did})['id']
        self.b=self.tasks.create(self.pid,{'name':'B','task_type':'detect','dataset_id':self.did})['id']

    def tearDown(self): self.temp.cleanup()

    def add(self, task=None, name='划痕'):
        return self.repo.change_class(self.pid,task or self.a,{'name':name})['id']

    def save(self, state='no_target', revision=1, boxes=None, task=None):
        return self.repo.save(self.pid,task or self.a,self.image,{'state':state,'revision':revision,'boxes':boxes or []})

    def box(self, cid): return {'class_id':cid,'x':0,'y':0,'width':40,'height':20}


class AnnotationBehaviorTests(AnnotationFixture):
    def test_class_scope_normalization_validation_and_persistence(self):
        cid=self.add(name=' Camera '); self.add(self.b,name='Camera')
        with self.assertRaises(ProjectError):self.add(name='ＣＡＭＥＲＡ')
        for name in ['', 'x'*61, 'a\nb']:
            with self.assertRaises(ProjectError):self.add(name=name)
        self.repo.change_class(self.pid,self.a,{'name':'划痕','revision':1},cid)
        restored=AnnotationRepository(Storage(self.path)).classes(self.pid,self.a)
        self.assertEqual((restored[0]['name'],restored[0]['revision']),('划痕',2))
        self.assertEqual(self.repo.classes(self.pid,self.b)[0]['name'],'Camera')

    def test_class_conflict_and_confirmation_preserve_saved_class(self):
        cid=self.add()
        self.repo.change_class(self.pid,self.a,{'name':'新名','revision':1},cid)
        with self.assertRaises(ProjectError):self.repo.change_class(self.pid,self.a,{'name':'旧页面','revision':1},cid)
        with self.assertRaises(ProjectError):self.repo.change_class(self.pid,self.a,{'confirmation_name':'错误','revision':2},cid,True)
        self.repo.change_class(self.pid,self.a,{'confirmation_name':'新名','revision':2},cid,True)
        self.assertEqual(self.repo.classes(self.pid,self.a),[])

    def test_default_is_unannotated_and_state_is_explicit_task_scoped_and_durable(self):
        original=self.repo.get(self.pid,self.b,self.image)
        self.assertEqual(original['state'],'unannotated')
        self.save(); self.assertEqual(self.repo.get(self.pid,self.b,self.image),original)
        self.assertEqual(AnnotationRepository(Storage(self.path)).get(self.pid,self.a,self.image)['state'],'no_target')
        self.save('excluded',2); self.save('unannotated',3)
        self.assertEqual(self.tasks.get(self.pid,self.a)['no_target_count'],0)
        self.assertEqual(self.tasks.images(self.pid,self.a)['images'][0]['annotation_state'],'unannotated')

    def test_stale_state_write_and_stale_task_delete_are_rejected(self):
        task=self.tasks.get(self.pid,self.a);self.save()
        with self.assertRaises(ProjectError) as error:self.save('excluded')
        self.assertEqual(error.exception.code,'REVISION_CONFLICT')
        with self.assertRaises(ProjectError):self.tasks.delete(self.pid,self.a,{'revision':task['revision'],'confirmation_name':'A'})
        self.assertEqual(self.repo.get(self.pid,self.a,self.image)['state'],'no_target')

    def test_annotated_requires_boxes_and_valid_scoped_class_and_geometry(self):
        cid=self.add(); foreign=self.add(self.b)
        bad=[[],[self.box(foreign)],[dict(self.box(cid),width=41)],[dict(self.box(cid),width=0)],
             [dict(self.box(cid),x=float('nan'))],[dict(self.box(cid),x=True)],[dict(self.box(cid),width=-1)],
             [dict(self.box(cid),width=10**400)],
             [dict(self.box(cid),extra=0)]]
        for boxes in bad:
            with self.assertRaises(ProjectError):self.save('annotated',boxes=boxes)
        for state in ['no_target','unannotated']:
            with self.assertRaises(ProjectError):self.save(state,boxes=[self.box(cid)])
        self.save('annotated',boxes=[self.box(cid)])
        self.assertEqual(self.tasks.get(self.pid,self.a)['annotated_count'],1)
        self.assertEqual(self.repo.get(self.pid,self.b,self.image)['boxes'],[])

    def test_excluding_keeps_boxes_and_returning_to_unannotated_removes_them(self):
        cid=self.add();boxes=[self.box(cid)]
        self.save('annotated',boxes=boxes)
        self.save('excluded',2,boxes)
        self.assertEqual(self.repo.get(self.pid,self.a,self.image)['boxes'],boxes)
        self.save('unannotated',3)
        self.assertEqual(self.repo.get(self.pid,self.a,self.image)['boxes'],[])

    def test_invalid_revisions_states_colors_and_counts_do_not_write(self):
        for revision in [True,0,'1',None]:
            with self.assertRaises(ProjectError):self.save(revision=revision)
        for state in ['done',None,[]]:
            with self.assertRaises(ProjectError):self.save(state)
        with self.assertRaises(ProjectError):self.repo.change_class(self.pid,self.a,{'name':'X','color':'red'})
        with self.assertRaises(ProjectError):self.save('annotated',boxes=[self.box('0'*32)]*201)
        self.assertEqual(self.repo.get(self.pid,self.a,self.image)['revision'],1)

    def test_referenced_class_is_protected_and_rename_keeps_box_reference(self):
        cid=self.add();self.save('annotated',boxes=[self.box(cid)])
        with self.assertRaises(ProjectError) as error:self.repo.change_class(self.pid,self.a,{'revision':1,'confirmation_name':'划痕'},cid,True)
        self.assertEqual(error.exception.code,'CLASS_IN_USE')
        self.repo.change_class(self.pid,self.a,{'name':'缺陷','revision':1},cid)
        self.assertEqual(self.repo.get(self.pid,self.a,self.image)['boxes'][0]['class_id'],cid)
        self.save('unannotated',2)
        self.repo.change_class(self.pid,self.a,{'revision':2,'confirmation_name':'缺陷'},cid,True)

    def test_task_deletion_cleans_only_its_workspace_and_keeps_assets(self):
        self.add();self.add(self.b);self.save();self.save('excluded',task=self.b)
        files={str(p):p.read_bytes() for p in (self.path.parent/'assets').rglob('*') if p.is_file()}
        task=self.tasks.get(self.pid,self.a)
        self.tasks.delete(self.pid,self.a,{'revision':task['revision'],'confirmation_name':'A'})
        self.assertEqual(self.repo.get(self.pid,self.b,self.image)['state'],'excluded')
        self.assertEqual(len(self.repo.classes(self.pid,self.b)),1)
        self.assertEqual(files,{str(p):p.read_bytes() for p in (self.path.parent/'assets').rglob('*') if p.is_file()})

    def test_cross_project_task_and_unlinked_image_access_is_rejected(self):
        other=self.store.create({'name':'另项目'})['id'];cid=self.add()
        for fn in [lambda:self.repo.classes(other,self.a),lambda:self.repo.get(other,self.a,self.image),
                   lambda:self.repo.change_class(self.pid,self.b,{'name':'错改','revision':1},cid),
                   lambda:self.repo.get(self.pid,self.a,'0'*32)]:
            with self.assertRaises(ProjectError):fn()
        stream=io.BytesIO();Image.new('RGB',(5,5),'blue').save(stream,'PNG')
        new=ImageRepository(self.store).import_image(self.pid,self.did,'new.png',stream.getvalue())['image']['id']
        with self.assertRaises(ProjectError):self.repo.get(self.pid,self.a,new)
        task=self.tasks.get(self.pid,self.a);self.tasks.sync(self.pid,self.a,{'revision':task['revision']})
        self.assertEqual(self.repo.get(self.pid,self.a,new)['state'],'unannotated')
        with self.assertRaises(ProjectError):self.repo.get(self.pid,self.b,new)

    def test_failed_annotation_write_rolls_back_revision_and_boxes(self):
        cid=self.add();self.save('annotated',boxes=[self.box(cid)])
        before=self.repo.get(self.pid,self.a,self.image);task=self.tasks.get(self.pid,self.a)
        with self.store.connect() as db:db.execute("CREATE TRIGGER fail_box BEFORE INSERT ON detection_boxes BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.save('annotated',2,[dict(self.box(cid),width=10)])
        self.assertEqual(self.repo.get(self.pid,self.a,self.image),before)
        self.assertEqual(self.tasks.get(self.pid,self.a),task)

    def test_foreign_key_rejects_cross_task_class_in_box(self):
        foreign=self.add(self.b);self.save()
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.connect() as db:db.execute('INSERT INTO detection_boxes VALUES(?,?,?,?,?,?,?,?)',(self.a,self.image,0,foreign,0,0,10,10))


class AnnotationMigrationTests(unittest.TestCase):
    def test_v5_upgrade_preserves_rows_and_backup_and_failed_upgrade_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'v5.db'
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE schema_version(version INTEGER PRIMARY KEY)');db.execute('INSERT INTO schema_version VALUES(5)')
                Storage.create_project_table(db,'projects');Storage.create_dataset_table(db);Storage.create_image_table(db);Storage.create_task_tables(db)
                db.execute('INSERT INTO projects VALUES(?,?,?,?,?,?,?)',('a'*32,'项目','项目','说明',3,'date','date'))
                db.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?)',('b'*32,'a'*32,'原图','原图','',1,'date','date'))
                db.execute('INSERT INTO vision_tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)',('c'*32,'a'*32,'b'*32,'任务','任务','','detect',2,'date','date','date'))
                db.commit();before={t:db.execute('SELECT * FROM '+t).fetchall() for t in ['projects','datasets','images','vision_tasks','task_images']}
            class FailingStorage(Storage):
                @staticmethod
                def create_annotation_tables(db):Storage.create_annotation_tables(db);raise RuntimeError('injected')
            with self.assertRaises(RuntimeError):FailingStorage(path)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],5)
                self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='task_classes'").fetchone())
            store=Storage(path)
            with store.connect() as db:
                for table,rows in before.items():self.assertEqual([tuple(row) for row in db.execute('SELECT * FROM '+table)],rows)
                self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16)
            self.assertEqual(AnnotationRepository(store).classes('a'*32,'c'*32),[])
            self.assertTrue(list((path.parent/'backups').glob('schema-v5-*.db')))


class AnnotationHttpTests(unittest.TestCase):
    def test_http_classes_and_annotation_lifecycle_and_invalid_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            server=create_server(0,Path(directory)/'http.db');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            def call(path,method='GET',body=None):
                req=Request(f'http://127.0.0.1:{server.server_port}'+path,method=method,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
                try:response=urlopen(req,timeout=5)
                except HTTPError as error:response=error
                with response:return response.status,json.loads(response.read())
            try:
                pid=server.storage.create({'name':'接口'})['id'];did=server.storage.create_dataset(pid,{'name':'图'})['id']
                stream=io.BytesIO();Image.new('RGB',(4,4),'red').save(stream,'PNG')
                image=server.images.import_image(pid,did,'a.png',stream.getvalue())['image']['id']
                tid=server.tasks.create(pid,{'name':'任务','task_type':'detect','dataset_id':did})['id']
                base=f'/api/projects/{pid}/tasks/{tid}';path=base+f'/images/{image}/annotation'
                status,result=call(base+'/classes','POST',{'name':'缺陷'});self.assertEqual(status,201)
                cid=result['class']['id'];self.assertEqual(len(call(base+'/classes')[1]['classes']),1)
                self.assertEqual(call(base+'/classes/'+cid,'PUT',{'name':'划痕','revision':1})[0],200)
                self.assertEqual(call(path)[1]['annotation']['state'],'unannotated')
                self.assertEqual(call(path,'PUT',{'state':'no_target','boxes':[],'revision':1})[0],200)
                self.assertEqual(call(path,'PUT',{'state':'excluded','boxes':[],'revision':1})[0],409)
                self.assertEqual(call(path,'PUT',{'state':'annotated','boxes':[],'revision':2})[0],422)
                self.assertEqual(call(path,'PUT',{'state':'excluded','boxes':[],'revision':2,'task_id':'x'})[0],422)
                self.assertEqual(call(base+'/classes/'+cid,'DELETE',{'revision':2,'confirmation_name':'划痕'})[0],200)
                self.assertEqual(call('/api/health')[1]['node'],'N-31')
            finally:server.shutdown();server.server_close();thread.join(3)
