import errno
import io
import json
import sqlite3
import threading
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from test_preparation import PreparationFixture
from app.storage import Storage,ProjectError
from app.versions import VersionRepository
from app.server import create_server
from PIL import Image


class VersionFixture(PreparationFixture):
    def setUp(self):
        super().setUp();self.versions=VersionRepository(self.store)

    def publish(self, data=None):
        data=data or self.prep.read(self.pid,self.a)
        if not data['split']:data=self.generate(train_percent=80,val_percent=20,test_percent=0)
        return self.versions.publish(self.pid,self.a,{'fingerprint':data['report']['fingerprint'],'split_revision':data['split']['revision']})

    def detail(self, result):return self.versions.read(self.pid,self.a,result['version']['id'])


class VersionTests(VersionFixture):
    def test_publish_is_persistent_sequential_and_retry_does_not_duplicate_or_touch_task(self):
        first=self.publish();task=self.tasks.get(self.pid,self.a);again=self.publish()
        self.assertFalse(first['reused']);self.assertTrue(again['reused']);self.assertEqual(first['version'],again['version'])
        self.assertEqual(self.tasks.get(self.pid,self.a),task)
        self.assertEqual(first['version']['number'],1)
        restored=VersionRepository(Storage(self.path)).read(self.pid,self.a,first['version']['id'])
        self.assertTrue(restored['integrity']['ok']);self.assertEqual(restored,self.detail(first))
        next_version=self.publish(self.generate(seed=99))
        self.assertEqual(next_version['version']['number'],2)
        self.assertEqual([v['number'] for v in self.versions.list(self.pid,self.a)],[2,1])

    def test_partial_subset_only_and_yolo_bundle_uses_unique_names_and_empty_negative_label(self):
        self.repo.save(self.pid,self.a,self.ids[0],{'state':'unannotated','revision':2,'boxes':[]})
        self.repo.save(self.pid,self.a,self.ids[1],{'state':'no_target','revision':2,'boxes':[]})
        result=self.publish();detail=self.detail(result);manifest=detail['manifest']
        self.assertNotIn(self.ids[0],{i['id'] for i in manifest['images']})
        self.assertEqual(len(manifest['images']),len(self.ids)-1)
        with self.versions.archive(self.pid,self.a,result['version']['id']) as (stream,size,name):
            raw=stream.read();self.assertEqual(size,len(raw));self.assertTrue(name.endswith('.zip'))
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            config=json.loads(archive.read('data.yaml'));self.assertEqual(config['names'],['划痕','凹陷'])
            self.assertNotIn('path',config);self.assertNotIn('download',config);self.assertNotIn('test',config)
            self.assertEqual(json.loads(archive.read('manifest.json')),manifest)
            self.assertEqual(len(archive.namelist()),len(manifest['images'])*2+2)
            for image in manifest['images']:
                self.assertTrue(image['image_path'].startswith('images/'+image['subset']+'/'))
                label=archive.read(image['label_path']).decode()
                if image['state']=='no_target':self.assertEqual(label,'')
                else:
                    self.assertEqual(label,'0 0.0625 0.175 0.125 0.25\n1 0.3125 0.175 0.125 0.25\n')
                    for line in label.splitlines():self.assertTrue(all(0<float(n)<=1 for n in line.split()[1:]))
                with Image.open(io.BytesIO(archive.read(image['image_path']))) as picture:
                    self.assertEqual(picture.size,(image['width'],image['height']))
        self.assertEqual(list(self.versions.staging.iterdir()),[])

    def test_workspace_edits_and_another_task_never_rewrite_frozen_snapshot(self):
        result=self.publish();before=self.detail(result);vid=result['version']['id']
        raw=(self.versions.root/vid/'manifest.json').read_bytes()
        self.repo.change_class(self.pid,self.a,{'name':'新类别名称','revision':1},self.cids[0])
        self.repo.save(self.pid,self.a,self.ids[0],{'state':'excluded','revision':2,'boxes':[]})
        self.repo_images.update_group(self.pid,self.did,self.ids[1],{'group_id':'新采集组','revision':1})
        task=self.tasks.get(self.pid,self.a);self.tasks.update(self.pid,self.a,{'name':'改名任务','revision':task['revision']})
        foreign=self.add(self.b,name='另一任务类别')
        self.repo.save(self.pid,self.b,self.ids[0],{'state':'annotated','revision':1,'boxes':[{'class_id':foreign,'x':0,'y':0,'width':10,'height':10}]})
        self.assertEqual(self.detail(result),before);self.assertEqual((self.versions.root/vid/'manifest.json').read_bytes(),raw)
        new=self.detail(self.publish(self.generate()))
        self.assertEqual(new['manifest']['classes'][0]['name'],'新类别名称')
        self.assertEqual(new['manifest']['task_name'],'改名任务')
        self.assertNotIn(self.ids[0],{i['id'] for i in new['manifest']['images']})

    def test_deleted_workspace_class_keeps_old_mapping_and_new_mapping_is_contiguous(self):
        original=self.publish();old=self.detail(original)
        for iid in self.ids:
            current=self.repo.get(self.pid,self.a,iid)
            self.repo.save(self.pid,self.a,iid,{'state':'annotated','revision':current['revision'],'boxes':[current['boxes'][1]]})
        self.repo.change_class(self.pid,self.a,{'revision':1,'confirmation_name':'划痕'},self.cids[0],True)
        new=self.detail(self.publish(self.generate()))
        self.assertEqual(new['manifest']['classes'],[{'id':self.cids[1],'name':'凹陷','index':0}])
        self.assertEqual(self.detail(original),old)

    def test_exif_image_bundle_uses_canonical_pixels_and_matching_full_image_box(self):
        stream=io.BytesIO();picture=Image.new('RGB',(40,20),'purple');exif=picture.getexif();exif[274]=6;picture.save(stream,'JPEG',exif=exif)
        iid=self.repo_images.import_image(self.pid,self.did,'rotated.jpg',stream.getvalue())['image']['id']
        task=self.tasks.get(self.pid,self.a);self.tasks.sync(self.pid,self.a,{'revision':task['revision']})
        self.repo.save(self.pid,self.a,iid,{'state':'annotated','revision':1,'boxes':[{'class_id':self.cids[0],'x':0,'y':0,'width':20,'height':40}]})
        result=self.publish();manifest=self.detail(result)['manifest'];image=next(i for i in manifest['images'] if i['id']==iid)
        self.assertEqual((image['width'],image['height']),(20,40))
        with self.versions.archive(self.pid,self.a,result['version']['id']) as (stream,_,_):
            with zipfile.ZipFile(stream) as archive:
                self.assertEqual(archive.read(image['label_path']).decode(),'0 0.5 0.5 1 1\n')
                with Image.open(io.BytesIO(archive.read(image['image_path']))) as canonical:
                    self.assertEqual(canonical.size,(20,40));self.assertIn(canonical.getexif().get(274),(None,1))
        original,_,_=self.repo_images.file(self.pid,self.did,iid,'original')
        with Image.open(original) as original_image:self.assertEqual(original_image.size,(40,20));self.assertEqual(original_image.getexif().get(274),6)

    def test_missing_stale_or_incomplete_split_never_publishes(self):
        report=self.prep.read(self.pid,self.a)['report']
        with self.assertRaises(ProjectError) as error:self.versions.publish(self.pid,self.a,{'fingerprint':report['fingerprint'],'split_revision':1})
        self.assertEqual(error.exception.code,'SPLIT_REQUIRED')
        plan=self.generate();self.repo.save(self.pid,self.a,self.ids[0],{'state':'no_target','revision':2,'boxes':[]})
        with self.assertRaises(ProjectError) as error:self.publish(plan)
        self.assertEqual(error.exception.code,'SPLIT_STALE')
        self.generate()
        with self.store.connect() as db:db.execute('DELETE FROM task_split_images WHERE task_id=? AND image_id=?',(self.a,self.ids[1]))
        with self.assertRaises(ProjectError) as error:self.publish()
        self.assertEqual(error.exception.code,'SPLIT_INVALID');self.assertEqual(self.versions.list(self.pid,self.a),[])

    def test_database_insert_failure_rolls_back_files_rows_and_parent_revision(self):
        self.generate();task=self.tasks.get(self.pid,self.a)
        with self.store.connect() as db:db.execute("CREATE TRIGGER fail_version BEFORE INSERT ON task_data_version_images BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.publish()
        self.assertEqual(self.versions.list(self.pid,self.a),[]);self.assertEqual(self.tasks.get(self.pid,self.a),task)
        self.assertEqual([p.name for p in self.versions.root.iterdir()],['.staging']);self.assertEqual(list(self.versions.staging.iterdir()),[])

    def test_disk_full_and_write_failure_leave_no_version_and_release_operation_slot(self):
        self.generate()
        with patch('app.versions.shutil.disk_usage') as usage:
            usage.return_value.free=0
            with self.assertRaises(ProjectError) as error:self.publish()
            self.assertEqual(error.exception.status,507)
        with patch('app.versions.encoded',side_effect=OSError(errno.ENOSPC,'injected')):
            with self.assertRaises(ProjectError) as error:self.publish()
            self.assertEqual(error.exception.code,'VERSION_WRITE_FAILED')
        self.assertEqual(self.versions.list(self.pid,self.a),[])
        self.assertEqual([p.name for p in self.versions.root.iterdir()],['.staging'])
        self.assertFalse(self.publish()['reused'])

    def test_partial_stage_write_and_publish_rename_failures_are_cleaned_up(self):
        self.generate();original=Path.write_bytes
        def fail_label(path,data):
            if path.suffix=='.txt':raise OSError(errno.ENOSPC,'injected')
            return original(path,data)
        with patch.object(Path,'write_bytes',fail_label):
            with self.assertRaises(ProjectError):self.publish()
        with patch.object(Path,'rename',side_effect=OSError(errno.EACCES,'injected')):
            with self.assertRaises(ProjectError):self.publish()
        self.assertEqual(self.versions.list(self.pid,self.a),[]);self.assertEqual(list(self.versions.staging.iterdir()),[])

    def test_damage_is_visible_and_export_never_rebuilds_from_current_workspace(self):
        result=self.publish();detail=self.detail(result);image=detail['manifest']['images'][0]
        label=self.versions.root/result['version']['id']/image['label_path'];label.write_text('damaged',encoding='utf-8')
        self.assertFalse(self.detail(result)['integrity']['ok'])
        with self.assertRaises(ProjectError) as error:
            with self.versions.archive(self.pid,self.a,result['version']['id']):pass
        self.assertEqual(error.exception.code,'VERSION_DAMAGED');self.assertEqual(label.read_text(),'damaged')
        original=self.versions.source_path(detail['manifest'],image,'original');original.unlink()
        self.assertGreaterEqual(self.detail(result)['integrity']['issue_count'],2)

    def test_changed_image_is_rejected_in_preview_and_archive(self):
        result=self.publish();detail=self.detail(result);image=detail['manifest']['images'][0]
        path=self.versions.source_path(detail['manifest'],image,'view');path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(ProjectError):self.versions.image(self.pid,self.a,result['version']['id'],image['id'])
        with self.assertRaises(ProjectError):
            with self.versions.archive(self.pid,self.a,result['version']['id']):pass

    def test_published_rows_are_immutable_and_task_delete_is_protected_without_side_effects(self):
        result=self.publish();task=self.tasks.get(self.pid,self.a);before=self.detail(result)
        for sql in ['UPDATE task_data_versions SET number=99','DELETE FROM task_data_versions','DELETE FROM task_data_version_images']:
            with self.assertRaises(sqlite3.IntegrityError):
                with self.store.connect() as db:db.execute(sql)
        with self.assertRaises(ProjectError) as error:self.tasks.delete(self.pid,self.a,{'revision':task['revision'],'confirmation_name':task['name']})
        self.assertEqual(error.exception.code,'TASK_HAS_VERSIONS');self.assertEqual(self.tasks.get(self.pid,self.a),task);self.assertEqual(self.detail(result),before)

    def test_project_task_version_and_image_scope_is_enforced(self):
        result=self.publish();vid=result['version']['id'];other=self.store.create({'name':'其他项目'})['id']
        self.assertEqual(self.versions.list(self.pid,self.b),[])
        for pid,tid in [(other,self.a),(self.pid,self.b)]:
            with self.assertRaises(ProjectError):self.versions.read(pid,tid,vid)
            with self.assertRaises(ProjectError):self.versions.image(pid,tid,vid,self.ids[0])
            with self.assertRaises(ProjectError):
                with self.versions.archive(pid,tid,vid):pass
        with self.assertRaises(ProjectError):self.versions.image(self.pid,self.a,vid,'0'*32)

    def test_http_publish_read_preview_archive_unknown_fields_and_no_update_or_delete(self):
        plan=self.generate();server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}/api/projects/{self.pid}/tasks/{self.a}/versions'
        def call(path='',method='GET',body=None):
            req=Request(base+path,method=method,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
            try:response=urlopen(req,timeout=10)
            except HTTPError as error:response=error
            with response:return response.status,response.read(),response.headers
        try:
            body={'fingerprint':plan['report']['fingerprint'],'split_revision':plan['split']['revision']}
            status,raw,_=call(method='POST',body=body);self.assertEqual(status,201);vid=json.loads(raw)['version']['id']
            self.assertEqual(call(method='POST',body=body)[0],200)
            self.assertEqual(call(method='POST',body={**body,'path':'evil'})[0],422)
            status,raw,_=call('/'+vid);self.assertEqual(status,200);detail=json.loads(raw);self.assertTrue(detail['integrity']['ok'])
            self.assertEqual(call('/'+vid+'/images/'+detail['manifest']['images'][0]['id'])[0],200)
            status,raw,headers=call('/'+vid+'/archive');self.assertEqual(status,200);self.assertEqual(headers['Content-Type'],'application/zip')
            self.assertEqual(int(headers['Content-Length']),len(raw));self.assertTrue(zipfile.is_zipfile(io.BytesIO(raw)))
            self.assertEqual(call('/'+vid,'PUT',{})[0],404);self.assertEqual(call('/'+vid,'DELETE',{})[0],404)
        finally:server.shutdown();server.server_close();thread.join(3)


class VersionMigrationTests(VersionFixture):
    def test_v7_upgrade_preserves_all_business_rows_and_failure_rolls_back(self):
        self.generate()
        with self.store.connect() as db:
            __import__('schema_utils').drop_evaluations(db);db.execute('DROP TABLE training_record_visibility');db.execute('DROP TABLE model_events');db.execute('DROP TABLE model_versions');db.execute('DROP TABLE training_remote_sessions');db.execute('DROP TABLE training_dispatches');db.execute('DROP TABLE training_jobs');db.execute('DROP INDEX version_task_identity');db.execute('DROP TABLE task_data_version_images');db.execute('DROP TABLE task_data_versions');db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=7')
            tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name!='schema_version'")]
            before={t:[tuple(r) for r in db.execute('SELECT * FROM '+t)] for t in tables}
        class FailingStorage(Storage):
            @staticmethod
            def create_version_tables(db):Storage.create_version_tables(db);raise RuntimeError('injected')
        with self.assertRaises(RuntimeError):FailingStorage(self.path)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],7)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='task_data_versions'").fetchone())
        restored=Storage(self.path)
        with restored.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16)
            for table,rows in before.items():self.assertEqual([tuple(r) for r in db.execute('SELECT * FROM '+table)],rows)
        self.assertTrue(list((self.path.parent/'backups').glob('schema-v7-*.db')))
