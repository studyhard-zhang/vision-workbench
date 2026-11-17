import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import zipfile
from unittest.mock import patch
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from test_versions import VersionFixture
from app.storage import Storage,ProjectError
from app.maintenance import MaintenanceRepository
from app.training import TrainingRepository
from app.server import create_server


class MaintenanceTests(VersionFixture):
    def setUp(self):
        super().setUp();self.publish()
        self.maintenance=MaintenanceRepository(self.store)
        self.maintenance.restores=self.path.parent/'isolated-restores';self.maintenance.restores.mkdir()

    def backup_path(self):
        result=self.maintenance.backup(self.pid)
        return self.maintenance.archive(result['id'])[0]

    def snapshot(self):
        with self.store.connect() as db:
            return {r[0]:[tuple(x) for x in db.execute('SELECT * FROM "'+r[0]+'"')] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}

    def test_project_scope_fresh_directory_preserves_ids_hashes_and_source(self):
        other=self.store.create({'name':'Other'})
        before=self.snapshot();source=self.backup_path()
        with zipfile.ZipFile(source) as z:
            self.assertNotIn('data/ssh-target.json',z.namelist())
            rows=json.loads(z.read('snapshot.json'))['tables']
            self.assertEqual([r['id'] for r in rows['projects']],[self.pid])
            self.assertNotIn(other['id'],z.read('snapshot.json').decode())
        result=self.maintenance.restore(source)
        self.assertEqual(before,self.snapshot())
        restored=Storage(Path(result['data_directory'])/'platform.db')
        with restored.connect() as db:
            self.assertFalse(db.execute('PRAGMA foreign_key_check').fetchall())
            for name,records in rows.items():self.assertEqual([dict(r) for r in db.execute('SELECT * FROM "'+name+'"')],records)
        self.assertFalse((Path(result['data_directory'])/'ssh-target.json').exists())
        again=self.maintenance.restore(source)
        self.assertNotEqual(again['directory'],result['directory'])

    def test_missing_or_changed_source_rejected(self):
        p=self.versions.assets.owned_path(self.did,self.ids[0],'original.png');p.write_bytes(b'bad')
        with self.assertRaises(ProjectError) as error:self.backup_path()
        self.assertEqual(error.exception.code,'BACKUP_SOURCE_DAMAGED')
        self.assertFalse(list(self.maintenance.archives.glob('*.zip')))

    def test_active_queued_job_blocks_backup_and_cleanup(self):
        version=self.versions.list(self.pid,self.a)[0]
        training=TrainingRepository(self.store,self.versions)
        import uuid
        training.create(self.pid,self.a,{'version_id':version['id'],'name':'Waiting','target_id':training.target()['id'],'epochs':1,'idempotency_key':uuid.uuid4().hex})
        for action in (lambda:self.backup_path(),self.maintenance.cleanup):
            with self.assertRaises(ProjectError) as error:action()
            self.assertEqual(error.exception.code,'MAINTENANCE_ACTIVE_JOB')

    def test_archive_corruption_extra_path_duplicate_and_missing_rejected_atomically(self):
        source=self.backup_path()
        with zipfile.ZipFile(source) as z:files={n:z.read(n) for n in z.namelist()}
        for mode in ('corrupt','extra','traversal','missing','duplicate'):
            path=self.path.parent/(mode+'.zip');contents=files.copy()
            if mode=='corrupt':contents['snapshot.json']=b'bad'
            if mode=='extra':contents['data/ssh-target.json']=b'secret'
            if mode=='traversal':contents['../outside']=b'bad'
            if mode=='missing':contents.pop('snapshot.json')
            with zipfile.ZipFile(path,'w') as z:
                for name,raw in contents.items():z.writestr(name,raw)
                if mode=='duplicate':z.writestr('snapshot.json',files['snapshot.json'])
            with self.subTest(mode=mode),self.assertRaises(ProjectError):self.maintenance.restore(path)
            self.assertFalse(list(self.maintenance.restores.iterdir()))

    def test_space_and_write_failure_leave_no_published_archive(self):
        usage=type('Usage',(),{'free':0})()
        with patch('app.maintenance.shutil.disk_usage',return_value=usage),self.assertRaises(ProjectError) as error:self.backup_path()
        self.assertEqual(error.exception.status,507)
        with patch('app.maintenance.zipfile.ZipFile.write',side_effect=OSError(28,'full')),self.assertRaises(ProjectError) as error:self.backup_path()
        self.assertEqual(error.exception.status,507)
        self.assertFalse(list(self.maintenance.archives.iterdir()))

    def test_rehashed_broken_foreign_key_and_restore_disk_failure_rollback(self):
        source=self.backup_path()
        with zipfile.ZipFile(source) as z:files={n:z.read(n) for n in z.namelist()}
        rows=json.loads(files['snapshot.json']);rows['tables']['images'][0]['dataset_id']='0'*32
        files['snapshot.json']=json.dumps(rows).encode();m=json.loads(files['manifest.json'])
        m['files']['snapshot.json']={'bytes':len(files['snapshot.json']),'sha256':hashlib.sha256(files['snapshot.json']).hexdigest()}
        files['manifest.json']=json.dumps(m).encode();broken=self.path.parent/'broken-reference.zip'
        with zipfile.ZipFile(broken,'w') as z:
            for name,raw in files.items():z.writestr(name,raw)
        with self.assertRaises(ProjectError) as error:self.maintenance.restore(broken)
        self.assertEqual(error.exception.code,'INVALID_PROJECT_BACKUP')
        self.assertFalse(list(self.maintenance.restores.iterdir()))
        with patch('app.maintenance.shutil.copyfileobj',side_effect=OSError(28,'disk full')),self.assertRaises(ProjectError) as error:self.maintenance.restore(source)
        self.assertEqual(error.exception.status,507)
        self.assertFalse(list(self.maintenance.restores.iterdir()))

    def test_cleanup_only_old_staging_files_and_impact_protection(self):
        stage=self.maintenance.root/'assets/.staging';old=stage/'orphan';recent=stage/'active'
        old.write_bytes(b'old');recent.write_bytes(b'recent');os.utime(old,(time.time()-90000,)*2)
        original=self.versions.assets.owned_path(self.did,self.ids[0],'original.png');raw=original.read_bytes()
        result=self.maintenance.cleanup();self.assertEqual(result,{'files':1,'released_bytes':3})
        self.assertTrue(recent.exists());self.assertEqual(raw,original.read_bytes())
        impact=self.maintenance.impact(self.pid);self.assertFalse(impact['permanent_delete_allowed']);self.assertGreater(impact['counts']['task_data_versions'],0)

    def test_http_backup_restore_and_write_gate(self):
        server=create_server(0,self.path);server.maintenance.restores=self.maintenance.restores
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        def post(path,body=b'{}',mime='application/json'):
            return urlopen(Request(base+path,data=body,headers={'Content-Type':mime}),timeout=20)
        try:
            with post(f'/api/projects/{self.pid}/backup') as response:record=json.load(response)['backup']
            with urlopen(base+'/api/maintenance/backups/'+record['id']) as response:raw=response.read()
            with post('/api/maintenance/restore',raw,'application/zip') as response:self.assertTrue(Path(json.load(response)['restore']['directory']).is_dir())
            server.mutation_lock.acquire()
            try:
                with self.assertRaises(HTTPError) as error:post('/api/projects',b'{"name":"blocked"}')
                self.assertEqual(error.exception.code,409)
            finally:server.mutation_lock.release()
        finally:server.shutdown();server.server_close();thread.join()
