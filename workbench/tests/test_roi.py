import hashlib,io,json,tempfile
from pathlib import Path
from unittest import TestCase
from PIL import Image
from test_versions import VersionFixture
from app.roi import save,clip,setting
from app.storage import ProjectError,Storage
from app.maintenance import MaintenanceRepository

class RoiTests(VersionFixture):
    def test_dataset_inference_copies_current_roi_and_remains_independent_after_reset(self):
        from app.inferences import InferenceInputRepository
        save(self.store,self.pid,self.did,{'revision':0,'region':[0,0,.5,1]})
        inputs=InferenceInputRepository(self.store,self.tasks);result=inputs.from_dataset(self.pid,self.a,self.did,self.ids[1]);self.assertEqual(result['input']['width'],20)
        path,_,_=inputs.file(self.pid,self.a,result['input']['id'],'view');before=path.read_bytes()
        save(self.store,self.pid,self.did,{'revision':1,'region':None});self.assertEqual(path.read_bytes(),before)
    def test_schema15_migration_rolls_back_on_failure_and_preserves_existing_rows(self):
        with self.store.connect() as db:
            before=db.execute('SELECT * FROM detection_boxes').fetchall();db.execute('DROP TABLE roi_annotations');db.execute('DROP TABLE dataset_rois');db.execute('UPDATE schema_version SET version=15')
        class Broken(Storage):
            @staticmethod
            def create_roi_tables(db):Storage.create_roi_tables(db);raise RuntimeError('injected migration failure')
        with self.assertRaises(RuntimeError):Broken(self.path)
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],15)
            self.assertFalse(db.execute("SELECT 1 FROM sqlite_master WHERE name='dataset_rois'").fetchall())
        restored=Storage(self.path)
        with restored.connect() as db:self.assertEqual(db.execute('SELECT * FROM detection_boxes').fetchall(),before)
    def test_crop_clips_labels_and_freezes_pixels_without_changing_original_or_old_version(self):
        old=self.publish();old_detail=self.detail(old);image_id=self.ids[1];original,_,_=self.repo_images.file(self.pid,self.did,image_id,'original');before=original.read_bytes()
        with self.store.connect() as db:labels=db.execute('SELECT * FROM detection_boxes').fetchall()
        save(self.store,self.pid,self.did,{'revision':0,'region':[.05,0,.9,1]})
        annotation=self.repo.get(self.pid,self.a,image_id);self.assertTrue(annotation['roi']);self.assertEqual(annotation['boxes'][0]['x'],0)
        path,_,_=self.repo_images.file(self.pid,self.did,image_id,'view')
        with Image.open(path) as picture:self.assertEqual(picture.size,(34,20))
        self.assertEqual(original.read_bytes(),before);self.assertEqual(self.detail(old),old_detail)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT * FROM detection_boxes').fetchall(),labels)
        frozen=self.publish(self.generate(seed=37));detail=self.detail(frozen);self.assertTrue(detail['integrity']['ok']);self.assertEqual(detail['manifest']['images'][0]['width'],34)
        self.assertTrue(self.versions.readiness(self.pid,self.a,frozen['version']['id'])['passed'])
        save(self.store,self.pid,self.did,{'revision':1,'region':None});self.assertTrue(self.detail(frozen)['integrity']['ok'])
    def test_roi_annotation_views_preserve_source_and_reject_stale_revision(self):
        iid=self.ids[1];before=self.repo.get(self.pid,self.a,iid)
        save(self.store,self.pid,self.did,{'revision':0,'region':[0,0,.5,1]});view=self.repo.get(self.pid,self.a,iid)
        result=self.repo.save(self.pid,self.a,iid,{'state':'no_target','boxes':[],'revision':view['revision']});self.assertEqual(result['state'],'no_target')
        with self.assertRaises(ProjectError):self.repo.save(self.pid,self.a,iid,{'state':'no_target','boxes':[],'revision':view['revision']})
        save(self.store,self.pid,self.did,{'revision':1,'region':None});self.assertEqual(self.repo.get(self.pid,self.a,iid)['boxes'],before['boxes'])
    def test_invalid_and_stale_roi_do_not_change_settings(self):
        for region in [[0,0,0,1],[-1,0,1,1],[0,0,float('nan'),1]]:
            with self.assertRaises(ProjectError):save(self.store,self.pid,self.did,{'revision':0,'region':region})
        with self.store.connect() as db:self.assertEqual(setting(db,self.did)['revision'],0)
        save(self.store,self.pid,self.did,{'revision':0,'region':[0,0,.5,1]})
        with self.assertRaises(ProjectError):save(self.store,self.pid,self.did,{'revision':0,'region':None})
    def test_roi_survives_project_backup_restore(self):
        save(self.store,self.pid,self.did,{'revision':0,'region':[0,0,.5,1]});v=self.repo.get(self.pid,self.a,self.ids[1]);self.repo.save(self.pid,self.a,self.ids[1],{'state':'no_target','boxes':[],'revision':v['revision']})
        maintenance=MaintenanceRepository(self.store);backup=maintenance.backup(self.pid)
        path,_,_=maintenance.archive(backup['id'])
        restored=maintenance.restore(path);target=Storage(Path(restored['data_directory'])/'platform.db')
        with target.connect() as db:
            self.assertEqual(setting(db,self.did)['revision'],1)
            self.assertEqual(db.execute('SELECT count(*) FROM roi_annotations').fetchone()[0],1)

class PolygonCropTests(TestCase):
    def test_concave_polygon_splits_into_separate_valid_regions(self):
        shapes=[{'class_id':'c','points':[[0,0],[10,0],[10,10],[8,10],[8,2],[2,2],[2,10],[0,10]]}]
        result=clip(shapes,{'x':0,'y':4,'width':10,'height':6});self.assertEqual(len(result),2)
        self.assertAlmostEqual(sum(s['width']*s['height'] for s in result),24)
