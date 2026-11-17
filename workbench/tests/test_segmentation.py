from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import uuid

from PIL import Image,ImageDraw
from app.storage import Storage,ProjectError
from app.tasks import TaskRepository
from app.annotations import AnnotationRepository
from app.images import ImageRepository
from app.preparation import PreparationRepository
from app.versions import VersionRepository
from app.training import TrainingRepository
from app.maintenance import MaintenanceRepository
from app.training_reports import parse_metrics
from tools.polygon_geometry import polygon


def populate(directory):
    storage=Storage(Path(directory)/'platform.db');pid=storage.create({'name':'分割流程QA'})['id'];did=storage.create_dataset(pid,{'name':'独立合成轮廓'})['id'];images=ImageRepository(storage);ids=[]
    for n in range(4):
        picture=Image.new('RGB',(320,256),(22+n*7,39,50));ImageDraw.Draw(picture).polygon([(60+n*10,45),(235,58+n*8),(220,205),(75,190)],fill=(210,90+n*20,60));stream=io.BytesIO();picture.save(stream,'PNG');ids.append(images.import_image(pid,did,f'shape-{n+1}.png',stream.getvalue())['image']['id'])
    tasks=TaskRepository(storage);tid=tasks.create(pid,{'name':'实例分割QA','task_type':'segment','dataset_id':did})['id'];repo=AnnotationRepository(storage);cid=repo.change_class(pid,tid,{'name':'多边形目标'})['id']
    for n,iid in enumerate(ids):repo.save(pid,tid,iid,{'state':'annotated','revision':1,'boxes':[{'class_id':cid,'points':[[60+n*10,45],[235,58+n*8],[220,205],[75,190]]}]})
    prep=PreparationRepository(storage);data=prep.read(pid,tid);data=prep.generate(pid,tid,{'train_percent':75,'val_percent':25,'test_percent':0,'seed':42,'fingerprint':data['report']['fingerprint'],'revision':0});versions=VersionRepository(storage);vid=versions.publish(pid,tid,{'fingerprint':data['report']['fingerprint'],'split_revision':data['split']['revision']})['version']['id']
    return storage,{'project_id':pid,'dataset_id':did,'task_id':tid,'class_id':cid,'image_ids':ids,'version_id':vid}


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store,self.ids=populate(self.tmp.name);self.pid=self.ids['project_id'];self.tid=self.ids['task_id'];self.repo=AnnotationRepository(self.store)
    def tearDown(self):self.tmp.cleanup()
    def test_polygons_round_trip_isolate_and_protect_class_and_revision(self):
        iid=self.ids['image_ids'][0];original=self.repo.get(self.pid,self.tid,iid);self.assertEqual(len(original['boxes'][0]['points']),4)
        with self.assertRaises(ProjectError):self.repo.save(self.pid,self.tid,iid,{'state':'annotated','revision':1,'boxes':original['boxes']})
        with self.assertRaises(ProjectError):self.repo.change_class(self.pid,self.tid,{'revision':1,'confirmation_name':'多边形目标'},self.ids['class_id'],True)
        self.assertEqual(self.repo.classes(self.pid,self.tid)[0]['box_count'],4)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM detection_boxes').fetchone()[0],0)
    def test_invalid_geometry_cannot_change_saved_annotation(self):
        iid=self.ids['image_ids'][0];before=self.repo.get(self.pid,self.tid,iid)
        for points in [[[0,0],[100,100],[0,100],[100,0]],[[1,1],[1,1],[20,20]],[[0,0],[1,1]],[[0,0],[400,0],[0,5]],[[0,0],[10,0],[20,0]],[[0,0],[float('nan'),1],[10,2]]]:
            with self.assertRaises(ProjectError):self.repo.save(self.pid,self.tid,iid,{'state':'annotated','revision':before['revision'],'boxes':[{'class_id':self.ids['class_id'],'points':points}]})
        self.assertEqual(before,self.repo.get(self.pid,self.tid,iid))
    def test_frozen_segment_labels_audit_and_workspace_independence(self):
        versions=VersionRepository(self.store);vid=self.ids['version_id'];before=versions.read(self.pid,self.tid,vid)
        self.assertEqual(before['manifest']['label_format'],'yolo-segment-polygon-normalized-v1');self.assertTrue(versions.readiness(self.pid,self.tid,vid)['passed'])
        with versions.archive(self.pid,self.tid,vid) as (f,*_):
            import zipfile
            with zipfile.ZipFile(f) as z:
                for im in before['manifest']['images']:self.assertEqual(len(z.read(im['label_path']).decode().split()),9)
        self.repo.save(self.pid,self.tid,self.ids['image_ids'][0],{'state':'no_target','revision':2,'boxes':[]});self.assertEqual(before,versions.read(self.pid,self.tid,vid));self.assertTrue(PreparationRepository(self.store).read(self.pid,self.tid)['split']['stale'])
    def test_segment_does_not_borrow_detection_capabilities(self):
        training=TrainingRepository(self.store);self.assertFalse(training.options(self.pid,self.tid)['segment_verified'])
        with self.assertRaises(ProjectError) as e:training.create(self.pid,self.tid,{'name':'训练','version_id':self.ids['version_id'],'target_id':training.target()['id'],'epochs':1,'idempotency_key':uuid.uuid4().hex})
        self.assertEqual(e.exception.code,'SEGMENT_NOT_VERIFIED')
    def test_backup_restore_preserves_polygons_and_task_type(self):
        m=MaintenanceRepository(self.store);result=m.backup(self.pid)
        # Inspect archive through the public adapter, then restore into a new namespace.
        archive=m.archives/(result['id']+'.zip');restored=m.restore(archive)
        copy=Storage(Path(restored['data_directory'])/'platform.db');repo=AnnotationRepository(copy)
        self.assertEqual(repo.get(self.pid,self.tid,self.ids['image_ids'][0]),self.repo.get(self.pid,self.tid,self.ids['image_ids'][0]))
        self.assertEqual(TaskRepository(copy).get(self.pid,self.tid)['task_type'],'segment')
    def test_migration14_adds_only_polygon_table_preserving_data(self):
        with self.store.connect() as db:
            before=[tuple(r) for r in db.execute('SELECT * FROM vision_tasks')];db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=14')
        upgraded=Storage(self.store.path)
        with upgraded.connect() as db:self.assertEqual([tuple(r) for r in db.execute('SELECT * FROM vision_tasks')],before);self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16);self.assertFalse(db.execute('PRAGMA foreign_key_check').fetchall())
    def test_mask_metrics_keep_zero_and_missing_distinct_from_boxes(self):
        r=parse_metrics(b'epoch,metrics/precision(B),metrics/precision(M),metrics/mAP50-95(M)\n1,0.9,0,\n',1,1)['rows'][0];self.assertEqual(r['metrics/precision(M)'],0);self.assertIsNone(r['metrics/mAP50-95(M)']);self.assertEqual(r['metrics/precision(B)'],.9)

    def test_migration15_failure_rolls_back_table_and_schema_then_can_retry(self):
        with self.store.connect() as db:db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=14')
        class FailingStorage(Storage):
            @staticmethod
            def create_segment_tables(db):Storage.create_segment_tables(db);raise RuntimeError('injected migration failure')
        with self.assertRaises(RuntimeError):FailingStorage(self.store.path)
        with closing(sqlite3.connect(self.store.path)) as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],14);self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='segmentation_polygons'").fetchone())
        Storage(self.store.path)

    def test_legacy_schema14_archive_restores_without_polygon_table(self):
        import zipfile
        pid=self.store.create({'name':'旧版项目'})['id'];m=MaintenanceRepository(self.store);item=m.backup(pid);source=m.archives/(item['id']+'.zip')
        with zipfile.ZipFile(source) as z:files={name:z.read(name) for name in z.namelist()}
        manifest=json.loads(files['manifest.json']);snapshot=json.loads(files['snapshot.json']);manifest['schema']=snapshot['schema']=14;snapshot['tables'].pop('segmentation_polygons');snapshot['tables'].pop('dataset_rois');snapshot['tables'].pop('roi_annotations');snapshot['tables']['schema_version']=[{'version':14}];files['snapshot.json']=json.dumps(snapshot,ensure_ascii=False).encode()
        manifest['files']['snapshot.json']={'bytes':len(files['snapshot.json']),'sha256':hashlib.sha256(files['snapshot.json']).hexdigest()};files['manifest.json']=json.dumps(manifest,ensure_ascii=False).encode()
        archive=Path(self.tmp.name)/'schema14.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name,raw in files.items():z.writestr(name,raw)
        result=m.restore(archive);restored=Storage(Path(result['data_directory'])/'platform.db');self.assertEqual(restored.get(pid)['name'],'旧版项目')

