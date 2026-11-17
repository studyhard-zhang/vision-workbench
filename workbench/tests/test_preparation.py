import io
import json
import sqlite3
import threading
from contextlib import closing
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from unittest import TestCase

from test_annotations import AnnotationFixture
from app.storage import Storage,ProjectError
from app.images import ImageRepository
from app.preparation import PreparationRepository,make_assignments,grouped_images,validate_config
from app.server import create_server
from PIL import Image


class PreparationFixture(AnnotationFixture):
    def setUp(self):
        super().setUp();self.prep=PreparationRepository(self.store);self.repo_images=ImageRepository(self.store)
        self.cids=[self.add(name='划痕'),self.add(name='凹陷')]
        for i in range(8):
            stream=io.BytesIO();Image.new('RGB',(40,20),(i*23,70,90)).save(stream,'PNG')
            self.repo_images.import_image(self.pid,self.did,f'part-{i}.png',stream.getvalue())
        for tid in [self.a,self.b]:
            task=self.tasks.get(self.pid,tid);self.tasks.sync(self.pid,tid,{'revision':task['revision']})
        self.ids=[row['id'] for row in self.tasks.images(self.pid,self.a)['images']]
        for iid in self.ids:
            self.repo.save(self.pid,self.a,iid,{'state':'annotated','revision':1,'boxes':[{'class_id':cid,'x':n*10,'y':1,'width':5,'height':5} for n,cid in enumerate(self.cids)]})

    def generate(self, **changes):
        data=self.prep.read(self.pid,self.a)
        values={'train_percent':70,'val_percent':20,'test_percent':10,'seed':42,'fingerprint':data['report']['fingerprint'],'revision':data['split']['revision'] if data['split'] else 0}
        values.update(changes);return self.prep.generate(self.pid,self.a,values)


class PreparationTests(PreparationFixture):
    def test_seeded_split_is_reproducible_persistent_and_has_nonempty_covered_sets(self):
        first=self.generate();second=self.generate()
        self.assertEqual(first['split']['images'],second['split']['images'])
        self.assertEqual(second['split']['revision'],1)
        self.assertEqual(first['report']['error_count'],0)
        for subset in ['train','val','test']:
            self.assertGreater(first['split']['distribution'][subset]['image_count'],0)
            self.assertGreater(first['split']['distribution'][subset]['positive_count'],0)
        self.assertTrue(all(first['split']['distribution']['train']['classes'].values()))
        self.assertFalse(PreparationRepository(Storage(self.path)).read(self.pid,self.a)['split']['stale'])

    def test_group_members_never_cross_subsets_and_zero_test_is_unset(self):
        for iid in self.ids[:3]:self.repo_images.update_group(self.pid,self.did,iid,{'revision':1,'group_id':'同一次采集'})
        data=self.generate(train_percent=80,val_percent=20,test_percent=0)
        subsets={row['subset'] for row in data['split']['images'] if row['image_id'] in self.ids[:3]}
        self.assertTrue(subsets.issubset({'train','val'}));self.assertEqual(data['split']['distribution']['test']['image_count'],0)
        self.assertEqual(data['split']['test_percent'],0)

    def test_unannotated_is_omitted_and_unused_class_only_warns(self):
        self.repo.save(self.pid,self.a,self.ids[0],{'state':'unannotated','revision':2,'boxes':[]})
        cid=self.add(name='没有样本的类别')
        data=self.prep.read(self.pid,self.a)
        self.assertEqual(data['report']['error_count'],0)
        self.assertNotIn('UNANNOTATED',{i['code'] for i in data['report']['issues']})
        self.assertTrue(any(i['code']=='UNUSED_CLASS' and i['severity']=='warning' for i in data['report']['issues']))
        result=self.generate()
        self.assertEqual(result['split']['distribution']['train']['classes'][cid],0)
        self.assertNotIn(self.ids[0],{r['image_id'] for r in result['split']['images']})

    def test_partial_labels_include_confirmed_negatives_and_ignore_unannotated_assets(self):
        # Two positives, one explicit negative, the rest unannotated or excluded.
        for iid in self.ids[2:]:
            state='no_target' if iid==self.ids[2] else 'excluded' if iid==self.ids[3] else 'unannotated'
            self.repo.save(self.pid,self.a,iid,{'state':state,'revision':2,'boxes':[]})
        ignored=self.ids[4]
        path,_,_=self.repo_images.file(self.pid,self.did,ignored,'original');path.unlink()
        path,_,_=self.repo_images.file(self.pid,self.did,ignored,'view');path.unlink()
        before=[self.repo.get(self.pid,self.a,iid) for iid in self.ids]
        result=self.generate(train_percent=60,val_percent=40,test_percent=0)
        self.assertEqual(result['report']['error_count'],0)
        self.assertEqual(result['report']['eligible_count'],3)
        self.assertEqual({r['image_id'] for r in result['split']['images']},set(self.ids[:3]))
        self.assertTrue(all(i['image_id'] not in self.ids[3:] for i in result['report']['issues']))
        self.assertEqual(sum(s['no_target_count'] for s in result['split']['distribution'].values()),1)
        self.assertEqual(sum(s['image_count'] for s in result['split']['distribution'].values()),3)
        self.assertEqual(before,[self.repo.get(self.pid,self.a,iid) for iid in self.ids])
        self.assertEqual(PreparationRepository(Storage(self.path)).read(self.pid,self.a)['split'],result['split'])

    def test_no_labels_has_one_summary_and_never_turns_unannotated_into_negatives(self):
        self.add(self.b,name='类别')
        data=self.prep.read(self.pid,self.b)
        self.assertEqual(data['report']['eligible_count'],0)
        self.assertEqual([i['code'] for i in data['report']['issues']],['NO_ELIGIBLE_IMAGES'])
        with self.assertRaises(ProjectError) as error:
            self.prep.generate(self.pid,self.b,{'train_percent':80,'val_percent':20,'test_percent':0,'seed':42,'revision':0,'fingerprint':data['report']['fingerprint']})
        self.assertEqual(error.exception.code,'DATA_NOT_READY')
        self.assertTrue(all(self.repo.get(self.pid,self.b,iid)['state']=='unannotated' for iid in self.ids))

    def test_newly_labeled_image_invalidates_and_joins_only_after_regeneration(self):
        iid=self.ids[0];original=self.repo.get(self.pid,self.a,iid)
        self.repo.save(self.pid,self.a,iid,{'state':'unannotated','revision':2,'boxes':[]})
        saved=self.generate()
        self.assertEqual(len(saved['split']['images']),len(self.ids)-1)
        self.repo.save(self.pid,self.a,iid,{'state':'annotated','revision':3,'boxes':original['boxes']})
        current=self.prep.read(self.pid,self.a)
        self.assertTrue(current['split']['stale'])
        self.assertNotIn(iid,{r['image_id'] for r in current['split']['images']})
        result=self.generate()
        self.assertFalse(result['split']['stale'])
        self.assertEqual(len(result['split']['images']),len(self.ids))

    def test_missing_assets_block_but_explicitly_excluded_image_is_not_used(self):
        path,_,_=self.repo_images.file(self.pid,self.did,self.ids[0],'view');path.unlink()
        self.assertIn('ASSET_UNAVAILABLE',{item['code'] for item in self.prep.read(self.pid,self.a)['report']['issues']})
        value=self.repo.get(self.pid,self.a,self.ids[0]);self.repo.save(self.pid,self.a,self.ids[0],{'state':'excluded','revision':value['revision'],'boxes':value['boxes']})
        result=self.generate();self.assertEqual(result['report']['error_count'],0)
        self.assertNotIn(self.ids[0],{row['image_id'] for row in result['split']['images']})

    def test_hash_changes_and_malformed_box_are_reported(self):
        self.generate()
        path,_,_=self.repo_images.file(self.pid,self.did,self.ids[0],'original');path.write_bytes(path.read_bytes()+b'changed')
        with self.store.connect() as db:db.execute('UPDATE detection_boxes SET width=1000 WHERE task_id=? AND image_id=?',(self.a,self.ids[1]))
        codes={item['code'] for item in self.prep.read(self.pid,self.a)['report']['issues']}
        self.assertTrue({'ASSET_CHANGED','INVALID_BOX'}.issubset(codes))
        self.assertTrue(self.prep.read(self.pid,self.a)['split']['stale'])

    def test_validation_class_gaps_are_warnings_while_training_still_covers_all_classes(self):
        for iid in self.ids[1:]:
            value=self.repo.get(self.pid,self.a,iid)
            self.repo.save(self.pid,self.a,iid,{'state':'annotated','revision':value['revision'],'boxes':[value['boxes'][0]]})
        result=self.generate(train_percent=80,val_percent=20,test_percent=0)
        self.assertEqual(result['report']['error_count'],0)
        self.assertTrue(any('凹陷' in warning for warning in result['split']['warnings']))
        self.assertTrue(all(result['split']['distribution']['train']['classes'].values()))

    def test_old_fingerprint_cannot_overwrite_saved_plan_after_annotation_change(self):
        saved=self.generate();before=saved['split']['images']
        self.repo.save(self.pid,self.a,self.ids[0],{'state':'no_target','revision':2,'boxes':[]})
        current=self.prep.read(self.pid,self.a);self.assertTrue(current['split']['stale'])
        values={key:saved['split'][key] for key in ['seed','train_percent','val_percent','test_percent','revision','fingerprint']}
        with self.assertRaises(ProjectError) as error:self.prep.generate(self.pid,self.a,values)
        self.assertEqual(error.exception.code,'PREPARATION_CHANGED')
        self.assertEqual(current['split']['images'],before)
        self.assertFalse(self.generate()['split']['stale'])

    def test_group_and_class_changes_invalidate_saved_plan_and_stale_plan_revision_is_rejected(self):
        saved=self.generate();self.repo_images.update_group(self.pid,self.did,self.ids[0],{'revision':1,'group_id':'新组'})
        self.assertFalse(self.prep.read(self.pid,self.a)['split']['stale'])
        self.generate();self.repo.change_class(self.pid,self.a,{'name':'划伤','revision':1},self.cids[0])
        self.assertTrue(self.prep.read(self.pid,self.a)['split']['stale'])
        self.assertEqual(self.generate(revision=saved['split']['revision'])['split']['revision'],saved['split']['revision']+1)

    def test_isolation_and_cross_project_access(self):
        self.generate();self.assertIsNone(self.prep.read(self.pid,self.b)['split'])
        other=self.store.create({'name':'其他项目'})['id']
        with self.assertRaises(ProjectError):self.prep.read(other,self.a)
        with self.assertRaises(ProjectError):self.prep.generate(other,self.a,{'train_percent':80,'val_percent':20,'test_percent':0,'seed':0,'revision':0})
        self.assertEqual(self.repo.get(self.pid,self.b,self.ids[0])['state'],'unannotated')

    def test_single_group_and_single_positive_group_cannot_be_split_as_valid_data(self):
        for iid in self.ids:self.repo_images.update_group(self.pid,self.did,iid,{'revision':1,'group_id':'相邻帧'})
        self.assertFalse(self.generate()['split']['stale'])

    def test_failed_split_transaction_preserves_previous_plan_and_task_revision(self):
        saved=self.generate();task=self.tasks.get(self.pid,self.a)
        with self.store.connect() as db:db.execute("CREATE TRIGGER fail_split BEFORE INSERT ON task_split_images BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.generate(seed=9)
        self.assertEqual(self.prep.read(self.pid,self.a)['split'],saved['split'])
        self.assertEqual(self.tasks.get(self.pid,self.a),task)

    def test_delete_task_removes_working_split_and_preserves_source_and_other_task(self):
        self.generate();task=self.tasks.get(self.pid,self.a);self.tasks.delete(self.pid,self.a,{'revision':task['revision'],'confirmation_name':task['name']})
        self.assertEqual(self.tasks.get(self.pid,self.b)['image_count'],9)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM task_split_images').fetchone()[0],0)
        self.assertEqual(self.store.get_dataset(self.pid,self.did)['image_count'],9)

    def test_http_read_generate_and_unknown_fields(self):
        self.repo.save(self.pid,self.a,self.ids[0],{'state':'unannotated','revision':2,'boxes':[]})
        server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}/api/projects/{self.pid}/tasks/{self.a}'
        def call(path,body=None):
            req=Request(base+path,method='POST' if body is not None else 'GET',data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
            try:response=urlopen(req,timeout=5)
            except HTTPError as error:response=error
            with response:return response.status,json.load(response)
        try:
            status,data=call('/preparation');self.assertEqual(status,200)
            body={'train_percent':80,'val_percent':20,'test_percent':0,'seed':42,'revision':0,'fingerprint':data['report']['fingerprint']}
            status,saved=call('/split',body);self.assertEqual(status,200);self.assertFalse(saved['split']['stale'])
            self.assertNotIn(self.ids[0],{r['image_id'] for r in saved['split']['images']})
            self.assertEqual(len(saved['split']['images']),len(self.ids)-1)
            self.assertEqual(call('/split',{**body,'execute':'bad'})[0],422)
            self.assertEqual(call('/split',body)[0],200)
        finally:server.shutdown();server.server_close();thread.join(3)


class SplitAlgorithmTests(TestCase):
    def test_transitive_duplicates_and_groups_stay_together_even_through_excluded_bridge(self):
        def row(i,group,original,state='annotated'):
            return {'id':str(i),'group_id':group,'original_sha256':original,'display_sha256':'d'+str(i),'state':state,'boxes':[{'class_id':'c'}] if state=='annotated' else []}
        rows=[row(1,'camera','a'),row(2,'camera','b','excluded'),row(3,'other','b'),row(4,'','c'),row(5,'','d')];rows[1]['display_sha256']=rows[0]['display_sha256']
        result=make_assignments(rows,[{'id':'c'}],{'train':60,'val':20,'test':20},42)
        self.assertEqual(result['1'],result['3']);self.assertNotIn('2',result)
        self.assertEqual(len(set(result.values())),3)

    def test_bad_configs_and_insufficient_positive_validation_are_rejected(self):
        valid={'train_percent':80,'val_percent':20,'test_percent':0,'seed':42}
        for change in [{'train_percent':True},{'seed':-1},{'seed':1.2},{'train_percent':90},{'val_percent':0},{'test_percent':'0'}]:
            with self.assertRaises(ProjectError):validate_config({**valid,**change})
        rows=[{'id':str(i),'group_id':'','original_sha256':str(i//2),'display_sha256':str(i//2),'state':'annotated' if i==0 else 'no_target','boxes':[{'class_id':'c'}] if i==0 else []} for i in range(3)]
        with self.assertRaises(ProjectError):make_assignments(rows,[{'id':'c'}],{'train':80,'val':20,'test':0},42)

    def test_same_data_seed_is_independent_of_input_order_across_many_seeds(self):
        rows=[{'id':str(i),'group_id':'g'+str(i//2),'original_sha256':str(i//2),'display_sha256':str(i//2),'state':'annotated','boxes':[{'class_id':'c'}]} for i in range(20)]
        for seed in range(30):
            a=make_assignments(rows,[{'id':'c'}],{'train':70,'val':20,'test':10},seed)
            b=make_assignments(list(reversed(rows)),[{'id':'c'}],{'train':70,'val':20,'test':10},seed)
            self.assertEqual(a,b)
            for i in range(0,20,2):self.assertEqual(a[str(i)],a[str(i+1)])


class PreparationMigrationTests(PreparationFixture):
    def test_v6_migration_preserves_all_rows_and_rolls_back_injected_failure(self):
        tables=['projects','datasets','images','vision_tasks','task_images','task_classes','task_annotations','detection_boxes']
        with self.store.connect() as db:
            before={t:[tuple(row) for row in db.execute('SELECT * FROM '+t)] for t in tables}
            __import__('schema_utils').drop_evaluations(db);db.execute('DROP TABLE training_record_visibility');db.execute('DROP TABLE model_events');db.execute('DROP TABLE model_versions');db.execute('DROP TABLE training_remote_sessions');db.execute('DROP TABLE training_dispatches');db.execute('DROP TABLE training_jobs');db.execute('DROP INDEX version_task_identity');db.execute('DROP TABLE task_data_version_images');db.execute('DROP TABLE task_data_versions');db.execute('DROP TABLE task_split_images');db.execute('DROP TABLE task_splits');db.execute('DROP TABLE segmentation_polygons');db.execute('UPDATE schema_version SET version=6')
        class FailingStorage(Storage):
            @staticmethod
            def create_preparation_tables(db):Storage.create_preparation_tables(db);raise RuntimeError('injected')
        with self.assertRaises(RuntimeError):FailingStorage(self.path)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],6)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='task_splits'").fetchone())
        restored=Storage(self.path)
        with restored.connect() as db:
            self.assertEqual(db.execute('SELECT version FROM schema_version').fetchone()[0],16)
            for table,records in before.items():self.assertEqual([tuple(row) for row in db.execute('SELECT * FROM '+table)],records)
        self.assertTrue(list((self.path.parent/'backups').glob('schema-v6-*.db')))
