import ast
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request,urlopen

from test_versions import VersionFixture
from app.storage import ProjectError
from app.server import create_server
from tools.verify_dataset import audit
from PIL import Image


class BundleAuditTests(VersionFixture):
    def setUp(self):
        super().setUp()
        self.repo.save(self.pid,self.a,self.ids[0],{'state':'unannotated','revision':2,'boxes':[]})
        self.repo.save(self.pid,self.a,self.ids[1],{'state':'no_target','revision':2,'boxes':[]})
        self.published=self.publish();self.vid=self.published['version']['id']
        with self.versions.archive(self.pid,self.a,self.vid) as (stream,_,_):self.raw=stream.read()

    def altered(self, mutate):
        with zipfile.ZipFile(io.BytesIO(self.raw)) as archive:files={n:archive.read(n) for n in archive.namelist()}
        manifest=json.loads(files['manifest.json']);mutate(files,manifest)
        files['manifest.json']=json.dumps(manifest,ensure_ascii=False).encode('utf-8')
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as archive:
            for name,value in files.items():archive.writestr(name,value)
        output.seek(0);return audit(output)

    def test_zip_and_extracted_directory_pass_without_changing_inputs(self):
        zipped=audit(io.BytesIO(self.raw));self.assertTrue(zipped['passed'],zipped)
        target=self.path.parent/'dataset';target.mkdir()
        with zipfile.ZipFile(io.BytesIO(self.raw)) as archive:archive.extractall(target)
        before={str(p.relative_to(target)):p.read_bytes() for p in target.rglob('*') if p.is_file()}
        extracted=audit(target);self.assertTrue(extracted['passed'],extracted)
        self.assertEqual(zipped['stats'],extracted['stats']);self.assertFalse(extracted['training_executed'])
        self.assertEqual(before,{str(p.relative_to(target)):p.read_bytes() for p in target.rglob('*') if p.is_file()})
        self.assertEqual(zipped['stats']['image_count'],len(self.ids)-1)

    def test_changed_label_fails_hash_check(self):
        result=self.altered(lambda files,m:files.update({m['images'][0]['label_path']:b'wrong'}))
        self.assertFalse(result['passed']);self.assertIn('校验失败',result['checks'][-1]['message'])

    def test_rehashed_wrong_coordinates_still_fail_semantic_check(self):
        def mutate(files,m):
            image=next(i for i in m['images'] if i['state']=='annotated');name=image['label_path']
            lines=files[name].decode().splitlines();items=lines[0].split();items[1]='0.99';lines[0]=' '.join(items)
            files[name]=('\n'.join(lines)+'\n').encode();m['files'][name]=hashlib.sha256(files[name]).hexdigest()
        result=self.altered(mutate);self.assertFalse(result['passed']);self.assertIn('坐标',result['checks'][-1]['message'])

    def test_unannotated_or_noncontiguous_classes_are_rejected(self):
        for mutate in [lambda f,m:m['images'][0].update(state='unannotated'),lambda f,m:m['classes'][0].update(index=9)]:
            with self.subTest(mutate=mutate):self.assertFalse(self.altered(mutate)['passed'])

    def test_group_leakage_and_statistics_mismatch_are_rejected(self):
        def group(files,m):
            train=next(i for i in m['images'] if i['subset']=='train');val=next(i for i in m['images'] if i['subset']=='val')
            train['group_id']=val['group_id']='leaked-group'
        def counts(files,m):m['distribution']['train']['image_count']+=1
        self.assertIn('跨集合',self.altered(group)['checks'][-1]['message'])
        self.assertIn('统计',self.altered(counts)['checks'][-1]['message'])

    def test_invalid_dimensions_and_corrupt_pixels_have_structured_failures(self):
        self.assertFalse(self.altered(lambda f,m:m['images'][0].update(width=-1,height=-1))['passed'])
        def corrupt(files,m):
            image=m['images'][0];files[image['image_path']]=b'not an image';image['display_sha256']=hashlib.sha256(b'not an image').hexdigest()
        self.assertFalse(self.altered(corrupt)['passed'])

    def test_missing_extra_and_duplicate_paths_fail_without_extracting(self):
        self.assertFalse(self.altered(lambda files,m:files.pop(m['images'][0]['label_path']))['passed'])
        self.assertFalse(self.altered(lambda files,m:files.update({'../escape.txt':b'bad'}))['passed'])
        output=io.BytesIO(self.raw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(output,'a') as archive:archive.writestr('data.yaml',b'{}')
        output.seek(0);self.assertIn('重复',audit(output)['checks'][-1]['message'])
        self.assertFalse((self.path.parent/'escape.txt').exists())

    def test_readiness_is_task_scoped_read_only_and_releases_temporary_files(self):
        with self.store.connect() as db:
            tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            before={t:[tuple(r) for r in db.execute('SELECT * FROM '+t)] for t in tables}
        report=self.versions.readiness(self.pid,self.a,self.vid)
        self.assertTrue(report['passed']);self.assertTrue(report['source_integrity_checked']);self.assertGreater(report['archive_bytes'],0)
        self.assertEqual(list(self.versions.staging.iterdir()),[])
        with self.assertRaises(ProjectError):self.versions.readiness(self.pid,self.b,self.vid)
        with self.store.connect() as db:
            self.assertEqual(before,{t:[tuple(r) for r in db.execute('SELECT * FROM '+t)] for t in tables})

    def test_damaged_reference_and_concurrent_operation_block_readiness(self):
        self.versions.slot.acquire()
        try:
            with self.assertRaises(ProjectError) as error:self.versions.readiness(self.pid,self.a,self.vid)
            self.assertEqual(error.exception.code,'VERSION_BUSY')
        finally:self.versions.slot.release()
        image=self.detail(self.published)['manifest']['images'][0]
        source=self.versions.source_path(self.detail(self.published)['manifest'],image,'original');source.unlink()
        with self.assertRaises(ProjectError) as error:self.versions.readiness(self.pid,self.a,self.vid)
        self.assertEqual(error.exception.code,'VERSION_DAMAGED')

    def test_cli_writes_external_report_and_rejects_overwriting_input(self):
        script=Path(__file__).resolve().parents[1]/'tools/verify_dataset.py'
        ast.parse(script.read_text(encoding='utf-8'),feature_version=(3,10))
        source=self.path.parent/'portable.zip';source.write_bytes(self.raw);report=self.path.parent/'report.json'
        done=subprocess.run([sys.executable,str(script),str(source),'--report',str(report)],capture_output=True)
        self.assertEqual(done.returncode,0,done.stderr.decode('utf-8'));self.assertTrue(json.loads(report.read_text(encoding='utf-8'))['passed'])
        denied=subprocess.run([sys.executable,str(script),str(source),'--report',str(source)],capture_output=True)
        self.assertEqual(denied.returncode,2);self.assertEqual(source.read_bytes(),self.raw)
        missing=subprocess.run([sys.executable,str(script),str(source.with_name('missing.zip'))],capture_output=True)
        self.assertEqual(missing.returncode,2);self.assertFalse(json.loads(missing.stdout)['passed'])


class DataFlowHttpTests(unittest.TestCase):
    def test_project_multiple_datasets_shared_tasks_partial_annotations_versions_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            server=create_server(0,Path(directory)/'platform.db');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            def call(path,method='GET',body=None,raw=None,headers=None,expected=200):
                data=raw if raw is not None else json.dumps(body).encode() if body is not None else None
                request=Request(base+path,method=method,data=data,headers=headers or {'Content-Type':'application/json'})
                try:response=urlopen(request,timeout=15)
                except HTTPError as error:response=error
                with response:
                    payload=response.read();self.assertEqual(response.status,expected,payload[:600])
                    return json.loads(payload) if 'application/json' in response.headers.get('Content-Type','') else payload
            try:
                pid=call('/api/projects','POST',{'name':'闭环项目'},expected=201)['project']['id'];project=f'/api/projects/{pid}'
                other=call('/api/projects','POST',{'name':'隔离项目'},expected=201)['project']['id']
                datasets=[call(project+'/datasets','POST',{'name':name},expected=201)['dataset']['id'] for name in ['原图一','原图二']]
                empty_task=call(project+'/tasks','POST',{'name':'空源可创建','task_type':'detect','dataset_id':datasets[1]},expected=201)['task']
                self.assertEqual(empty_task['image_count'],0)
                raw_images=[];image_ids=[]
                for index in range(6):
                    stream=io.BytesIO();Image.new('RGB',(96,64),(index*35,40,160)).save(stream,'PNG');raw=stream.getvalue();raw_images.append(raw)
                    result=call(project+'/datasets/'+datasets[0]+'/images','POST',raw=raw,headers={'Content-Type':'image/png','X-File-Name':quote(f'part-{index}.png'),'X-Group-Id':quote('组一' if index in (0,3) else '组二' if index in (1,2) else '')},expected=201)
                    image_ids.append(result['image']['id'])
                duplicate=call(project+'/datasets/'+datasets[0]+'/images','POST',raw=raw_images[0],headers={'Content-Type':'image/png','X-File-Name':'again.png'})
                self.assertEqual(duplicate['status'],'duplicate');self.assertEqual(duplicate['image']['id'],image_ids[0])
                reused=call(project+'/datasets/'+datasets[1]+'/images','POST',raw=raw_images[0],headers={'Content-Type':'image/png','X-File-Name':'same.png'},expected=201)['image']
                self.assertNotEqual(reused['id'],image_ids[0]);self.assertEqual(len(call(project+'/datasets')['datasets']),2)
                synced=call(project+'/tasks/'+empty_task['id']+'/sync-images','POST',{'revision':1})
                self.assertEqual(synced['task']['image_count'],1)
                tasks=[call(project+'/tasks','POST',{'name':name,'task_type':'detect','dataset_id':datasets[0]},expected=201)['task'] for name in ['任务A','任务B']]
                a=project+'/tasks/'+tasks[0]['id'];b=project+'/tasks/'+tasks[1]['id']
                cid=call(a+'/classes','POST',{'name':'目标A'},expected=201)['class']['id']
                bid=call(b+'/classes','POST',{'name':'独立目标B'},expected=201)['class']['id']
                for index,iid in enumerate(image_ids[:4]):
                    body={'state':'annotated' if index<3 else 'no_target','revision':1,'boxes':[{'class_id':cid,'x':12,'y':8,'width':24,'height':16}] if index<3 else []}
                    call(a+'/images/'+iid+'/annotation','PUT',body)
                call(a+'/images/'+image_ids[0]+'/annotation','PUT',body,expected=409)
                self.assertEqual(call(b+'/images/'+image_ids[0]+'/annotation')['annotation']['state'],'unannotated')
                preparation=call(a+'/preparation');self.assertEqual(preparation['report']['eligible_count'],4)
                self.assertEqual(preparation['report']['error_count'],0)
                plan=call(a+'/split','POST',{'fingerprint':preparation['report']['fingerprint'],'revision':0,'train_percent':80,'val_percent':20,'test_percent':0,'seed':42})
                published=call(a+'/versions','POST',{'fingerprint':plan['report']['fingerprint'],'split_revision':plan['split']['revision']},expected=200)['version']
                version_path=a+'/versions/'+published['id'];frozen=call(version_path)['manifest']
                self.assertEqual({i['id'] for i in frozen['images']},set(image_ids[:4]))
                report=call(version_path+'/readiness');self.assertTrue(report['passed']);self.assertEqual(report['stats']['image_count'],4)
                self.assertFalse(report['training_executed'])
                downloaded=call(version_path+'/archive');self.assertTrue(audit(io.BytesIO(downloaded))['passed'])
                tool=call('/tools/verify-dataset.py');ast.parse(tool.decode('utf-8'),feature_version=(3,10))
                call('/api/projects/'+other+'/tasks/'+tasks[0]['id']+'/versions/'+published['id']+'/readiness',expected=404)
                call(b+'/versions/'+published['id']+'/readiness',expected=404)
                call(b+'/images/'+image_ids[0]+'/annotation','PUT',{'state':'annotated','revision':1,'boxes':[{'class_id':bid,'x':1,'y':1,'width':10,'height':10}]})
                new_image=io.BytesIO();Image.new('RGB',(96,64),'gold').save(new_image,'PNG')
                call(project+'/datasets/'+datasets[0]+'/images','POST',raw=new_image.getvalue(),headers={'Content-Type':'image/png','X-File-Name':'new.png'},expected=201)
                btask=call(b)['task'];call(b+'/sync-images','POST',{'revision':btask['revision']})
                self.assertEqual(call(a)['task']['image_count'],6);self.assertEqual(call(b)['task']['image_count'],7)
                atask=call(a)['task'];call(a+'/sync-images','POST',{'revision':atask['revision']})
                call(a+'/classes/'+cid,'PUT',{'name':'目标A新版','revision':1})
                self.assertEqual(call(version_path)['manifest'],frozen);self.assertTrue(call(version_path+'/readiness')['passed'])
                self.assertTrue(call(a+'/preparation')['split']['stale'])
                prep=call(a+'/preparation')
                plan2=call(a+'/split','POST',{'fingerprint':prep['report']['fingerprint'],'revision':prep['split']['revision'],'train_percent':80,'val_percent':20,'test_percent':0,'seed':42})
                v2=call(a+'/versions','POST',{'fingerprint':plan2['report']['fingerprint'],'split_revision':plan2['split']['revision']},expected=200)['version']
                self.assertEqual(v2['number'],2);self.assertEqual(v2['image_count'],4)
                self.assertEqual(call(version_path)['manifest'],frozen);self.assertEqual(call(b+'/versions')['total'],0)
                atask=call(a)['task'];call(a,'DELETE',{'revision':atask['revision'],'confirmation_name':atask['name']},expected=409)
                call(project+'/datasets/'+datasets[0],'DELETE',{'revision':1,'confirmation_name':'原图一'},expected=409)
                self.assertTrue(call(a+'/versions/'+v2['id']+'/readiness')['passed'])
            finally:server.shutdown();server.server_close();thread.join(3)
