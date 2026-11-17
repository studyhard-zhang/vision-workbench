import hashlib
import json
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from test_inferences import InferenceFixture
from app.storage import ProjectError
from app.server import create_server


class DatasetInferenceSourcesTests(InferenceFixture):
    def test_dataset_copy_preserves_original_labels_and_deduplicates(self):
        iid=self.ids[0]
        with self.store.connect() as db:
            before={t:[tuple(r) for r in db.execute('SELECT * FROM '+t)] for t in ('images','task_images','task_annotations','detection_boxes','task_data_versions')}
        path,_,_=self.repo_images.file(self.pid,self.did,iid,'original');raw=path.read_bytes()
        result=self.infer_inputs.from_dataset(self.pid,self.a,self.did,iid)
        self.assertEqual(result['input']['original_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertTrue(result['source']['original_unchanged'])
        again=self.infer_inputs.from_dataset(self.pid,self.a,self.did,iid);self.assertTrue(again['reused']);self.assertEqual(result['input']['id'],again['input']['id'])
        self.assertEqual(path.read_bytes(),raw)
        with self.store.connect() as db:
            for table,rows in before.items():self.assertEqual(rows,[tuple(r) for r in db.execute('SELECT * FROM '+table)])

    def test_cross_project_missing_and_tampered_source_rejected(self):
        other=self.store.create({'name':'Other'});dataset=self.store.create_dataset(other['id'],{'name':'foreign'})
        for did,iid in ((dataset['id'],self.ids[0]),(self.did,'0'*32),('../bad',self.ids[0])):
            with self.subTest(did=did),self.assertRaises(ProjectError):self.infer_inputs.from_dataset(self.pid,self.a,did,iid)
        path,_,_=self.repo_images.file(self.pid,self.did,self.ids[0],'original');path.write_bytes(b'changed')
        with self.assertRaises(ProjectError) as error:self.infer_inputs.from_dataset(self.pid,self.a,self.did,self.ids[0])
        self.assertEqual(error.exception.code,'SOURCE_IMAGE_DAMAGED')

    def test_http_dataset_source_and_unknown_fields(self):
        server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        endpoint=f'http://127.0.0.1:{server.server_port}/api/projects/{self.pid}/tasks/{self.a}/inference-inputs/from-dataset'
        def post(values):return urlopen(Request(endpoint,data=json.dumps(values).encode(),headers={'Content-Type':'application/json'}),timeout=15)
        try:
            with post({'dataset_id':self.did,'image_id':self.ids[0]}) as response:self.assertEqual(json.load(response)['input']['status'],'READY')
            with self.assertRaises(HTTPError) as error:post({'dataset_id':self.did,'image_id':self.ids[0],'path':'arbitrary'})
            self.assertEqual(error.exception.code,422)
        finally:server.shutdown();server.server_close();thread.join()
