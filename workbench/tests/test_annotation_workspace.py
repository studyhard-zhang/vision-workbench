import io
import json
import threading
import unittest
from urllib.request import Request, urlopen
from test_annotations import AnnotationFixture
from app.images import ImageRepository
from app.storage import ProjectError
from app.server import create_server
from PIL import Image


class WorkspaceTests(AnnotationFixture):
    def test_http_accepts_200_fractional_boxes_larger_than_old_body_limit(self):
        cid=self.add()
        box={'class_id':cid,'x':0.1234567890123456,'y':0.2345678901234567,'width':10.123456789012345,'height':11.234567890123456}
        payload=json.dumps({'revision':1,'state':'annotated','boxes':[box]*200},indent=2).encode()
        self.assertGreater(len(payload),32768)
        server=create_server(0,self.path);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            req=Request(f'http://127.0.0.1:{server.server_port}/api/projects/{self.pid}/tasks/{self.a}/images/{self.image}/annotation',method='PUT',data=payload,headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=5) as response:
                self.assertEqual(response.status,200);self.assertEqual(len(json.loads(response.read())['annotation']['boxes']),200)
        finally:server.shutdown();server.server_close();thread.join(3)

    def test_filtered_counts_pagination_and_scope(self):
        cid=self.add();self.save('annotated',boxes=[self.box(cid)])
        repo=ImageRepository(self.store)
        for i in range(3):
            stream=io.BytesIO();Image.new('RGB',(30,20),(i,90,0)).save(stream,'PNG')
            repo.import_image(self.pid,self.did,f'camera-{i}.png',stream.getvalue(),'工位 B')
        task=self.tasks.get(self.pid,self.a);self.tasks.sync(self.pid,self.a,{'revision':task['revision']})
        self.assertEqual(self.tasks.images(self.pid,self.a,state='annotated')['total'],1)
        result=self.tasks.images(self.pid,self.a,2,2,query='工位 B',state='unannotated')
        self.assertEqual((result['total'],result['pages'],len(result['images'])),(3,2,1))
        self.assertEqual(self.tasks.images(self.pid,self.b,state='unannotated')['total'],1)
        self.assertEqual(self.tasks.images(self.pid,self.a,query='CAMERA-1')['total'],1)
        self.assertEqual(self.tasks.images(self.pid,self.a,query='不存在')['total'],0)
        with self.assertRaises(ProjectError):self.tasks.images(self.pid,self.a,state='done')

    def test_exif_canonical_pixels_round_trip_without_changing_original_or_other_task(self):
        stream=io.BytesIO();original=Image.new('RGB',(80,40),'blue');exif=Image.Exif();exif[274]=6;original.save(stream,'JPEG',exif=exif)
        repo=ImageRepository(self.store);image=repo.import_image(self.pid,self.did,'rotated.jpg',stream.getvalue())['image']
        self.assertEqual((image['width'],image['height']),(40,80))
        for tid in [self.a,self.b]:
            task=self.tasks.get(self.pid,tid);self.tasks.sync(self.pid,tid,{'revision':task['revision']})
        cid=self.add();box={'class_id':cid,'x':10.125,'y':45.375,'width':29.875,'height':34.625}
        saved=self.repo.save(self.pid,self.a,image['id'],{'revision':1,'state':'annotated','boxes':[box]})
        self.assertEqual(saved['boxes'],[box]);self.assertEqual(self.repo.get(self.pid,self.b,image['id'])['state'],'unannotated')
        file,_,_=repo.file(self.pid,self.did,image['id'],'original');self.assertEqual(file.read_bytes(),stream.getvalue())
        with self.assertRaises(ProjectError):self.repo.save(self.pid,self.a,image['id'],{'revision':2,'state':'annotated','boxes':[{**box,'width':70}]})
