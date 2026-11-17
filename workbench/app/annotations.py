"""Task-scoped classes and revisioned annotation workspace. No training dependencies."""
import math
import sqlite3
import unicodedata
import uuid
import json
from tools.polygon_geometry import polygon

from .storage import ProjectError, timestamp, validate_revision
from .tasks import TaskRepository

STATES = ('unannotated', 'annotated', 'no_target', 'excluded')
CLASS_COLORS = ('#e76f51', '#2a9d8f', '#e9c46a', '#5a86c5', '#9b6bcc',
                '#e07aa8', '#8dad45', '#cf8a3f', '#38a6c1', '#7c879c')


class AnnotationRepository:
    def __init__(self, storage):
        self.storage = storage
        self.tasks = TaskRepository(storage)

    @staticmethod
    def touch(db, task_id):
        db.execute('UPDATE vision_tasks SET revision=revision+1,updated_at=? WHERE id=?', (timestamp(), task_id))

    def classes(self, project_id, task_id):
        with self.storage.connect() as db:
            self.tasks.get(project_id, task_id, db)
            return [dict(row) for row in db.execute('''SELECT c.id,c.task_id,c.name,c.color,c.revision,
                ((SELECT count(*) FROM detection_boxes b WHERE b.task_id=c.task_id AND b.class_id=c.id)+(SELECT count(*) FROM segmentation_polygons b WHERE b.task_id=c.task_id AND b.class_id=c.id)) AS box_count
                FROM task_classes c WHERE task_id=? ORDER BY c.created_at,c.id''', (task_id,))]

    @staticmethod
    def next_color(db, task_id):
        used = [row[0] for row in db.execute('SELECT color FROM task_classes WHERE task_id=?', (task_id,))]
        return next((color for color in CLASS_COLORS if color not in used), CLASS_COLORS[len(used) % len(CLASS_COLORS)])

    def adopt_default_colors(self):
        """One-time release maintenance for existing classes; keep IDs and annotation references."""
        changed = 0
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for task in db.execute('SELECT id FROM vision_tasks ORDER BY id').fetchall():
                rows = db.execute('SELECT id,color FROM task_classes WHERE task_id=? ORDER BY created_at,id', (task['id'],)).fetchall()
                touched = False
                for index, row in enumerate(rows):
                    color = CLASS_COLORS[index % len(CLASS_COLORS)]
                    if row['color'] != color:
                        db.execute('UPDATE task_classes SET color=?,revision=revision+1,updated_at=? WHERE id=?', (color,timestamp(),row['id']))
                        changed += 1; touched = True
                if touched: self.touch(db,task['id'])
        return changed

    def change_class(self, project_id, task_id, values, class_id=None, delete=False):
        if 'color' in values:
            raise ProjectError(422, 'UNKNOWN_FIELD', '类别颜色由系统自动分配，请刷新页面后重试。')
        if not delete:
            name = values.get('name')
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60 or any(unicodedata.category(c)=='Cc' for c in name):
                raise ProjectError(422, 'INVALID_CLASS', '类别名称需为 1–60 个字符，不能包含控制字符。', {'name':'请检查类别名称。'})
            name = name.strip()
        try:
            with self.storage.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                self.tasks.get(project_id, task_id, db)
                if class_id:
                    row = db.execute('SELECT * FROM task_classes WHERE task_id=? AND id=?', (task_id,class_id)).fetchone()
                    if row is None:
                        raise ProjectError(404, 'CLASS_NOT_FOUND', '当前任务中不存在该类别。')
                    if row['revision'] != validate_revision(values.get('revision'), '类别'):
                        raise ProjectError(409, 'REVISION_CONFLICT', '类别已变化，当前输入已保留。请关闭后刷新类别重试。')
                now = timestamp()
                if delete:
                    if values.get('confirmation_name') != row['name']:
                        raise ProjectError(422, 'CONFIRMATION_REQUIRED', '请输入完整类别名称以确认删除。')
                    if db.execute('SELECT 1 FROM detection_boxes WHERE task_id=? AND class_id=? UNION ALL SELECT 1 FROM segmentation_polygons WHERE task_id=? AND class_id=?', (task_id,class_id,task_id,class_id)).fetchone():
                        raise ProjectError(409, 'CLASS_IN_USE', '类别已有标注引用，请先处理相关标注。')
                    for overlay in db.execute('SELECT boxes_json FROM roi_annotations WHERE task_id=?',(task_id,)):
                        if any(shape['class_id']==class_id for shape in json.loads(overlay['boxes_json'])):raise ProjectError(409,'CLASS_IN_USE','类别仍被ROI标注引用。')
                    db.execute('DELETE FROM task_classes WHERE task_id=? AND id=?', (task_id,class_id))
                elif class_id:
                    db.execute('UPDATE task_classes SET name=?,name_key=?,revision=revision+1,updated_at=? WHERE task_id=? AND id=?',
                        (name,unicodedata.normalize('NFKC',name).casefold(),now,task_id,class_id))
                else:
                    class_id = uuid.uuid4().hex
                    db.execute('INSERT INTO task_classes(id,task_id,name,name_key,color,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',
                        (class_id,task_id,name,unicodedata.normalize('NFKC',name).casefold(),self.next_color(db,task_id),now,now))
                self.touch(db,task_id)
                return {'id':class_id}
        except sqlite3.IntegrityError as exc:
            if 'task_classes.task_id, task_classes.name_key' not in str(exc):
                raise
            raise ProjectError(409,'CLASS_NAME_EXISTS','当前任务已有同名类别，请换一个名称。',{'name':'类别名称在当前任务内不能重复。'}) from exc

    def get(self, project_id, task_id, image_id, db=None):
        if db is None:
            with self.storage.connect() as connection:
                return self.get(project_id,task_id,image_id,connection)
        task=self.tasks.get(project_id,task_id,db,roi_counts=False)
        if not db.execute('SELECT 1 FROM task_images WHERE task_id=? AND image_id=?',(task_id,image_id)).fetchone():
            raise ProjectError(404,'TASK_IMAGE_NOT_FOUND','图片未关联到当前任务，请先同步图片。')
        row = db.execute('SELECT * FROM task_annotations WHERE task_id=? AND image_id=?',(task_id,image_id)).fetchone()
        result = dict(row) if row else {'task_id':task_id,'image_id':image_id,'state':'unannotated','revision':1,'updated_at':None}
        result['boxes'] = [dict(box) for box in db.execute('SELECT class_id,x,y,width,height FROM detection_boxes WHERE task_id=? AND image_id=? ORDER BY position',(task_id,image_id))]
        if task['task_type']=='segment':
            image=db.execute('SELECT width,height FROM images WHERE id=?',(image_id,)).fetchone()
            result['boxes']=[{'class_id':p['class_id'],**polygon(json.loads(p['points_json']),image['width'],image['height'])} for p in db.execute('SELECT class_id,points_json FROM segmentation_polygons WHERE task_id=? AND image_id=? ORDER BY position',(task_id,image_id))]
        from .roi import present
        image=dict(db.execute('SELECT * FROM images WHERE id=?',(image_id,)).fetchone())
        return present(db,image,result)

    def save(self, project_id, task_id, image_id, values):
        if self.get(project_id,task_id,image_id).get('roi'):
            from .roi import save_annotation
            return save_annotation(self,project_id,task_id,image_id,values)
        if self.tasks.get(project_id,task_id)['task_type']=='segment':return self.save_polygons(project_id,task_id,image_id,values)
        revision = validate_revision(values.get('revision'),'标注')
        state, boxes = values.get('state'), values.get('boxes')
        if state not in STATES or not isinstance(boxes,list) or len(boxes)>200:
            raise ProjectError(422,'INVALID_ANNOTATION','请提交有效状态和最多 200 个矩形框。')
        if (state == 'annotated' and not boxes) or (state in ('unannotated','no_target') and boxes):
            raise ProjectError(422,'INVALID_ANNOTATION','已标注须包含合法框；未标注和确认无目标不能包含框。')
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current = self.get(project_id,task_id,image_id,db)
            if current['revision'] != revision:
                raise ProjectError(409,'REVISION_CONFLICT','图片状态或标注已变化，当前输入已保留。请关闭后重新打开图片再修改。')
            image = db.execute('SELECT width,height FROM images WHERE id=?',(image_id,)).fetchone()
            clean=[]
            for box in boxes:
                if not isinstance(box,dict) or set(box) != {'class_id','x','y','width','height'}:
                    raise ProjectError(422,'INVALID_BOX','矩形框字段无效。')
                coords=[box[k] for k in ('x','y','width','height')]
                if any(type(v) not in (float,int) or (type(v) is float and not math.isfinite(v)) for v in coords):
                    raise ProjectError(422,'INVALID_BOX','矩形框坐标必须为有限数值。')
                x,y,w,h=coords
                if x<0 or y<0 or w<=0 or h<=0 or x+w>image['width'] or y+h>image['height']:
                    raise ProjectError(422,'INVALID_BOX','矩形框必须位于方向统一后的图片范围内，且面积大于零。')
                if not isinstance(box['class_id'],str) or not db.execute('SELECT 1 FROM task_classes WHERE task_id=? AND id=?',(task_id,box['class_id'])).fetchone():
                    raise ProjectError(422,'INVALID_BOX_CLASS','矩形框类别必须属于当前任务。')
                clean.append((box['class_id'],x,y,w,h))
            now=timestamp()
            db.execute('''INSERT INTO task_annotations(task_id,image_id,state,revision,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(task_id,image_id) DO UPDATE SET state=excluded.state,revision=excluded.revision,updated_at=excluded.updated_at''',
                (task_id,image_id,state,revision+1,now))
            db.execute('DELETE FROM detection_boxes WHERE task_id=? AND image_id=?',(task_id,image_id))
            for index,box in enumerate(clean):
                db.execute('INSERT INTO detection_boxes VALUES(?,?,?,?,?,?,?,?)',(task_id,image_id,index,*box))
            self.touch(db,task_id)
            return self.get(project_id,task_id,image_id,db)

    def save_polygons(self, project_id, task_id, image_id, values):
        revision=validate_revision(values.get('revision'),'标注');state=values.get('state');shapes=values.get('boxes')
        if state not in STATES or not isinstance(shapes,list) or len(shapes)>200 or (state=='annotated' and not shapes) or (state in ('unannotated','no_target') and shapes):
            raise ProjectError(422,'INVALID_ANNOTATION','请提交有效状态和最多200个多边形。')
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE');current=self.get(project_id,task_id,image_id,db)
            if current['revision']!=revision:raise ProjectError(409,'REVISION_CONFLICT','标注已变化，当前草稿保留，请重新打开图片。')
            image=db.execute('SELECT width,height FROM images WHERE id=?',(image_id,)).fetchone();clean=[]
            for shape in shapes:
                if not isinstance(shape,dict) or set(shape) not in ({'class_id','points'},{'class_id','points','x','y','width','height'}):raise ProjectError(422,'INVALID_POLYGON','多边形字段无效，不能使用检测框代替。')
                try:checked=polygon(shape['points'],image['width'],image['height'])
                except ValueError as exc:raise ProjectError(422,'INVALID_POLYGON',str(exc)) from exc
                if any(k in shape and (type(shape[k]) not in (float,int) or abs(shape[k]-checked[k])>1e-6) for k in ('x','y','width','height')):raise ProjectError(422,'INVALID_POLYGON','外接框与顶点不一致。')
                if not isinstance(shape['class_id'],str) or not db.execute('SELECT 1 FROM task_classes WHERE task_id=? AND id=?',(task_id,shape['class_id'])).fetchone():raise ProjectError(422,'INVALID_BOX_CLASS','类别必须属于当前任务。')
                clean.append((shape['class_id'],json.dumps(checked['points'],allow_nan=False)))
            now=timestamp()
            db.execute('INSERT INTO task_annotations VALUES(?,?,?,?,?) ON CONFLICT(task_id,image_id) DO UPDATE SET state=excluded.state,revision=excluded.revision,updated_at=excluded.updated_at',(task_id,image_id,state,revision+1,now))
            db.execute('DELETE FROM segmentation_polygons WHERE task_id=? AND image_id=?',(task_id,image_id))
            db.executemany('INSERT INTO segmentation_polygons VALUES(?,?,?,?,?)',[(task_id,image_id,i,*p) for i,p in enumerate(clean)])
            self.touch(db,task_id);return self.get(project_id,task_id,image_id,db)
