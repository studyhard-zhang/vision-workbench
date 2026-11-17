"""Business tasks and explicit source-image linkage, separate from training jobs."""
from .storage import ProjectError, timestamp, validate_project, validate_revision
import uuid
import sqlite3


TASK_SELECT = """
    SELECT t.*, d.name AS dataset_name,
        (SELECT count(*) FROM task_classes WHERE task_id=t.id) AS class_count,
        (SELECT count(*) FROM task_annotations WHERE task_id=t.id AND state='annotated') AS annotated_count,
        (SELECT count(*) FROM task_annotations WHERE task_id=t.id AND state='no_target') AS no_target_count,
        (SELECT count(*) FROM task_annotations WHERE task_id=t.id AND state='excluded') AS excluded_count,
        (SELECT count(*) FROM task_images WHERE task_id=t.id) AS image_count,
        (SELECT count(*) FROM images WHERE dataset_id=t.dataset_id) AS source_image_count,
        (SELECT count(*) FROM images i WHERE i.dataset_id=t.dataset_id AND NOT EXISTS
            (SELECT 1 FROM task_images ti WHERE ti.task_id=t.id AND ti.image_id=i.id)) AS pending_image_count
    FROM vision_tasks t JOIN datasets d ON d.id=t.dataset_id AND d.project_id=t.project_id
"""


class TaskRepository:
    def __init__(self, storage):
        self.storage = storage

    @staticmethod
    def public(row):
        return {key: row[key] for key in row.keys() if key != 'name_key'}

    def get(self, project_id, task_id, db=None, *, roi_counts=True):
        if db is None:
            with self.storage.connect() as connection:
                return self.get(project_id, task_id, connection,roi_counts=roi_counts)
        self.storage.get(project_id, db)
        row = db.execute(TASK_SELECT + ' WHERE t.project_id=? AND t.id=?', (project_id,task_id)).fetchone()
        if row is None:
            raise ProjectError(404,'TASK_NOT_FOUND','任务不存在、已被删除，或不属于当前项目。')
        value=self.public(row)
        from .roi import setting,effective
        import json
        if roi_counts and json.loads(setting(db,value['dataset_id'])['region_json']):
            states=[effective(db,value,dict(image))['state'] for image in db.execute('SELECT i.* FROM images i JOIN task_images ti ON ti.image_id=i.id WHERE ti.task_id=?',(task_id,))]
            for state in ('annotated','no_target','excluded'):value[state+'_count']=states.count(state)
        return value

    def list(self, project_id):
        with self.storage.connect() as db:
            self.storage.get(project_id,db)
            return [self.public(row) for row in db.execute(TASK_SELECT + ' WHERE t.project_id=? ORDER BY t.updated_at DESC,t.id DESC',(project_id,))]

    def create(self, project_id, values):
        name,description,key=validate_project(values,'任务')
        if values.get('task_type') not in ('detect','segment'):
            raise ProjectError(422,'UNSUPPORTED_TASK_TYPE','请选择目标检测或实例分割任务。',{'task_type':'请选择支持的任务类型。'})
        dataset_id=values.get('dataset_id')
        if not isinstance(dataset_id,str) or len(dataset_id)!=32:
            raise ProjectError(422,'DATASET_REQUIRED','请选择当前项目的数据集。',{'dataset_id':'请选择关联数据集。'})
        task_id,now=uuid.uuid4().hex,timestamp()
        try:
            with self.storage.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                self.storage.get_dataset(project_id,dataset_id,db)
                db.execute('INSERT INTO vision_tasks(id,project_id,dataset_id,name,name_key,description,task_type,created_at,updated_at,last_synced_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (task_id,project_id,dataset_id,name,key,description,values['task_type'],now,now,now))
                db.execute('INSERT INTO task_images(task_id,image_id,dataset_id,added_at) SELECT ?,id,dataset_id,? FROM images WHERE dataset_id=?',
                    (task_id,now,dataset_id))
                return self.get(project_id,task_id,db)
        except sqlite3.IntegrityError as exc:
            if 'vision_tasks.project_id, vision_tasks.name_key' not in str(exc):raise
            raise ProjectError(409,'TASK_NAME_EXISTS','当前项目中已有同名任务，请换一个名称。',{'name':'任务名称在同一项目内不能重复。'}) from exc

    def update(self,project_id,task_id,values):
        name,description,key=validate_project(values,'任务')
        revision=validate_revision(values.get('revision'),'任务')
        try:
            with self.storage.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                current=self.get(project_id,task_id,db)
                if current['revision']!=revision:
                    raise ProjectError(409,'REVISION_CONFLICT','任务已在其他页面更新。当前输入已保留，请刷新任务后重试。')
                db.execute('UPDATE vision_tasks SET name=?,name_key=?,description=?,revision=revision+1,updated_at=? WHERE id=?',
                    (name,key,description,timestamp(),task_id))
                return self.get(project_id,task_id,db)
        except sqlite3.IntegrityError as exc:
            if 'vision_tasks.project_id, vision_tasks.name_key' not in str(exc):raise
            raise ProjectError(409,'TASK_NAME_EXISTS','当前项目中已有同名任务，请换一个名称。',{'name':'任务名称在同一项目内不能重复。'}) from exc

    def sync(self,project_id,task_id,values):
        revision=validate_revision(values.get('revision'),'任务')
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current=self.get(project_id,task_id,db)
            if current['revision']!=revision:
                raise ProjectError(409,'REVISION_CONFLICT','任务已变化，请刷新后重新同步。')
            now=timestamp()
            cursor=db.execute('INSERT INTO task_images(task_id,image_id,dataset_id,added_at) SELECT ?,i.id,i.dataset_id,? FROM images i WHERE i.dataset_id=? AND NOT EXISTS (SELECT 1 FROM task_images ti WHERE ti.task_id=? AND ti.image_id=i.id)',
                (task_id,now,current['dataset_id'],task_id))
            added=cursor.rowcount
            if added:
                db.execute('UPDATE vision_tasks SET revision=revision+1,updated_at=?,last_synced_at=? WHERE id=?',(now,now,task_id))
            return {'task':self.get(project_id,task_id,db),'added_count':added}

    def images(self,project_id,task_id,page=1,page_size=24,query='',state=''):
        if not 1<=page<=1_000_000 or not 1<=page_size<=100:
            raise ProjectError(422,'INVALID_QUERY','任务图片分页参数无效。')
        if not isinstance(query,str) or len(query)>1000 or state not in ('','unannotated','annotated','no_target','excluded'):
            raise ProjectError(422,'INVALID_QUERY','任务图片搜索或状态筛选无效。')
        with self.storage.connect() as db:
            task=self.get(project_id,task_id,db)
            source=" FROM task_images ti JOIN images i ON i.id=ti.image_id AND i.dataset_id=ti.dataset_id LEFT JOIN task_annotations a ON a.task_id=ti.task_id AND a.image_id=ti.image_id WHERE ti.task_id=?"
            params=[task_id]
            if query.strip():
                source+=' AND (instr(lower(i.filename),lower(?))>0 OR instr(lower(i.group_id),lower(?))>0)'
                params.extend([query.strip(),query.strip()])
            from .roi import setting,effective,image_info
            import json
            if json.loads(setting(db,task['dataset_id'])['region_json']):
                all_rows=[dict(row) for row in db.execute('SELECT i.*'+source+' ORDER BY i.created_at DESC,i.id DESC',params)]
                projected=[{**image_info(db,row),'annotation_state':effective(db,task,row)['state']} for row in all_rows]
                if state:projected=[row for row in projected if row['annotation_state']==state]
                total=len(projected)
                return {'images':projected[(page-1)*page_size:page*page_size],'total':total,'page':page,'page_size':page_size,'pages':max(1,(total+page_size-1)//page_size)}
            if state:
                source+=" AND COALESCE(a.state,'unannotated')=?";params.append(state)
            total=db.execute('SELECT count(*)'+source,params).fetchone()[0]
            rows=db.execute("SELECT i.*,COALESCE(a.state,'unannotated') AS annotation_state"+source+' ORDER BY i.created_at DESC,i.id DESC LIMIT ? OFFSET ?',
                [*params,page_size,(page-1)*page_size]).fetchall()
            from .roi import image_info
            return {'images':[{**image_info(db,dict(row)),'annotation_state':__import__('app.roi',fromlist=['effective']).effective(db,task,dict(row))['state']} for row in rows],'total':total,'page':page,'page_size':page_size,
                    'pages':max(1,(total+page_size-1)//page_size)}

    def delete(self,project_id,task_id,values):
        revision=validate_revision(values.get('revision'),'任务')
        try:
            with self.storage.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                task=self.get(project_id,task_id,db)
                if task['revision']!=revision:
                    raise ProjectError(409,'REVISION_CONFLICT','任务已变化，请关闭弹窗并刷新后重试。')
                if values.get('confirmation_name')!=task['name']:
                    raise ProjectError(422,'CONFIRMATION_REQUIRED','请输入完整任务名称以确认删除。')
                if db.execute('SELECT 1 FROM task_data_versions WHERE task_id=? LIMIT 1',(task_id,)).fetchone():
                    raise ProjectError(409,'TASK_HAS_VERSIONS','任务已有已发布数据版本，当前不能删除；请保留任务与原图引用。')
                db.execute('DELETE FROM task_split_images WHERE task_id=?',(task_id,))
                db.execute('DELETE FROM task_splits WHERE task_id=?',(task_id,))
                db.execute('DELETE FROM detection_boxes WHERE task_id=?',(task_id,))
                db.execute('DELETE FROM segmentation_polygons WHERE task_id=?',(task_id,))
                db.execute('DELETE FROM task_annotations WHERE task_id=?',(task_id,))
                db.execute('DELETE FROM task_classes WHERE task_id=?',(task_id,))
                db.execute('DELETE FROM task_images WHERE task_id=?',(task_id,))
                db.execute('DELETE FROM vision_tasks WHERE id=?',(task_id,))
                return task
        except sqlite3.IntegrityError as exc:
            raise ProjectError(409,'TASK_IN_USE','任务已有其他记录引用，请先处理关联内容。') from exc
