"""Dataset ROI and reversible annotation views. Original assets/annotations stay intact."""
import hashlib,io,json,math,os,uuid
from PIL import Image
from .storage import ProjectError,timestamp,validate_revision
from tools.polygon_geometry import polygon

def setting(db,dataset_id):
    row=db.execute('SELECT * FROM dataset_rois WHERE dataset_id=?',(dataset_id,)).fetchone()
    return dict(row) if row else {'dataset_id':dataset_id,'revision':0,'region_json':'null'}

def bounds(db,image):
    if 'dataset_id' not in image:return None  # inference inputs remain independent snapshots
    value=setting(db,image['dataset_id']);region=json.loads(value['region_json'])
    if not region:return None
    w,h=image['width'],image['height'];x=max(0,min(w-1,round(region[0]*w)));y=max(0,min(h-1,round(region[1]*h)))
    right=max(x+1,min(w,round(region[2]*w)));bottom=max(y+1,min(h,round(region[3]*h)))
    return {'x':x,'y':y,'width':right-x,'height':bottom-y,'revision':value['revision'],'key':hashlib.sha256(value['region_json'].encode()).hexdigest()}

def clip(shapes,area):
    from shapely.geometry import Polygon,box
    result=[];x,y,w,h=(area[k] for k in ['x','y','width','height'])
    for shape in shapes:
        if 'points' in shape:
            clipped=Polygon(shape['points']).intersection(box(x,y,x+w,y+h));parts=list(clipped.geoms) if hasattr(clipped,'geoms') else [clipped]
            for part in parts:
                if part.geom_type!='Polygon' or part.area==0:continue
                if part.area<1:raise ProjectError(422,'ROI_TINY_POLYGON','ROI裁剪产生不足1像素的分割目标，请调整区域后保存。原标注保留。')
                if part.interiors:raise ProjectError(422,'ROI_COMPLEX_POLYGON','ROI裁剪产生当前标注格式不支持的孔洞，请调整区域。')
                try:checked=polygon([[max(0,min(w,px-x)),max(0,min(h,py-y))] for px,py in list(part.exterior.coords)[:-1]],w,h)
                except ValueError as exc:raise ProjectError(422,'ROI_COMPLEX_POLYGON',str(exc)) from exc
                result.append({'class_id':shape['class_id'],**checked})
        else:
            l=max(x,shape['x']);t=max(y,shape['y']);r=min(x+w,shape['x']+shape['width']);b=min(y+h,shape['y']+shape['height'])
            if r>l and b>t:result.append({'class_id':shape['class_id'],'x':l-x,'y':t-y,'width':r-l,'height':b-t})
    if len(result)>200:raise ProjectError(422,'ROI_COMPLEX_POLYGON','裁剪后目标超过200个，请调整区域。')
    return result

def present(db,image,annotation):
    area=bounds(db,image)
    if not area:return annotation
    result=dict(annotation);saved=db.execute('SELECT * FROM roi_annotations WHERE task_id=? AND image_id=? AND roi_key=?',(annotation['task_id'],image['id'],area['key'])).fetchone()
    result['boxes']=json.loads(saved['boxes_json']) if saved else clip(annotation['boxes'],area)
    result['state']=saved['state'] if saved else ('unannotated' if annotation['state']=='annotated' and not result['boxes'] else annotation['state'])
    result['revision']=annotation['revision']*1000000+area['revision']*1000+(saved['revision'] if saved else 0);result['roi']=area
    return result

def effective(db,task,image):
    row=db.execute('SELECT * FROM task_annotations WHERE task_id=? AND image_id=?',(task['id'],image['id'])).fetchone()
    value=dict(row) if row else {'task_id':task['id'],'image_id':image['id'],'state':'unannotated','revision':1}
    if task['task_type']=='segment':value['boxes']=[{'class_id':p['class_id'],'points':json.loads(p['points_json'])} for p in db.execute('SELECT class_id,points_json FROM segmentation_polygons WHERE task_id=? AND image_id=? ORDER BY position',(task['id'],image['id']))]
    else:value['boxes']=[dict(p) for p in db.execute('SELECT class_id,x,y,width,height FROM detection_boxes WHERE task_id=? AND image_id=? ORDER BY position',(task['id'],image['id']))]
    return present(db,image,value)

def image_info(db,image):
    area=bounds(db,image)
    return {**image,'source_width':image['width'],'source_height':image['height'],'width':area['width'],'height':area['height'],'roi':area} if area else image

def cropped(path,area):
    with Image.open(path) as source:
        result=source.crop((area['x'],area['y'],area['x']+area['width'],area['y']+area['height'])).convert('RGB');out=io.BytesIO();result.save(out,format='PNG');result.close();return out.getvalue()

def save(storage,project_id,dataset_id,values):
    if set(values)!={'region','revision'}:raise ProjectError(422,'INVALID_ROI','请提交ROI区域及当前修订。')
    region=values.get('region');revision=values.get('revision')
    if type(revision) is not int or revision<0:raise ProjectError(422,'INVALID_REVISION','ROI修订无效，请刷新。')
    if region is not None and (not isinstance(region,list) or len(region)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in region) or not (0<=region[0]<region[2]<=1 and 0<=region[1]<region[3]<=1)):raise ProjectError(422,'INVALID_ROI','请选择图片内的有效矩形ROI。')
    with storage.connect() as db:
        db.execute('BEGIN IMMEDIATE');storage.get_dataset(project_id,dataset_id,db);old=setting(db,dataset_id)
        if old['revision']!=revision:raise ProjectError(409,'ROI_CHANGED','ROI已在其他窗口修改，请刷新后重新设置。')
        if json.loads(old['region_json'])==region:return old
        db.execute('INSERT INTO dataset_rois VALUES(?,?,?) ON CONFLICT(dataset_id) DO UPDATE SET region_json=excluded.region_json,revision=excluded.revision',(dataset_id,json.dumps(region),revision+1))
        # Validate clipping before committing; no source labels are rewritten.
        from .annotations import AnnotationRepository
        annotations=AnnotationRepository(storage)
        for row in db.execute('SELECT ti.task_id,ti.image_id FROM task_images ti WHERE dataset_id=?',(dataset_id,)).fetchall():annotations.get(project_id,row['task_id'],row['image_id'],db)
        db.execute('UPDATE task_annotations SET revision=revision+1 WHERE task_id IN (SELECT id FROM vision_tasks WHERE dataset_id=?)',(dataset_id,))
        db.execute('UPDATE vision_tasks SET revision=revision+1,updated_at=? WHERE dataset_id=?',(timestamp(),dataset_id))
        db.execute('UPDATE datasets SET revision=revision+1,updated_at=? WHERE id=?',(timestamp(),dataset_id))
        return setting(db,dataset_id)

def save_annotation(repository,project_id,task_id,image_id,values):
    with repository.storage.connect() as db:
        db.execute('BEGIN IMMEDIATE');current=repository.get(project_id,task_id,image_id,db)
        if current['revision']!=validate_revision(values.get('revision')):raise ProjectError(409,'REVISION_CONFLICT','ROI或标注已变化，请重新打开图片。')
        area=current['roi'];state=values.get('state');shapes=values.get('boxes');task=repository.tasks.get(project_id,task_id,db)
        if state not in ('annotated','no_target','unannotated','excluded') or not isinstance(shapes,list) or len(shapes)>200 or (state=='annotated' and not shapes) or (state in ('no_target','unannotated') and shapes):raise ProjectError(422,'INVALID_ANNOTATION','标注状态与目标不一致。')
        clean=[]
        for shape in shapes:
            if not isinstance(shape,dict):raise ProjectError(422,'INVALID_ANNOTATION','标注结构无效。')
            if not db.execute('SELECT 1 FROM task_classes WHERE task_id=? AND id=?',(task_id,shape.get('class_id'))).fetchone():raise ProjectError(422,'INVALID_CLASS','标注类别无效。')
            if task['task_type']=='segment':
                if set(shape) not in ({'class_id','points'},{'class_id','points','x','y','width','height'}):raise ProjectError(422,'INVALID_POLYGON','多边形字段无效。')
                try:clean.append({'class_id':shape['class_id'],**polygon(shape['points'],area['width'],area['height'])})
                except (ValueError,KeyError) as exc:raise ProjectError(422,'INVALID_POLYGON',str(exc)) from exc
            else:
                if set(shape)!={'class_id','x','y','width','height'} or any(type(shape[k]) not in (int,float) or not math.isfinite(shape[k]) for k in ['x','y','width','height']) or not(0<=shape['x']<shape['x']+shape['width']<=area['width'] and 0<=shape['y']<shape['y']+shape['height']<=area['height']):raise ProjectError(422,'INVALID_BOX','矩形框须位于ROI内。')
                clean.append(shape)
        previous=db.execute('SELECT revision FROM roi_annotations WHERE task_id=? AND image_id=? AND roi_key=?',(task_id,image_id,area['key'])).fetchone()
        db.execute('INSERT INTO roi_annotations VALUES(?,?,?,?,?,?) ON CONFLICT(task_id,image_id,roi_key) DO UPDATE SET state=excluded.state,revision=excluded.revision,boxes_json=excluded.boxes_json',(task_id,image_id,area['key'],state,(previous['revision'] if previous else 0)+1,json.dumps(clean)))
        repository.touch(db,task_id);return repository.get(project_id,task_id,image_id,db)
