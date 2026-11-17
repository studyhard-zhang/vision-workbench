"""Read-only geometry illustrations from immutable labels, without a training runtime."""
import base64
import io
import json
import random

from PIL import Image,ImageOps
from .storage import ProjectError
from tools.training_config import configuration,parameters,geometry




class AugmentationRepository:
    def __init__(self,versions):self.versions=versions

    def preview(self,project_id,task_id,values):
        if not isinstance(values,dict) or set(values)-{'version_id','image_id','configuration','sample'}:
            raise ProjectError(422,'INVALID_PREVIEW','增强预览请求含不支持字段。')
        try:cfg=configuration(values.get('configuration'))
        except ValueError as exc:raise ProjectError(422,'INVALID_CONFIGURATION',str(exc)) from exc
        if cfg['profile'] in ('official26','custom26'):raise ProjectError(422,'PREVIEW_UNAVAILABLE','官方默认方案包含随机组合增强，请在训练中查看实际效果，或选择手动几何方案查看示意。')
        sample=values.get('sample',0)
        if type(sample) is not int or not 0<=sample<=9999:raise ProjectError(422,'INVALID_PREVIEW','示例编号需为0至9999。')
        with self.versions.storage.connect() as db:row=self.versions.row(db,project_id,task_id,values.get('version_id'))
        manifest=json.loads(row['manifest_json'])
        self.versions.require_integrity(self.versions.integrity(row,manifest))
        image=next((item for item in manifest['images'] if item['id']==values.get('image_id')),None)
        if image is None:raise ProjectError(404,'VERSION_IMAGE_NOT_FOUND','图片不属于所选冻结版本。')
        raw,_=self.versions.image(project_id,task_id,row['id'],image['id'])
        # Validation/test examples are deterministic preprocessing only: never flip them.
        rng=random.Random(f"{cfg['seed']}:{image['id']}:{sample}")
        vertical=image['subset']=='train' and rng.random()<cfg['flipud']
        horizontal=image['subset']=='train' and rng.random()<cfg['fliplr']
        size=parameters(cfg,1)['imgsz'];layout=geometry(image['width'],image['height'],size,image['boxes'],horizontal,vertical)
        with Image.open(io.BytesIO(raw)) as source:
            resized=source.convert('RGB').resize((layout['resized_width'],layout['resized_height']),Image.Resampling.BILINEAR)
        canvas=Image.new('RGB',(size,size),(114,114,114));canvas.paste(resized,(layout['left'],layout['top']))
        if vertical:canvas=ImageOps.flip(canvas)
        if horizontal:canvas=ImageOps.mirror(canvas)
        output=io.BytesIO();canvas.save(output,format='PNG')
        return {'version_id':row['id'],'manifest_sha256':row['manifest_sha256'],'image_id':image['id'],
            'filename':image['filename'],'subset':image['subset'],'configuration':cfg,'sample':sample,
            'geometry':layout,'classes':manifest['classes'],'parameters':parameters(cfg,1),
            'image_data':'data:image/png;base64,'+base64.b64encode(output.getvalue()).decode(),
            'scope':'几何示意：等比缩放、114留边及翻转，框坐标同步。使用Pillow插值，不声明与训练像素一致。实际训练随机顺序、可选库增强和小框筛选须以实例实测为准；不会修改冻结标签。',
            'validation_notice':'验证/测试集仅显示确定性缩放留边，不应用训练翻转。'}
