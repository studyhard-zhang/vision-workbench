"""Bounded configuration contract shared by controller and remote executor; standard library only."""
import math

BASE={'imgsz':640,'batch':1,'workers':0,'cache':False,'amp':False,'plots':False,'seed':42,
      'save':True,'optimizer':'auto','device':0}
# Ultralytics 8.4.203, ultralytics/cfg/default.yaml. Device 0 selects the authorized GPU.
OFFICIAL26 = {'patience': 100,
 'batch': 16,
 'imgsz': 640,
 'save': True,
 'cache': False,
 'device': 0,
 'workers': 8,
 'optimizer': 'auto',
 'seed': 0,
 'deterministic': True,
 'rect': False,
 'cos_lr': False,
 'close_mosaic': 10,
 'amp': True,
 'multi_scale': 0.0,
 'overlap_mask': True,
 'mask_ratio': 4,
 'plots': True,
 'lr0': 0.01,
 'lrf': 0.01,
 'momentum': 0.937,
 'weight_decay': 0.0005,
 'warmup_epochs': 3.0,
 'warmup_momentum': 0.8,
 'warmup_bias_lr': 0.1,
 'box': 7.5,
 'cls': 0.5,
 'dfl': 1.5,
 'hsv_h': 0.015,
 'hsv_s': 0.7,
 'hsv_v': 0.4,
 'degrees': 0.0,
 'translate': 0.1,
 'scale': 0.5,
 'shear': 0.0,
 'perspective': 0.0,
 'flipud': 0.0,
 'fliplr': 0.5,
 'bgr': 0.0,
 'mosaic': 1.0,
 'mixup': 0.0,
 'cutmix': 0.0,
 'copy_paste': 0.0,
 'copy_paste_mode': 'flip'}
OFFICIAL_CONFIG = {'profile':'official26'}
PROFILES={'custom26':{'imgsz':640,'batch':16},'official26':{'imgsz':640,'batch':16},'geometry640':{'imgsz':640,'batch':1},'geometry512':{'imgsz':512,'batch':2},'precision':{'imgsz':1824,'batch':-1}}
DISABLED={key:0.0 for key in ('mosaic','mixup','cutmix','copy_paste','degrees','translate','scale','shear','perspective','hsv_h','hsv_s','hsv_v')}
DISABLED.update(close_mosaic=0,multi_scale=False,rect=False)
SEGMENT_CONFIG={'profile':'geometry640','fliplr':0.,'flipud':0.,'seed':42,'patience':100,'optimizer':'auto'}
PRECISION_CONFIG={'profile':'precision','imgsz':1824,'batch':-1,'fliplr':0.,'flipud':0.,'seed':42,'patience':100,'optimizer':'auto'}


def number(value, low, high):
    return type(value) in (int,float) and math.isfinite(value) and low<=value<=high


def configuration(value):
    if not isinstance(value,dict) or set(value)-{'profile','fliplr','flipud','seed','patience','optimizer','lr0','momentum','imgsz','batch'}:
        raise ValueError('训练方案含不支持的字段。')
    if not isinstance(value.get('profile'),str) or value['profile'] not in PROFILES:raise ValueError('请选择受控几何增强方案。')
    if value.get('profile')=='custom26':
        result=dict(value)
        if 'imgsz' in result and (type(result['imgsz']) is not int or not 320<=result['imgsz']<=2048 or result['imgsz']%32):raise ValueError('输入尺寸须为320至2048的32倍数。')
        if 'batch' in result and (type(result['batch']) is not int or result['batch'] not in (-1,1,2,4,8,16,32)):raise ValueError('批大小请选择自动-1，或1/2/4/8/16/32。')
        for key in ('fliplr','flipud'):
            if key in result and not number(result[key],0,1):raise ValueError('翻转概率须为0至1。')
        for key,low,high in [('seed',0,2147483647),('patience',1,300)]:
            if key in result and (type(result[key]) is not int or not low<=result[key]<=high):raise ValueError(key+'超出允许范围。')
        optimizer=result.get('optimizer','auto')
        if optimizer not in ('auto','AdamW'):raise ValueError('仅支持auto或AdamW优化器。')
        for key,low,high in [('lr0',.000001,.1),('momentum',.01,.999)]:
            if key in result and (optimizer=='auto' or not number(result[key],low,high)):raise ValueError('请在AdamW下设置有效的学习率和动量。')
        return {key:val for key,val in result.items() if key=='profile' or val!=OFFICIAL26[key]}
    if value.get('profile')=='official26':
        if set(value)!={'profile'}:raise ValueError('官方默认方案不接受参数覆盖，请切换手动方案。')
        return dict(OFFICIAL_CONFIG)
    result={'fliplr':.5,'flipud':0.,'seed':42,'patience':100,'optimizer':'auto',**value}
    if result['profile']=='precision':
        result.setdefault('imgsz',1824);result.setdefault('batch',-1)
        if type(result['imgsz']) is not int or not 320<=result['imgsz']<=2048 or result['imgsz']%32:raise ValueError('输入尺寸须为320至2048的32倍数。')
        if type(result['batch']) is not int or result['batch'] not in (-1,1,2,4,8,16,32):raise ValueError('批大小请选择自动-1，或1/2/4/8/16/32。')
    elif 'imgsz' in result or 'batch' in result:raise ValueError('请使用精度优先方案调整输入尺寸和批大小。')
    for key in ('fliplr','flipud'):
        if not number(result[key],0,1):raise ValueError('翻转概率需为0至1的有限数。')
    if type(result['seed']) is not int or not 0<=result['seed']<=2147483647:raise ValueError('随机种子需为0至2147483647的整数。')
    if type(result['patience']) is not int or not 1<=result['patience']<=300:raise ValueError('早停耐心值需为1至300的整数。')
    if result['optimizer'] not in ('auto','AdamW'):raise ValueError('仅支持auto或AdamW优化器。')
    if result['optimizer']=='auto':
        if 'lr0' in value or 'momentum' in value:raise ValueError('auto自动决定学习率和动量，请勿提交无效覆盖值。')
    else:
        result.setdefault('lr0',.001);result.setdefault('momentum',.9)
        if not number(result['lr0'],.000001,.1) or not number(result['momentum'],.01,.999):
            raise ValueError('AdamW初始学习率需为0.000001至0.1，动量需为0.01至0.999。')
    return result


def parameters(config, epochs):
    if type(epochs) is not int or not 1<=epochs<=300:raise ValueError('训练轮数需为1至300的整数。')
    if config is None:return {**BASE,'epochs':epochs}
    cfg=configuration(config)
    if cfg['profile']=='custom26':return {**OFFICIAL26,**{k:v for k,v in cfg.items() if k!='profile'},'epochs':epochs}
    if cfg['profile']=='official26':return {**OFFICIAL26,'epochs':epochs}
    return {**BASE,**PROFILES[cfg['profile']],**DISABLED,**{k:v for k,v in cfg.items() if k!='profile'},'epochs':epochs}


def validate_snapshot(snapshot):
    params=snapshot['requested_parameters']
    expected=parameters(snapshot.get('configuration'),params.get('epochs'))
    if snapshot.get('task_type')=='segment':
        if snapshot.get('configuration')!=SEGMENT_CONFIG and snapshot.get('configuration',{}).get('profile') not in ('precision','official26','custom26'):raise ValueError('分割训练方案无效。')
    if params!=expected or any(type(params[k]) is not type(expected[k]) for k in ('imgsz','batch','workers','amp','seed','device','epochs')):
        raise ValueError('训练参数超出受控方案范围或与方案不一致。')
    return params


def geometry(width,height,size,boxes,horizontal=False,vertical=False):
    ratio=size/max(width,height);rw=min(math.ceil(width*ratio),size);rh=min(math.ceil(height*ratio),size)
    left=round((size-rw)/2-.1);top=round((size-rh)/2-.1)
    transformed=[]
    for box in boxes:
        x=box['x']*rw/width+left;y=box['y']*rh/height+top
        w=box['width']*rw/width;h=box['height']*rh/height
        if horizontal:x=size-x-w
        if vertical:y=size-y-h
        item={**box,'x':x,'y':y,'width':w,'height':h}
        if 'points' in box:
            item['points']=[[(size-(px*rw/width+left)) if horizontal else px*rw/width+left,(size-(py*rh/height+top)) if vertical else py*rh/height+top] for px,py in box['points']]
        transformed.append(item)
    return {'width':size,'height':size,'resized_width':rw,'resized_height':rh,'left':left,'top':top,
            'scale_x':rw/width,'scale_y':rh/height,'horizontal':horizontal,'vertical':vertical,'boxes':transformed}

