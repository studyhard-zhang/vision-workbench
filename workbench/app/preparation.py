"""Task data validation and deterministic, leakage-aware working splits. No GPU imports."""
import hashlib
import json
import math
from collections import Counter, defaultdict
import time

from PIL import Image

from .storage import ProjectError, timestamp
from .tasks import TaskRepository
from .images import ImageRepository
from tools.polygon_geometry import polygon

SUBSETS = ('train', 'val', 'test')
ALGORITHM = 'content-cover-v3'
CLASS_ALGORITHM = 'class-stratified-v1'


def digest(value):
    def clean(item):
        if isinstance(item,dict):return {k:clean(v) for k,v in item.items()}
        if isinstance(item,list):return [clean(v) for v in item]
        if isinstance(item,float) and not math.isfinite(item):return str(item)
        return item
    return hashlib.sha256(json.dumps(clean(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_config(values):
    keys=('train_percent','val_percent','test_percent')
    ratios=[values.get(key) for key in keys]
    if any(type(n) is not int or n<0 or n>100 for n in ratios) or sum(ratios)!=100 or ratios[0]==0 or ratios[1]==0:
        raise ProjectError(422,'INVALID_SPLIT','训练/验证比例须大于 0，测试可为 0；整数比例合计必须为 100%。')
    seed=values.get('seed')
    if type(seed) is not int or not 0<=seed<=4294967295:
        raise ProjectError(422,'INVALID_SPLIT','随机种子须为 0～4294967295 的整数。')
    return dict(zip(SUBSETS,ratios)),seed


def grouped_images(images):
    parent={row['id']:row['id'] for row in images}
    def root(key):
        while key!=parent[key]:parent[key]=parent[parent[key]];key=parent[key]
        return key
    seen={}
    for row in images:
        tokens=[('original',row['original_sha256']),('display',row['display_sha256'])]
        for token in tokens:
            if token in seen:
                a,b=root(row['id']),root(seen[token]);parent[max(a,b)]=min(a,b)
            else:seen[token]=row['id']
    groups=defaultdict(list)
    for row in images:
        if row['state'] in ('annotated','no_target'):groups[root(row['id'])].append(row)
    return list(groups.values())


def make_assignments(images, classes, ratios, seed):
    groups=grouped_images(images);active=[s for s in SUBSETS if ratios[s]]
    if len(groups)<len(active):
        raise ProjectError(422,'SPLIT_GROUPS_INSUFFICIENT',f'有效图片仅有 {len(groups)} 个独立重复图片组，无法分成 {len(active)} 个非空集合。请补充独立图片。')
    def rank(group):return digest([seed,sorted(row['id'] for row in group)])
    groups.sort(key=rank)
    covered=[{box['class_id'] for row in group for box in row['boxes']} for group in groups]
    # Declared classes without samples do not prevent using the labeled subset.
    all_classes={row['id'] for row in classes} & set().union(*covered)
    frequencies={cid:sum(cid in group for group in covered) for cid in all_classes}
    if any(count==0 for count in frequencies.values()):
        raise ProjectError(422,'SPLIT_CLASS_MISSING','存在没有有效目标框的类别，训练集无法覆盖。')
    # Reserve a positive validation group, then cover every class in training.
    train=None;validation=None
    for candidate, found in enumerate(covered):
        if not found or any(frequencies[cid]==1 for cid in found):continue
        chosen=set();missing=set(all_classes)
        while missing:
            options=[i for i in range(len(groups)) if i!=candidate and covered[i]&missing]
            if not options:break
            best=min(options,key=lambda i:(-len(covered[i]&missing),len(groups[i]),rank(groups[i])))
            chosen.add(best);missing-=covered[best]
        if not missing and len(groups)-len(chosen)-1 >= (1 if ratios['test'] else 0):
            train=chosen;validation=candidate;break
    if train is None:
        raise ProjectError(422,'SPLIT_COVERAGE_UNAVAILABLE','当前分组中未找到同时覆盖全部训练类别、且验证集含目标的方案。请补充独立目标样本、检查重复图片，或调整种子后重试。')
    count=len([row for row in images if row['state'] in ('annotated','no_target')])
    targets={s:count*ratios[s]/100 for s in SUBSETS}
    owners={i:'train' for i in train};owners[validation]='val'
    if ratios['test']:
        candidates=[i for i in range(len(groups)) if i not in owners]
        owners[min(candidates,key=lambda i:(abs(len(groups[i])-targets['test']),rank(groups[i]))) ]='test'
    counts=Counter()
    for i,subset in owners.items():counts[subset]+=len(groups[i])
    remaining=sorted((i for i in range(len(groups)) if i not in owners),key=lambda i:(-len(groups[i]),rank(groups[i])))
    for i in remaining:
        size=len(groups[i])
        def cost(s):return ((counts[s]+size-targets[s])**2-(counts[s]-targets[s])**2)/max(targets[s],1)
        subset=min(active,key=lambda s:(cost(s),SUBSETS.index(s)))
        owners[i]=subset;counts[subset]+=size
    return {row['id']:owners[i] for i,group in enumerate(groups) for row in group}


def make_class_assignments(images, classes, ratios, seed):
    groups=grouped_images(images)
    active=[subset for subset in SUBSETS if ratios[subset]]
    if len(groups)<len(active):
        raise ProjectError(422,'SPLIT_GROUPS_INSUFFICIENT','独立图片数量不足，无法生成非空集合。')
    groups.sort(key=lambda group:digest([seed,sorted(row['id'] for row in group)]))
    quantities=[Counter(box['class_id'] for row in group for box in row['boxes']) for group in groups]
    totals=sum(quantities,Counter())
    categories={category['id']:category['name'] for category in classes if totals[category['id']]}
    candidates={cid:[i for i,counts in enumerate(quantities) if counts[cid]] for cid in categories}
    rare=[categories[cid] for cid,indices in candidates.items() if len(indices)<2]
    if rare:
        raise ProjectError(422,'SPLIT_CLASS_SAMPLES_INSUFFICIENT','以下类别不足2个独立图片组，无法同时覆盖训练和验证：'+'、'.join(rare)+'。请补充独立图片。')
    # Satisfy both coverage constraints before balancing instance counts. Explicit
    # search also handles overlapping labels where a greedy reservation can fail.
    stack=[{}]
    visited=0
    started=time.monotonic()
    owners=None
    while stack:
        current=stack.pop();visited+=1
        if visited>20000 or time.monotonic()-started>4:
            raise ProjectError(422,'SPLIT_SEARCH_LIMIT','类别共现关系复杂，覆盖方案搜索达到限时。请调整种子或增加独立样本后重试，原划分已保留。')
        covered={subset:set().union(*(set(quantities[i]) for i,owner in current.items() if owner==subset)) for subset in ('train','val')}
        needs=[(subset,cid) for subset in ('train','val') for cid in categories if cid not in covered[subset]]
        if not needs:
            if ratios['test'] and len(current)==len(groups):continue
            owners=current;break
        choices=[(subset,cid,[i for i in candidates[cid] if i not in current]) for subset,cid in needs]
        if any(not indices for _,_,indices in choices):continue
        subset,cid,indices=min(choices,key=lambda item:(len(item[2]),item[1],item[0]))
        missing=set(categories)-covered[subset]
        def reservation_cost(index):
            gap=sum(abs(number-totals[label]*ratios[subset]/100)/max(totals[label],1)
                    for label,number in quantities[index].items() if label in missing)
            return -len(set(quantities[index])&missing),gap,len(groups[index]),index
        indices.sort(key=reservation_cost)
        for index in reversed(indices):stack.append({**current,index:subset})
    if owners is None:
        raise ProjectError(422,'SPLIT_CLASS_COVERAGE_UNAVAILABLE','当前类别共现及重复图片分组无法同时覆盖训练、验证并满足非空集合要求。请补充独立样本或调整测试集比例。')
    image_total=sum(map(len,groups))
    target_images={subset:image_total*ratios[subset]/100 for subset in active}
    if ratios['test']:
        remaining=[i for i in range(len(groups)) if i not in owners]
        owners[min(remaining,key=lambda i:(abs(len(groups[i])-target_images['test']),i))]='test'
    counts={subset:Counter() for subset in active};image_counts=Counter()
    for index,subset in owners.items():counts[subset].update(quantities[index]);image_counts[subset]+=len(groups[index])
    remaining=sorted((i for i in range(len(groups)) if i not in owners),key=lambda i:(-sum(quantities[i].values()),-len(groups[i]),i))
    for index in remaining:
        def cost(subset):
            result=0
            for cid,number in quantities[index].items():
                target=totals[cid]*ratios[subset]/100
                before=counts[subset][cid]-target
                result+=((before+number)**2-before**2)/max(totals[cid],1)
            before=image_counts[subset]-target_images[subset]
            return result+0.1*((before+len(groups[index]))**2-before**2)/max(image_total,1)
        subset=min(active,key=lambda subset:(cost(subset),SUBSETS.index(subset)))
        owners[index]=subset;counts[subset].update(quantities[index]);image_counts[subset]+=len(groups[index])
    return {row['id']:owners[i] for i,group in enumerate(groups) for row in group}


class PreparationRepository:
    def __init__(self, storage):
        self.storage=storage;self.tasks=TaskRepository(storage);self.assets=ImageRepository(storage)

    def inspect(self, project_id, task_id, db):
        task=self.tasks.get(project_id,task_id,db)
        classes=[dict(row) for row in db.execute('SELECT id,name FROM task_classes WHERE task_id=? ORDER BY created_at,id',(task_id,))]
        class_ids={row['id'] for row in classes}
        rows=[dict(row) for row in db.execute('''SELECT i.*,COALESCE(a.state,'unannotated') AS state
            FROM task_images ti JOIN images i ON i.id=ti.image_id AND i.dataset_id=ti.dataset_id
            LEFT JOIN task_annotations a ON a.task_id=ti.task_id AND a.image_id=ti.image_id
            WHERE ti.task_id=? ORDER BY i.id''',(task_id,))]
        boxes=defaultdict(list)
        for row in db.execute('SELECT image_id,class_id,x,y,width,height FROM detection_boxes WHERE task_id=? ORDER BY image_id,position',(task_id,)):
            value=dict(row);image_id=value.pop('image_id');boxes[image_id].append(value)
        if task['task_type']=='segment':
            sizes={r['id']:(r['width'],r['height']) for r in rows}
            for p in db.execute('SELECT image_id,class_id,points_json FROM segmentation_polygons WHERE task_id=? ORDER BY image_id,position',(task_id,)):
                try:shape=polygon(json.loads(p['points_json']),*sizes[p['image_id']])
                except (ValueError,KeyError):raise ProjectError(409,'INVALID_POLYGON','分割标注损坏，请检查并重新保存相关图片。')
                boxes[p['image_id']].append({'class_id':p['class_id'],**shape})
        issues=[];asset_signatures={};class_counts=Counter();class_images=defaultdict(set)
        def issue(code,message,row=None,severity='error'):
            issues.append({'code':code,'message':message,'severity':severity,'image_id':row['id'] if row else None,'filename':row['filename'] if row else None})
        for row in rows:
            row['boxes']=boxes[row['id']]
            from .roi import present,bounds,image_info
            area=bounds(db,row)
            if area:
                from .annotations import AnnotationRepository
                projected=AnnotationRepository(self.storage).get(project_id,task_id,row['id'],db)
                row.update(state=projected['state'],boxes=projected['boxes'],roi=area,source_width=row['width'],source_height=row['height'])
                row['width']=area['width'];row['height']=area['height']

            # Only explicit labels (including confirmed negatives) are training inputs.
            # Unannotated/excluded assets and drafts must not block this subset.
            if row['state'] not in ('annotated','no_target'):continue
            if row['state']=='annotated' and not row['boxes']:issue('EMPTY_ANNOTATION','已标注图片没有目标框。',row)
            if row['state']=='no_target' and row['boxes']:issue('STATE_BOX_CONFLICT','图片状态与目标框不一致。',row)
            for box in row['boxes']:
                numbers=[box[k] for k in ('x','y','width','height')]
                valid=all(type(v) in (float,int) and math.isfinite(v) for v in numbers)
                if valid:
                    x,y,w,h=numbers;valid=x>=0 and y>=0 and w>0 and h>0 and x+w<=row['width'] and y+h<=row['height']
                if not valid:issue('INVALID_BOX','存在越界、零面积或非法坐标框。',row)
                if box['class_id'] not in class_ids:issue('INVALID_CLASS','框引用的类别不属于当前任务。',row)
                elif valid and row['state']=='annotated':
                    class_counts[box['class_id']]+=1;class_images[box['class_id']].add(row['id'])
            signatures=[]
            for kind,field in [('original','original_sha256'),('view','display_sha256')]:
                try:
                    path,_,_=self.assets.file(project_id,task['dataset_id'],row['id'],kind)
                    actual=hashlib.sha256(path.read_bytes()).hexdigest();signatures.append(actual)
                    if actual!=row[field] and not (kind=='view' and row.get('roi')):issue('ASSET_CHANGED',f'{"原图" if kind=="original" else "预览图"}校验值不匹配，请检查文件。',row)
                    if kind=='view' and row.get('roi'):row['display_sha256']=actual
                    with Image.open(path) as picture:
                        if kind=='view' and picture.size!=(row['width'],row['height']):issue('IMAGE_DIMENSIONS','方向统一后的图片尺寸与记录不一致。',row)
                        picture.verify()
                except (OSError,ValueError,ProjectError):
                    signatures.append('unavailable');issue('ASSET_UNAVAILABLE',f'{"原图" if kind=="original" else "预览图"}缺失或无法解码。',row)
            asset_signatures[row['id']]=signatures
        eligible=[row for row in rows if row['state'] in ('annotated','no_target')]
        if not rows:issue('NO_IMAGES','任务没有关联图片，请导入并同步图片。')
        if not classes:issue('NO_CLASSES','任务尚未定义类别。')
        if not eligible:issue('NO_ELIGIBLE_IMAGES','没有已标注或确认无目标的有效图片。')
        for category in classes:
            if eligible and not class_counts[category['id']]:issue('UNUSED_CLASS',f'类别“{category["name"]}”暂无有效目标框，本次不要求该类别覆盖。',severity='warning')
        if eligible and not any(row['state']=='annotated' for row in eligible):issue('NO_POSITIVE_IMAGES','训练集和验证集均须包含目标标注，当前只有无目标图片。')
        content={'task_id':task_id,'dataset_id':task['dataset_id'],'type':task['task_type'],'classes':classes,
                 'images':[{k:row[k] for k in ('id','original_sha256','display_sha256','width','height','state','boxes')} | {'roi':row.get('roi')} for row in rows],
                 'assets':asset_signatures}
        fingerprint=digest(content)
        states=Counter(row['state'] for row in rows)
        report={'fingerprint':fingerprint,'image_count':len(rows),'eligible_count':len(eligible),'group_count':len(grouped_images(rows)),
                'states':{state:states[state] for state in ('unannotated','annotated','no_target','excluded')},
                'classes':[{**category,'box_count':class_counts[category['id']],'image_count':len(class_images[category['id']])} for category in classes],
                'issues':issues,'error_count':sum(i['severity']=='error' for i in issues),'warning_count':sum(i['severity']=='warning' for i in issues)}
        return report,rows,classes

    @staticmethod
    def distribution(assignments, rows, classes):
        stats={s:{'image_count':0,'positive_count':0,'no_target_count':0,'classes':{c['id']:0 for c in classes}} for s in SUBSETS}
        lookup={row['id']:row for row in rows}
        for image_id,subset in assignments.items():
            if image_id not in lookup:continue
            row=lookup[image_id];item=stats[subset];item['image_count']+=1
            item['positive_count']+=row['state']=='annotated';item['no_target_count']+=row['state']=='no_target'
            for box in row['boxes']:
                if box['class_id'] in item['classes']:item['classes'][box['class_id']]+=1
        return stats

    def read(self, project_id, task_id):
        with self.storage.connect() as db:
            db.execute('BEGIN');report,rows,classes=self.inspect(project_id,task_id,db)
            split=self.saved(db,task_id,report,rows,classes)
            return {'report':report,'split':split}

    def saved(self, db, task_id, report, rows, classes):
        row=db.execute('SELECT * FROM task_splits WHERE task_id=?',(task_id,)).fetchone()
        if row is None:return None
        result=dict(row);mapping={item['image_id']:item['subset'] for item in db.execute('SELECT * FROM task_split_images WHERE task_id=?',(task_id,))}
        result['stale']=result['fingerprint']!=report['fingerprint']
        result['distribution']=self.distribution(mapping,rows,classes)
        result['images']=sorted([{'image_id':row['id'],'filename':row['filename'],'group_id':row['group_id'],'subset':mapping[row['id']]} for row in rows if row['id'] in mapping],
                                key=lambda row:(SUBSETS.index(row['subset']),row['filename'].casefold(),row['image_id']))
        result['warnings']=[]
        if not result['stale']:
            for subset in ('val','test'):
                if result[f'{subset}_percent']:
                    missing=[c['name'] for c in classes if not result['distribution'][subset]['classes'][c['id']]]
                    if missing:result['warnings'].append(f'{"验证集" if subset=="val" else "测试集"}缺少类别：'+ '、'.join(missing)+'，不能据此宣称这些类别已验证。')
        return result

    def generate(self, project_id, task_id, values):
        ratios,seed=validate_config(values)
        strategy=values.get('strategy','images')
        if strategy not in ('images','classes'):raise ProjectError(422,'INVALID_SPLIT','请选择按图片数量或按缺陷类别分层。')
        algorithm=CLASS_ALGORITHM if strategy=='classes' else ALGORITHM
        revision=values.get('revision')
        if type(revision) is not int or revision<0:raise ProjectError(422,'INVALID_REVISION','划分修订无效，请重新检查。')
        with self.storage.connect() as db:
            db.execute('BEGIN IMMEDIATE');report,rows,classes=self.inspect(project_id,task_id,db)
            current=db.execute('SELECT revision,created_at FROM task_splits WHERE task_id=?',(task_id,)).fetchone()
            prior=db.execute('SELECT * FROM task_splits WHERE task_id=?',(task_id,)).fetchone()
            if prior and prior['algorithm']==algorithm and prior['fingerprint']==report['fingerprint']==values.get('fingerprint') and prior['seed']==seed and all(prior[k+'_percent']==ratios[k] for k in SUBSETS) and revision in (prior['revision'],prior['revision']-1):return {'report':report,'split':self.saved(db,task_id,report,rows,classes)}
            if revision!=(current['revision'] if current else 0):raise ProjectError(409,'REVISION_CONFLICT','划分已在其他页面变化，请重新检查后再生成。')
            if values.get('fingerprint')!=report['fingerprint']:raise ProjectError(409,'PREPARATION_CHANGED','类别、标注、图片或ROI已变化，请重新检查后再生成。')
            if report['error_count']:raise ProjectError(422,'DATA_NOT_READY','数据检查存在阻断项，请先处理后再划分。')
            allocator=make_class_assignments if strategy=='classes' else make_assignments
            mapping=allocator(rows,classes,ratios,seed);now=timestamp()
            db.execute('''INSERT INTO task_splits VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET
                revision=excluded.revision,fingerprint=excluded.fingerprint,seed=excluded.seed,train_percent=excluded.train_percent,
                val_percent=excluded.val_percent,test_percent=excluded.test_percent,algorithm=excluded.algorithm,updated_at=excluded.updated_at''',
                (task_id,revision+1,report['fingerprint'],seed,ratios['train'],ratios['val'],ratios['test'],algorithm,current['created_at'] if current else now,now))
            db.execute('DELETE FROM task_split_images WHERE task_id=?',(task_id,))
            db.executemany('INSERT INTO task_split_images VALUES(?,?,?)',[(task_id,i,s) for i,s in sorted(mapping.items())])
            db.execute('UPDATE vision_tasks SET revision=revision+1,updated_at=? WHERE id=?',(now,task_id))
            return {'report':report,'split':self.saved(db,task_id,report,rows,classes)}
