"""Read-only dataset audit. Python 3.10+, Pillow; no torch/Ultralytics/network calls.

python verify_dataset.py data-v001.zip --report data-check.json
Accepts either the original ZIP or an extracted dataset directory. Never extracts ZIPs.
"""
import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
import warnings
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

AUDITOR_VERSION='1.0.0'
MAX_METADATA=64*1024*1024
MAX_IMAGE=256*1024*1024
SUBSETS=('train','val','test')


class AuditError(Exception):pass


def require(condition, message):
    if not condition:raise AuditError(message)


def safe_name(value):
    require(isinstance(value,str) and value and '\\' not in value and ':' not in value,'数据包包含非法文件路径。')
    path=PurePosixPath(value)
    require(not path.is_absolute() and str(path)==value and all(p not in ('.','..') for p in path.parts),'数据包包含非法文件路径。')
    return value


class Reader:
    def __init__(self, source):
        self.archive=None;self.root=None
        if isinstance(source,(str,Path)) and Path(source).is_dir():
            self.root=Path(source).resolve();self.names=set()
            for path in self.root.rglob('*'):
                require(not path.is_symlink(),'检查目录不能包含符号链接。')
                if path.is_file():self.names.add(safe_name(path.relative_to(self.root).as_posix()))
        else:
            self.archive=zipfile.ZipFile(source);names=[]
            for info in self.archive.infolist():
                if info.is_dir():continue
                names.append(safe_name(info.filename))
                require((info.external_attr>>16)&0o170000 != 0o120000,'数据包不能包含符号链接。')
                require(not info.flag_bits&1,'不支持加密数据包。')
            require(len(names)==len(set(names)),'数据包存在重复文件名。');self.names=set(names)
        require(len(self.names)<=100000,'文件数量超出当前检查上限。')

    def close(self):
        if self.archive:self.archive.close()

    def read(self, name, limit=MAX_METADATA):
        safe_name(name);require(name in self.names,'文件缺失：'+name)
        if self.archive:
            require(self.archive.getinfo(name).file_size<=limit,'文件超过当前检查上限：'+name)
            with self.archive.open(name) as stream:raw=stream.read(limit+1)
        else:
            path=(self.root/name).resolve();require(path.is_relative_to(self.root),'文件路径超出数据目录。')
            require(path.stat().st_size<=limit,'文件超过当前检查上限：'+name)
            with path.open('rb') as stream:raw=stream.read(limit+1)
        require(len(raw)<=limit,'文件超过当前检查上限：'+name)
        return raw


def image_module():
    try:from PIL import Image
    except ModuleNotFoundError:
        # Only the existing workbench runtime may use its vendored Pillow.
        # Python 3.10 on the laptop uses its own Pillow 11, never Pillow 13.
        vendor=Path(__file__).resolve().parent.parent/'.vendor'
        if sys.version_info>=(3,11) and Path(__file__).resolve().parent.name=='tools' and (vendor/'PIL').is_dir():
            sys.path.append(str(vendor));from PIL import Image
        else:raise AuditError('当前Python缺少Pillow，请使用已配置的工作台环境或笔记本ty2环境。')
    return Image


def finite_number(value):return type(value) in (int,float) and math.isfinite(value)


def polygon(points, width, height):
    if not isinstance(points,list) or not 3<=len(points)<=256:
        raise ValueError('每个多边形需要3至256个顶点。')
    for point in points:
        if not isinstance(point,list) or len(point)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in point):
            raise ValueError('顶点须为有限数值的[x,y]坐标。')
        if not 0<=point[0]<=width or not 0<=point[1]<=height:raise ValueError('多边形顶点必须位于图片内。')
    if len(set(map(tuple,points)))!=len(points):raise ValueError('顶点不能重复，闭合时无需再次填写首点。')
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,c):return abs(cross(a,b,c))<1e-9 and min(a[0],b[0])<=c[0]<=max(a[0],b[0]) and min(a[1],b[1])<=c[1]<=max(a[1],b[1])
    n=len(points)
    for i,a in enumerate(points):
        b=points[(i+1)%n]
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1):continue
            c,d=points[j],points[(j+1)%n]
            if (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b):
                raise ValueError('多边形不能自交或自接触，请调整顶点。')
    area=abs(sum(a[0]*points[(i+1)%n][1]-points[(i+1)%n][0]*a[1] for i,a in enumerate(points)))/2
    if area<1:raise ValueError('多边形面积至少为1平方像素。')
    xs,ys=zip(*points)
    return {'x':min(xs),'y':min(ys),'width':max(xs)-min(xs),'height':max(ys)-min(ys),'points':points}



def audit(source):
    started=time.perf_counter();reader=None
    report={'auditor_version':AUDITOR_VERSION,'checked_at':datetime.now(timezone.utc).isoformat(),
            'scope':'frozen-detection-data','passed':False,'training_executed':False,'checks':[],'stats':{}}
    def passed(code,message):report['checks'].append({'code':code,'status':'passed','message':message})
    try:
        reader=Reader(source)
        raw_manifest=reader.read('manifest.json');manifest=json.loads(raw_manifest)
        report['manifest_sha256']=hashlib.sha256(raw_manifest).hexdigest()
        require(manifest['format_version']==1 and manifest['label_format'] in ('yolo-detect-xywh-normalized-v1','yolo-segment-polygon-normalized-v1') and manifest['coordinate_system']=='canonical-exif-pixels','不支持的数据清单格式或坐标系。')
        segment=manifest['label_format']=='yolo-segment-polygon-normalized-v1'
        if segment:
            report['scope']='frozen-segmentation-data'
        require(type(manifest['number']) is int and manifest['number']>0,'版本序号无效。')
        for key in ('version_id','project_id','task_id','dataset_id'):
            require(isinstance(manifest[key],str) and re.fullmatch('[0-9a-f]{32}',manifest[key]),'版本身份无效：'+key)
        report.update(version_id=manifest['version_id'],number=manifest['number'])
        images=manifest['images'];classes=manifest['classes'];split=manifest['split']
        require(isinstance(images,list) and images and isinstance(classes,list) and classes,'版本没有有效图片或类别。')
        require([c['index'] for c in classes]==list(range(len(classes))) and all(type(c['index']) is int for c in classes),'类别编号必须从0开始连续。')
        indices={c['id']:c['index'] for c in classes}
        require(len(indices)==len(classes) and all(isinstance(c['name'],str) and c['name'] for c in classes),'类别ID重复或名称无效。')
        config=json.loads(reader.read('data.yaml'))
        expected_config={'train':'images/train','val':'images/val','names':[c['name'] for c in classes]}
        if split['test_percent']:expected_config['test']='images/test'
        require(config==expected_config,'data.yaml与冻结类别/集合不一致，或包含额外配置。')
        passed('identity_config','版本身份、连续类别映射及data.yaml一致。')

        expected_files={'manifest.json','data.yaml'};ids=set();paths=set()
        for image in images:
            iid=image['id'];subset=image['subset'];fmt=image['format']
            require(isinstance(iid,str) and re.fullmatch('[0-9a-f]{32}',iid) and iid not in ids,'图片ID无效或重复。');ids.add(iid)
            require(subset in SUBSETS and fmt in ('JPEG','PNG') and image['state'] in ('annotated','no_target'),'图片状态/格式/归属无效，未标注和排除不能纳入。')
            suffix='jpg' if fmt=='JPEG' else 'png'
            require(image['image_path']==f'images/{subset}/{iid}.{suffix}' and image['label_path']==f'labels/{subset}/{iid}.txt','图片与标签路径不匹配。')
            expected_files.update((image['image_path'],image['label_path']));paths.add(image['label_path'])
        require(reader.names==expected_files,'数据包存在缺失文件或未在冻结清单中的额外文件。')
        require(set(manifest['files'])==paths|{'data.yaml'},'冻结文件校验清单不完整。')
        for name,expected in manifest['files'].items():
            require(hashlib.sha256(reader.read(name)).hexdigest()==expected,'版本文件校验失败：'+name)
        passed('members_hashes','文件与冻结成员一一对应，标签与配置校验值一致。')

        Image=image_module();counts={s:Counter() for s in SUBSETS};class_counts={s:Counter() for s in SUBSETS}
        report['runtime']={'python':sys.version.split()[0],'pillow':getattr(Image,'__version__','unknown'),'platform':sys.platform}
        tokens={};sizes=[];total_bytes=sum(len(reader.read(n)) for n in ['manifest.json',*manifest['files']])
        for image in images:
            subset=image['subset'];w,h=image['width'],image['height']
            require(type(w) is int and type(h) is int and w>0 and h>0 and w*h<=40000000,'图片尺寸无效或超出4000万像素。')
            raw=reader.read(image['image_path'],MAX_IMAGE);total_bytes+=len(raw)
            require(hashlib.sha256(raw).hexdigest()==image['display_sha256'],'训练图片校验失败：'+image['filename'])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error',Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(raw)) as picture:
                        picture.verify()
                    with Image.open(io.BytesIO(raw)) as picture:
                        require(picture.size==(w,h) and picture.format==image['format'] and picture.getexif().get(274) in (None,1),'训练图方向、格式或尺寸与冻结记录不符。')
                        picture.load()
            except (Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
                raise AuditError('图片解码尺寸超出检查范围。') from exc
            sizes.append((w,h))
            label=reader.read(image['label_path']).decode('utf-8')
            boxes=image['boxes'];require(isinstance(boxes,list),'框清单格式无效。')
            require((image['state']=='annotated' and boxes) or (image['state']=='no_target' and not boxes),'状态与框不一致。')
            lines=label.splitlines();require(len(lines)==len(boxes),'目标框与标签行数不一致。')
            if image['state']=='no_target':require(label=='','无目标图片必须为空标签文件。')
            for line,box in zip(lines,boxes):
                values=line.split();require(bool(values) and values[0].isdigit() and (len(values)>=7 and len(values)%2==1 if segment else len(values)==5),'YOLO标签行格式无效。')
                x,y,bw,bh=[box[k] for k in ('x','y','width','height')]
                require(all(finite_number(v) for v in (x,y,bw,bh)) and x>=0 and y>=0 and bw>0 and bh>0 and x+bw<=w and y+bh<=h,'冻结框越界、非有限或零面积。')
                require(box['class_id'] in indices and int(values[0])==indices[box['class_id']],'标签类别与冻结映射不一致。')
                numbers=[float(v) for v in values[1:]];expected=((x+bw/2)/w,(y+bh/2)/h,bw/w,bh/h)
                if segment:
                    checked=polygon(box['points'],w,h)
                    require(all(math.isclose(box[k],checked[k],abs_tol=1e-6) for k in ('x','y','width','height')),'外接框与分割顶点不一致。')
                    expected=tuple(v for px,py in checked['points'] for v in (px/w,py/h))
                require(len(numbers)==len(expected) and all(math.isfinite(v) and (0<=v<=1 if segment else 0<v<=1) and math.isclose(v,e,rel_tol=1e-10,abs_tol=1e-12) for v,e in zip(numbers,expected)),'标签坐标与冻结标注不一致。')
                class_counts[subset][box['class_id']]+=1
            counts[subset]['image_count']+=1;counts[subset]['positive_count']+=image['state']=='annotated';counts[subset]['no_target_count']+=image['state']=='no_target'
            for key,value in [('original',image['original_sha256']),('display',image['display_sha256']),('group',image['group_id'])]:
                if not value:continue
                token=(key,value);require(token not in tokens or tokens[token]==subset,'同组或完全重复的图片跨集合。');tokens[token]=subset
        passed('canonical_images','训练图片哈希、方向、尺寸及完整解码通过。')
        passed('labels','正样本框与归一化YOLO标签一致，负样本为空标签。')

        ratios=[split[s+'_percent'] for s in SUBSETS]
        require(all(type(v) is int and 0<=v<=100 for v in ratios) and sum(ratios)==100 and ratios[0]>0 and ratios[1]>0,'划分比例无效。')
        present=set().union(*(set(c) for c in class_counts.values()))
        require(set(class_counts['train'])==present,'训练集未覆盖有效标注中出现的类别。')
        for subset in SUBSETS:
            stat=counts[subset]
            require(bool(stat['image_count'])==bool(split[subset+'_percent']),'集合非空状态与划分设置不符。')
            if subset in ('train','val'):require(stat['positive_count']>0,'训练集和验证集各需有效目标图片。')
            expected={k:stat[k] for k in ('image_count','positive_count','no_target_count')}
            expected['classes']={c['id']:class_counts[subset][c['id']] for c in classes}
            require(manifest['distribution'][subset]==expected,'冻结统计与实际图片/标签不一致。')
        passed('split_statistics','集合、分组防泄漏、训练类别覆盖与冻结统计一致。')
        report['stats']={'image_count':len(images),'class_count':len(classes),'box_count':sum(sum(c.values()) for c in class_counts.values()),
                         'no_target_count':sum(c['no_target_count'] for c in counts.values()),'bytes_checked':total_bytes,
                         'min_width':min(w for w,h in sizes),'max_width':max(w for w,h in sizes),'min_height':min(h for w,h in sizes),'max_height':max(h for w,h in sizes),
                         'subsets':{s:dict(image_count=counts[s]['image_count'],positive_count=counts[s]['positive_count'],no_target_count=counts[s]['no_target_count']) for s in SUBSETS}}
        report['passed']=True
    except (AuditError,OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,ArithmeticError,zipfile.BadZipFile,EOFError,SyntaxError) as exc:
        report['checks'].append({'code':'invalid_data','status':'failed','message':str(exc) or '数据包结构或文件无效。'})
    finally:
        if reader:reader.close()
        report['elapsed_seconds']=round(time.perf_counter()-started,4)
    return report


def main():
    parser=argparse.ArgumentParser(description='视训工作台：只读核对冻结检测数据，不启动训练、不联网。')
    parser.add_argument('source',type=Path,help='YOLO数据ZIP或解压后的目录')
    parser.add_argument('--report',type=Path,help='可选：保存JSON检查报告')
    args=parser.parse_args();result=audit(args.source);payload=json.dumps(result,ensure_ascii=False,indent=2)
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    print(payload)
    if args.report:
        source=args.source.resolve();output=args.report.resolve()
        if output==source or (source.is_dir() and output.is_relative_to(source)):
            parser.error('报告须保存在数据包/数据目录之外，不能覆盖输入。')
        args.report.write_text(payload+'\n',encoding='utf-8')
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
