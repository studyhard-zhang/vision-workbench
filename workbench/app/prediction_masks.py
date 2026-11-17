"""Bounds for JSON mask contours; no training or image runtime required."""
import math


def validate_segments(prediction,width,height):
    segments=prediction.get('segments')
    if not isinstance(segments,list) or len(segments)>256:raise ValueError('缺少分割轮廓或轮廓数量超限。')
    total=0
    for ring in segments:
        if not isinstance(ring,dict) or set(ring)!={'points','hole'} or type(ring['hole']) is not bool:raise ValueError('轮廓结构或孔洞标记无效。')
        points=ring['points']
        if not isinstance(points,list) or not 3<=len(points)<=4096:raise ValueError('预测轮廓顶点数无效。')
        total+=len(points)
        for point in points:
            if not isinstance(point,list) or len(point)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in point) or not 0<=point[0]<=width or not 0<=point[1]<=height:raise ValueError('预测轮廓坐标超出原图。')
    if total>32768:raise ValueError('单个目标的轮廓过于复杂。')
