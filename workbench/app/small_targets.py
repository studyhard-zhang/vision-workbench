"""Read-only projected target sizes and bounded experiment guidance, not tuning results."""
import math

SIZES=(640,1024,1280,1536,1824,2048)
def analyze(manifest,input_size,kind='detect',adamw_verified=False):
    train=[image for image in manifest['images'] if image['subset']=='train'];fractions=[];thin=0
    for image in train:
        for box in image['boxes']:
            short=min(box['width'],box['height']);long=max(box['width'],box['height'])
            if short>0:
                fractions.append(short/max(image['width'],image['height']))
                if long/short>=10:thin+=1
    fractions.sort();p10=fractions[int((len(fractions)-1)*.1)] if fractions else None
    required=math.ceil(16/p10/32)*32 if p10 else None
    suggested=next((size for size in SIZES if size>=required),2048) if required else None
    def at(size):
        return {'imgsz':size,'p10_short_pixels':round(p10*size,2) if p10 else None,'under4':sum(value*size<4 for value in fractions),'under8':sum(value*size<8 for value in fractions),'under16':sum(value*size<16 for value in fractions)}
    current=at(input_size)
    guidance=[
                '先确认原图中目标确有细节；放大标注不会增加原始像素。统一ROI可减少无关背景，调整后重新划分生成新版本，并在相同区域验证推理。',
                '先固定数据版本、划分、模型和随机种子，只改变输入尺寸比较；更大尺寸更耗显存。若2048仍过细，优先改善采集或ROI。切片训练/推理需要单独实现，不是当前的自动功能。',
                '优化器auto会覆盖初始学习率与动量。需要比较学习率时使用显式AdamW，可先比较lr0=0.001、0.0003、0.0001，momentum=0.9；这些是待验证的实验候选，不是最佳值。',
                ('当前模式的AdamW已有运行证据，可在高级参数手动设置；仍需实际验证精度。' if adamw_verified and kind=='detect' else '当前模式未开放显式AdamW运行；保留现有验证门禁。本轮不自动调用GPU验证，也不把auto回传的AdamW当成显式AdamW已验证。'),
                '先沿用weight_decay=0.0005、lrf=0.01、warmup_epochs=3等运行库默认项，避免同时改变多项；这些参数当前未作为可编辑项开放。少样本可先比较100轮与早停30轮，留意过拟合，不盲目增加轮数。',
                '黑点/发丝应标全，并补充容易混淆的纹理、灰尘、反光和无目标图片。当前受控方案已关闭Mosaic及随机缩放，避免过早把微小目标缩没；是否启用增强需单独对比。',
                '用固定验证图片比较Precision、Recall、mAP及漏检/误检实例。推理置信度只改变展示/保留结果的取舍，不改变模型训练质量；best.pt验证指标与末轮数值分开看。'
            ]
    if kind=='segment':guidance.insert(2,'当前任务为实例分割。运行库默认mask_ratio=4，会对训练掩膜降采样；细黑点/发丝可另行评估mask_ratio=1或2，但更耗显存，需先完成兼容验证，当前未开放修改。')
    return {'target_count':len(fractions),'thin_box_count':thin,'negative_images':sum(not image['boxes'] for image in train),'train_images':len(train),
            'current':current,'sizes':[at(size) for size in sorted({512,input_size,*SIZES})],'suggested_imgsz':suggested,'size_limit_insufficient':bool(required and required>2048),
            'scope':'冻结训练集的标注框短边按等比例缩放估算；8/16px仅作排查起点，不是可检出保证。斜向或弯曲发丝的真实宽度可能远小于框短边。',
            'guidance':guidance}
