"""Read the immutable training size for new evaluation and inference requests."""
from .storage import ProjectError


def model_imgsz(model):
    size=model['manifest']['training_snapshot']['requested_parameters']['imgsz']
    if type(size) is not int or not 320<=size<=2048 or size%32:
        raise ProjectError(409,'MODEL_INPUT_SIZE_UNSUPPORTED','模型训练尺寸超出当前支持范围，请保留模型并检查训练快照。')
    return size
