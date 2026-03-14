"""
02
图像的预处理,包括尺寸和光照的标准化
"""


from utils.datatype.OriginalFrames import OriginalFrames
from utils.process.size_control import size_process


def image_process(original_frames):
    """
    图像预处理流程
    Args:
        original_frames(OriginalFrames):原始图像组
    Returns:
        OriginalFrames:预处理后的图像组
    """
    for frame in original_frames.frames :
        id = frame.id
        image = frame.image
        frame.image = size_process(image)
        original_frames.change_frame(id, frame)
    return original_frames