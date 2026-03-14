"""
03
通过图像的指标筛选剔除冗余数据
"""


CLARITY_LIMIT = 2000



def remove_redundancy_by_classify(original_frames):
    """
    通过图像的指标筛选剔除冗余数据
    Args:
        original_frames(OriginalFrames): 原始帧序列对象
    Returns:
        OriginalFrames: 剔除冗余数据后的帧序列对象
    """
    #TODO
