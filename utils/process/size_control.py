"""
控制修改尺寸相关的功能
"""


import cv2
import numpy as np
import math


def size_process(img, size):
    """
    将图片修改为便于vggt处理的形状
    Args:
        img(np.array):图像
        size(int):裁剪后正方形边长
    Returns:
        np.array:裁剪后的图像数据
    Notes:
        - 截断后的图片为正方形
        - 下采样后以左上角作为起点,保留对应尺寸的正方形
    """
    h, w = img.shape[:2]
    scale = size / min(w, h)
    
    new_w, new_h = math.ceil(w*scale), math.ceil(h*scale)
    if scale < 1:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif scale > 1:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img
    
    # 截断后的图片为正方形，下采样后以右上角作为起点
    cropped_img = resized[0 : size, 0: size]

    return cropped_img
