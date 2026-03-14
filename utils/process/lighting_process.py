"""
光照预处理相关函数
"""


import cv2
import numpy as np
import torch


def control_lighting_by_clahe(img, clipLimit = 2.0, tileGridSize = (8, 8)):
    """
    使用CLAHE算法进行光照预处理
    Args:
        img (np.array): 输入图像
        clipLimit (float): 对比度限制以避免光照割裂(默认2.0)
        tileGridSize (tuple): 图像划分小块的大小(默认(8, 8))
    Returns:
        np.array: 预处理后的图像
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    if len(img.shape) == 2:
        new_img = clahe.apply(img)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        new_lab = cv2.merge((cl, a, b))
        new_img = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)
    return new_img


def control_lighting_by_msr(img, sigmas=[15, 80, 250], weights=None):
    """
    使用MSR算法进行光照预处理
    Args:
        img (np.array): 输入图像
        sigmas (list): 高斯模糊的尺度列表
        weights (list): 不同尺度的权重，默认均分
    Returns:
        np.array: 预处理后的图像
    """
    if len(img.shape) == 2:
        type =  'gray'
        processing_img = img
    else:
        type = 'lab'
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        processing_img, a, b = cv2.split(lab)

    img_float = processing_img.astype(np.float32)
    img_log = np.log1p(img_float)
    msr_out = np.zeros_like(img_float)
    
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
        
    for sigma, weight in zip(sigmas, weights):
        blurred = cv2.GaussianBlur(img_float, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        blurred_log = np.log1p(blurred)
        msr_out += weight * (img_log - blurred_log)
        
    msr_out = (msr_out - np.min(msr_out)) / (np.max(msr_out) - np.min(msr_out) + 1e-6) * 255.0
    msr_out = np.uint8(np.clip(msr_out, 0, 255))
        
    if type == 'gray':
        return msr_out
    else:
        new_lab = cv2.merge((msr_out, a, b))
        new_img = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)
        return new_img


def control_lighting_by_usi3d(img):
    """
    使用USI3D模型进行光照预处理
    Args:
        img (np.array): 输入图像
    Returns:
        np.array: 预处理后的图像
    """
    #TODO


def control_lighting_by_iclight(img):
    """
    使用ICLight模型进行光照预处理
    Args:
        img (np.array): 输入图像
    Returns:
        np.array: 预处理后的图像
    """
    #TODO