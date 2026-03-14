"""
光照预处理相关函数
"""


import cv2


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
    new_img = clahe.apply(img)
    return new_img


def control_lighting_by_msr(img):
    """
    使用MSR算法进行光照预处理
    Args:
        img (np.array): 输入图像
    Returns:
        np.array: 预处理后的图像
    """
    #TODO


def control_lighting_by_usi3d(img):
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