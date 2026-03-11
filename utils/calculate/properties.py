#计算图片相关属性用于关键帧的筛选

import cv2
from utils.datatype.Frame import Frame

def image_clarity(Frame, scale = 0.5):
    """
    计算图像的清晰度
    Args:
        Frame:当前Frame数据
        scale:缩放比例
    Return:
        清晰度得分
    """
    img = Frame.image   #[3, h, w]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    resized = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    laplacian = cv2.Laplacian(resized, cv2.CV_64F)
    score = laplacian.var()   #使用方差作为最终的得分
    return float(score)


def overleap_degree(Frame1, Frame2):
    """
    计算两张图片的重叠度
    Args:
        Frame1:第一张图片
        Frame2:第二张图片
    Return:
        重叠度得分
    """
    img1 = Frame1.image
    img2 = Frame2.image
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    