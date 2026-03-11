#计算图片相关属性用于关键帧的筛选

import cv2
from utils.datatype.Frame import Frame

def image_clarity(Frame, scale = 0.5):
    """
    计算图像的清晰度
    Args:
        Frame(Frame):当前帧
        scale(float):缩放比例
    Return:
        float:清晰度得分
    """
    img = Frame.image   #[3, h, w]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    resized = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    laplacian = cv2.Laplacian(resized, cv2.CV_64F)
    score = laplacian.var()   #使用方差作为最终的得分

    return float(score)


def overleap_degree(Frame1, Frame2, scale = 0.5, max_features = 500, distance_rate = 0.75):
    """
    计算两张图片的重叠度
    Args:
        Frame1(Frame):第一帧
        Frame2(Frame):第二帧
    Return:
        int:重叠对数量
    """
    img1 = Frame1.image
    img2 = Frame2.image

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    resized1 = cv2.resize(gray1, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(gray2, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    orb = ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(resized1, None)
    kp2, des2 = orb.detectAndCompute(resized2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)     #使用汉明距离表示相似度,越小越相似
    matches = bf.knnMatch(des1, des2, k=2)   #返回值按顺序包括从大到小的两组匹配点

    n = 0
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < distance_rate * n.distance:  #确保匹配是有效的
                n += 1
    
    return n


def ave_move(Frame1, Frame2):
    """
    计算平均移动距离
    Args:
        Frame1(Frame):第一帧
        Frame2(Frame):第二帧
    Return:
        float:平均移动距离
    """
    #TODO: 写完这段