#计算图片相关属性用于关键帧的筛选

import cv2
from utils.datatype.Frame import Frame
import math

def image_clarity(frame, scale = 0.5):
    """
    计算图像的清晰度
    Args:
        frame(Frame):当前帧
        scale(float):缩放比例
    Returns:
        float:清晰度得分
    """
    img = frame.image   #[h, w, 3]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    resized = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    laplacian = cv2.Laplacian(resized, cv2.CV_64F)
    score = laplacian.var()   #使用方差作为最终的得分

    return float(score)


def overleap_point(frame1, frame2, scale = 0.5, max_features = 500, distance_rate = 0.75):
    """
    计算两张图片的重叠点数量
    Args:
        frame1(Frame):第一帧
        frame2(Frame):第二帧
        scale(float):下采样倍率
        max_features(int):单张图片特征点最大数量
        distance_rate(float):有效特征点阈值
    Returns:
        int:重叠对数
    """
    #TODO:重复纹理
    #TODO:修改为重叠率
    #TODO:异常处理
    img1 = frame1.image
    img2 = frame2.image

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    resized1 = cv2.resize(gray1, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(gray2, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(resized1, None)
    kp2, des2 = orb.detectAndCompute(resized2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)     #使用汉明距离表示相似度,越小越相似
    matches = bf.knnMatch(des1, des2, k=2) 

    overleap_number = 0
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < distance_rate * n.distance:  #确保匹配是有效的
                overleap_number += 1
    
    return overleap_number


def ave_move(frame1, frame2, scale = 0.5, max_features = 500, distance_rate = 0.75):
    """
    计算平均移动距离
    Args:
        frame1(Frame):第一帧
        frame2(Frame):第二帧
        scale(float):下采样倍率
        max_features(int):单张图片特征点最大数量
        distance_rate(float):有效特征点阈值
    Returns:
        float:缩放后图像的平均移动距离
    """
    #TODO:异常处理
    #TODO:下采样还原
    img1 = frame1.image
    img2 = frame2.image

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    resized1 = cv2.resize(gray1, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(gray2, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(resized1, None)
    kp2, des2 = orb.detectAndCompute(resized2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)     #使用汉明距离表示相似度,越小越相似
    matches = bf.knnMatch(des1, des2, k=2) 

    useable_number = 0
    sum_distance = 0
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < distance_rate * n.distance:  #确保匹配是有效的
                useable_number += 1
                point1 = kp1[m.queryIdx].pt
                point2 = kp2[m.trainIdx].pt
                sum_distance += math.dist(point1, point2)
    try:
        return sum_distance / useable_number
    except ZeroDivisionError:
        return -1.0


def time_distance(frame1, frame2):
    """
    计算两帧之间的时间距离
    Args:
        frame1(Frame):第一帧
        frame2(Frame):第二帧
    Returns:
        int:时间戳差值
    """
    time1 = frame1.timestamp
    time2 = frame2.timestamp

    return abs(time2 - time1)


def information_gain(frame1, frame2):
    """
    计算两帧之间的信息增益
    Args:
        frame1(Frame):第一帧
        frame2(Frame):第二帧
    Returns:
        float:两帧之间的信息增益
    """
    #TODO:写完这段