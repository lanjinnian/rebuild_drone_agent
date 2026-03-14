"""
计算图片相关属性用于关键帧的筛选
image_clarity: 计算图像的清晰度
overleap_point: 计算两张图片的重叠点数量
ave_move: 计算图像的平均移动距离
time_distance: 计算两张图片的时间间隔
"""

import cv2
import numpy as np
import math

def image_clarity(img, scale = 0.5):
    """
    计算图像的清晰度
    Args:
        img(np.ndarray):当前帧图像 [h, w, 3]
        scale(float):缩放比例
    Returns:
        float:清晰度得分
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    resized = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    laplacian = cv2.Laplacian(resized, cv2.CV_64F)
    score = laplacian.var()   #使用方差作为最终的得分

    return float(score)


def overleap_point(img1, img2, scale = 0.5, max_features = 500, distance_rate = 0.75):
    """
    计算两张图片的重叠点数量
    Args:
        img1(np.ndarray):第一帧图像
        img2(np.ndarray):第二帧图像
        scale(float):下采样倍率
        max_features(int):单张图片特征点最大数量
        distance_rate(float):有效特征点阈值
    Returns:
        int:重叠对数
    """
    #TODO:重复纹理
    #TODO:修改为重叠率
    #TODO:异常处理

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    resized1 = cv2.resize(gray1, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(gray2, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    orb = cv2.ORB.create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(resized1, None)   # type: ignore
    kp2, des2 = orb.detectAndCompute(resized2, None)  # type: ignore

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)     #使用汉明距离表示相似度,越小越相似
    matches = bf.knnMatch(des1, des2, k=2) 

    overleap_number = 0
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < distance_rate * n.distance:  #确保匹配是有效的
                overleap_number += 1
    
    return overleap_number


def ave_move(img1, img2, scale = 0.5, max_features = 500, distance_rate = 0.75):
    """
    计算平均移动距离
    Args:
        img1(np.ndarray):第一帧图像
        img2(np.ndarray):第二帧图像
        scale(float):下采样倍率
        max_features(int):单张图片特征点最大数量
        distance_rate(float):有效特征点阈值
    Returns:
        float:缩放后图像的平均移动距离
    """
    #TODO:异常处理
    #TODO:下采样还原

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    resized1 = cv2.resize(gray1, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(gray2, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    orb = cv2.ORB.create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(resized1, None)  # type: ignore
    kp2, des2 = orb.detectAndCompute(resized2, None)  # type: ignore

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


def time_distance(time1, time2):
    """
    计算两帧之间的时间距离
    Args:
        time1(float or int):第一帧时间戳
        time2(float or int):第二帧时间戳
    Returns:
        float or int:时间戳差值
    """

    return abs(time2 - time1)

