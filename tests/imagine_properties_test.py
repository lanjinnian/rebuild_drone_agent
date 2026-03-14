"""
测试utils中图片属性相关的功能是否完善
"""

from utils.calculate.properties import *
from utils.datatype.Frame import Frame
import cv2
import numpy as np

IMAGE_PATH_1 = "Z:/rebuild_drone_agent/data/examples/picture/02_H_4/000000.png"
IMAGE_PATH_2 = "Z:/rebuild_drone_agent/data/examples/picture/02_H_4/000003.png"

Frame1 = Frame(image=cv2.imread(IMAGE_PATH_1), image_id=0, timestamp=0, location=None)
Frame2 = Frame(image=cv2.imread(IMAGE_PATH_2), image_id=1, timestamp=1, location=None)

clarity1 = image_clarity(Frame1.image)
clarity2 = image_clarity(Frame2.image)

overleap_point1 = overleap_point(Frame1.image, Frame2.image)

ave_move1 = ave_move(Frame1.image, Frame2.image)

time_distance1 = time_distance(Frame1.timestamp, Frame2.timestamp)

print(f"第一张照片清晰度分数为{clarity1},第二张照片清晰度分数为{clarity2},两张图片的重叠率为{overleap_point1},平均移动距离{ave_move1},时间距离{time_distance1}")
