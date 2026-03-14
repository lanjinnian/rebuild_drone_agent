"""
三维重建中的帧基类
"""

import cv2
import numpy as np
import os

class Frame:
    def __init__(self, image, image_id, timestamp, location):
        self.image = image              #RGB格式图像数据
        self.id = image_id              #帧的全局标识ID
        self.timestamp = timestamp      #帧的时间戳
        self.location = location        #帧的GPS信息


    def __repr__(self):
        shape = self.image.shape if self.image is not None else None
        return f"Frame(id={self.id}, shape={shape})"


    def show(self):
        """
        使用 OpenCV 显示图像
        """
        if self.image is None:
            raise ValueError(f"Frame {self.id} has no image data to display")
        
        cv2.imshow(f"Frame {self.id}", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    