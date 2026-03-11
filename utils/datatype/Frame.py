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


    def save(self, filename=None, directory=None):
        """
        保存帧为 .npy 文件
        
        Args:
            filename (str, optional): 文件名（不需要扩展名，会自动添加 .npy）
            directory (str, optional): 保存目录，默认为当前目录
        
        Raises:
            ValueError: 如果图像数据为 None，或未提供文件名
        """
        if self.image is None:
            raise ValueError(f"Frame {self.id} has no image data to save")
        
        if filename is None:
            raise ValueError("Must provide 'filename'")
        
        if directory is None:
            directory = "."
        
        if not filename.endswith('.npy'):
            filename = f"{filename}.npy"
        
        filepath = os.path.join(directory, filename)
        np.save(filepath, self.image)


    def get_metadata(self):
        """
        获取帧的元数据
        
        Returns:
            dict: 包含 id, timestamp, location, shape 的字典
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "location": self.location,
            "shape": self.image.shape if self.image is not None else None
        }
    