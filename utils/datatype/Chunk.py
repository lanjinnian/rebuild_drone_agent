"""
分块数据类型
"""

"""
数据预处理阶段Frame整体组合
"""

from utils.datatype.Frame import Frame
import numpy as np
import os

class OriginalFrames:
    def __init__(self, id):
        self.frame_number = 0       #帧的数量
        self.frame_id_list = []     #帧的id列表
        self.frames = []            #帧的列表
        self.id = id                #分块的id


    def add_frame(self, frame):
        """
        添加帧
        Args:
            frame(Frame):需要添加的帧
        """
        self.frame_number += 1
        self.frame_id_list.append(frame.id)
        self.frames.append(frame)


    def delete_frame(self, id):
        """
        删除帧
        Args:
            id(int):需要删除的帧的id
        """
        location = self.frame_id_list.index(id)
        self.frame_id_list.remove(id)
        self.frames.pop(location)


    def change_frame(self, id, new_frame):
        """
        修改特定帧
        Args:
            id(int):需要修改的帧的id
            new_frame(Frame):改动为该帧
        Raises:
            ValueError:新帧的id不匹配
        Notes:
            - 只修改帧的内容,不修改帧的id
        """
        location = self.frame_id_list.index(id)
        if new_frame.id == id:
            self.frames[location] = new_frame
        else:
            raise ValueError("新帧的id必须与原有的帧相匹配")
        

