"""
01
数据的加载部分,由于数据缺乏,暂时使用视频加载得到
"""

from utils.datatype.OriginalFrames import OriginalFrames
from utils.datatype.Frame import Frame
from utils.load.video_load import read_video
import time


def data_load_from_video(path):
    """
    从视频中加载模拟数据
    Args:
        path(str):视频的地址
    Returns:
        OriginalFrames:根据视频加载的Frames组合
    Notes:
        - 使用当前时间模拟时间戳
    """
    fps, frame_list = read_video(path)
    original_frames = OriginalFrames()
    current_time = int(time.time() * 1000)
    for i, image in enumerate(frame_list):
        new_frame = Frame(image=image, image_id=i, timestamp=current_time, location=None)
        original_frames.add_frame(new_frame)
        current_time += 1000 / fps
    return original_frames