"""
01
数据的加载部分,由于数据缺乏,暂时使用视频加载得到
"""

from utils.datatype.OriginalFrames import OriginalFrames
from utils.datatype.Frame import Frame
from utils.load.video_load import read_video
import


def data_load_from_video(path):
    """
    从视频中加载模拟数据
    Args:
        path(str):视频的地址
    Returns:
        OriginalFrames:根据视频加载的Frames组合
    """
    fps, frame_list = read_video(path)
    original_frames = OriginalFrames()
    time