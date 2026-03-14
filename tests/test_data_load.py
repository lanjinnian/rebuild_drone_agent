"""
测试 data_load_from_video 功能
Created by Gemini
"""
import os
from core.data_process.data_load import data_load_from_video

def test_data_load_from_video():
    """测试从视频中加载帧序列的功能"""
    test_video_path = "Z:/rebuild_drone_agent/data/examples/video/H01.mp4"
        
    frames_obj = data_load_from_video(test_video_path)
    
    # 验证是否成功返回了对象
    assert frames_obj is not None, "返回的 OriginalFrames 对象不应为 None"
    assert hasattr(frames_obj, 'frames'), "返回的对象应包含 'frames' 属性"
    
    # 验证内部提取的帧数
    assert len(frames_obj.frames) > 0, "从视频提取的帧数不应该为0"

    print(f"成功加载视频的帧序列，总共包含了 {len(frames_obj.frames)} 帧。")

if __name__ == "__main__":
    # 如果直接运行脚本，则执行测试函数
    test_data_load_from_video()
