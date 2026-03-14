"""
视频读取相关的函数
"""


import cv2


def read_video(file_path):
    """
    读取视频
    Args:
        file_path(str):带读取视频数据
    Returns:
        fps:视频的帧率
        frame_list(list[np.ndarray]):视频帧列表
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {file_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)
    cap.release()
    
    return fps, frame_list
    