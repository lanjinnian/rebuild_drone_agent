from pathlib import Path

import cv2

from src.datatype import BaseFrame, OriginalFrames


def load_original_frames_from_video(video_path: str | Path) -> OriginalFrames:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / 6)) if fps and fps > 0 else 5
    frame_interval = max(frame_interval, 1)

    original_frames = OriginalFrames()
    source_frame_index = 0
    frame_id = 0

    try:
        while True:
            success, image = capture.read()
            if not success:
                break

            if source_frame_index % frame_interval == 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame = BaseFrame(
                    id=frame_id,
                    image=image,
                    gps_location=None,
                )
                original_frames.add_frame(frame)
                frame_id += 1

            source_frame_index += 1
    finally:
        capture.release()

    return original_frames
