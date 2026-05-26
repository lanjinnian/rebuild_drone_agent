import logging
from collections.abc import Iterator
from pathlib import Path

import cv2

from config import VIDEO_EXTRACT_FPS, VIDEO_SEGMENT_SECONDS
from src.datatype import BaseFrame, OriginalFrames


logger = logging.getLogger(__name__)


def load_original_frames_from_video(video_path: str | Path) -> OriginalFrames:
    segments = list(
        iter_original_frame_segments_from_video(video_path, segment_seconds=None)
    )
    return segments[0] if segments else OriginalFrames()


def iter_original_frame_segments_from_video(
    video_path: str | Path,
    segment_seconds: int | float | None = VIDEO_SEGMENT_SECONDS,
    extract_fps: int | float = VIDEO_EXTRACT_FPS,
) -> Iterator[OriginalFrames]:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = _calculate_frame_interval(fps, extract_fps)
    segment_source_frames = _calculate_segment_source_frames(fps, segment_seconds)
    logger.info(
        "读取视频成功: path=%s, fps=%.3f, total_frames=%d, "
        "frame_interval=%d, extract_fps=%.3f, segment_seconds=%s",
        video_path,
        fps,
        frame_count,
        frame_interval,
        extract_fps,
        segment_seconds,
    )

    source_frame_index = 0
    frame_id = 0
    segment_start_source_index = 0
    segment_index = 0
    segment_frames = OriginalFrames()

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
                segment_frames.add_frame(frame)
                frame_id += 1

            source_frame_index += 1

            if (
                segment_source_frames is not None
                and source_frame_index - segment_start_source_index >= segment_source_frames
            ):
                if segment_frames.frames:
                    logger.info(
                        "视频片段读取完成: path=%s, segment_index=%d, "
                        "source_frame_range=[%d,%d), extracted_frames=%d",
                        video_path,
                        segment_index,
                        segment_start_source_index,
                        source_frame_index,
                        len(segment_frames.frames),
                    )
                    yield segment_frames
                    segment_index += 1

                segment_frames = OriginalFrames()
                segment_start_source_index = source_frame_index
    finally:
        capture.release()

    if segment_frames.frames:
        logger.info(
            "视频片段读取完成: path=%s, segment_index=%d, "
            "source_frame_range=[%d,%d), extracted_frames=%d",
            video_path,
            segment_index,
            segment_start_source_index,
            source_frame_index,
            len(segment_frames.frames),
        )
        yield segment_frames

    logger.info(
        "视频帧提取成功: path=%s, extracted_frames=%d, source_frames_read=%d",
        video_path,
        frame_id,
        source_frame_index,
    )


def _calculate_segment_source_frames(
    fps: float,
    segment_seconds: int | float | None,
) -> int | None:
    if segment_seconds is None:
        return None

    if segment_seconds <= 0:
        raise ValueError(
            f"segment_seconds must be greater than 0, got {segment_seconds}"
        )

    if fps and fps > 0:
        return max(int(round(fps * segment_seconds)), 1)

    return None


def _calculate_frame_interval(fps: float, extract_fps: int | float) -> int:
    if extract_fps <= 0:
        raise ValueError(f"extract_fps must be greater than 0, got {extract_fps}")

    if fps and fps > 0:
        return max(int(round(fps / extract_fps)), 1)

    return 5
