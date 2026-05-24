from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from config import KEY_FRAME_DELETE_RATIO
from src.datatype import BaseFrame, OriginalFrames
from utils.calcula import (
    calculate_optical_flow_score,
    calculate_overlap_score,
    calculate_sharpness_score,
)


logger = logging.getLogger(__name__)


@dataclass
class FrameScore:
    frame: BaseFrame
    optical_flow_score: float = 0.0
    overlap_score: float = 0.0
    sharpness_score: float = 0.0
    total_score: float = 0.0


def select_key_frames(
    frames: OriginalFrames,
    delete_ratio: float = KEY_FRAME_DELETE_RATIO,
) -> OriginalFrames:
    """Select key frames by normalized optical-flow, overlap, and sharpness scores."""
    if not 0.0 <= delete_ratio < 1.0:
        raise ValueError(f"delete_ratio must be in [0.0, 1.0), got {delete_ratio}")

    if not frames.frames:
        return OriginalFrames()

    frame_scores = _calculate_frame_scores(frames.frames)
    _normalize_and_sum_scores(frame_scores)

    delete_count = int(len(frame_scores) * delete_ratio)
    deleted_frame_ids = {
        frame_score.frame.id
        for frame_score in sorted(frame_scores, key=lambda item: item.total_score)[
            :delete_count
        ]
    }

    selected_frames = OriginalFrames()
    for frame in frames.frames:
        if frame.id not in deleted_frame_ids:
            selected_frames.add_frame(frame)

    logger.info(
        "关键帧计算完成: total=%d, selected=%d, deleted=%d, delete_ratio=%.3f, "
        "optical_flow=%s, overlap=%s, sharpness=%s, total_score=%s",
        len(frame_scores),
        len(selected_frames.frames),
        delete_count,
        delete_ratio,
        _score_stats([frame_score.optical_flow_score for frame_score in frame_scores]),
        _score_stats([frame_score.overlap_score for frame_score in frame_scores]),
        _score_stats([frame_score.sharpness_score for frame_score in frame_scores]),
        _score_stats([frame_score.total_score for frame_score in frame_scores]),
    )
    return selected_frames


def _calculate_frame_scores(frames: list[BaseFrame]) -> list[FrameScore]:
    frame_scores = [FrameScore(frame=frame) for frame in frames]

    for index, frame_score in enumerate(frame_scores):
        frame_score.sharpness_score = calculate_sharpness_score(frame_score.frame.image)

        if index == 0:
            continue

        previous_image = frame_scores[index - 1].frame.image
        current_image = frame_score.frame.image
        frame_score.optical_flow_score = calculate_optical_flow_score(
            previous_image,
            current_image,
        )
        frame_score.overlap_score = calculate_overlap_score(previous_image, current_image)

    return frame_scores


def _normalize_and_sum_scores(frame_scores: list[FrameScore]) -> None:
    optical_flow_scores = _normalize_scores(
        [frame_score.optical_flow_score for frame_score in frame_scores]
    )
    overlap_scores = _normalize_scores(
        [frame_score.overlap_score for frame_score in frame_scores]
    )
    sharpness_scores = _normalize_scores(
        [frame_score.sharpness_score for frame_score in frame_scores]
    )

    for index, frame_score in enumerate(frame_scores):
        frame_score.optical_flow_score = optical_flow_scores[index]
        frame_score.overlap_score = overlap_scores[index]
        frame_score.sharpness_score = sharpness_scores[index]
        frame_score.total_score = (
            frame_score.optical_flow_score
            + frame_score.overlap_score
            + frame_score.sharpness_score
        )


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    values = np.asarray(scores, dtype=np.float64)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return [0.0 for _ in scores]

    normalized = (values - min_value) / (max_value - min_value)
    return [float(score) for score in normalized]


def _score_stats(scores: list[float]) -> str:
    if not scores:
        return "count=0"

    values = np.asarray(scores, dtype=np.float64)
    return (
        f"count={values.size}, min={values.min():.6f}, max={values.max():.6f}, "
        f"mean={values.mean():.6f}, std={values.std():.6f}"
    )
