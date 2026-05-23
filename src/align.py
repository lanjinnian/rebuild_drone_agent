from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


from config import ALIGN_CONF_THRESHOLD
from src.datatype import ChunkToProcess


def extract_overlap_points(
    chunk_a: ChunkToProcess,
    chunk_b: ChunkToProcess,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Extract matched valid overlap points from two chunks."""
    frame_index_a = _frame_id_to_index(chunk_a)
    frame_index_b = _frame_id_to_index(chunk_b)

    overlap_frame_ids = [
        frame_id for frame_id in chunk_a.frame_ids if frame_id in frame_index_b
    ]
    if not overlap_frame_ids:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    points_a = np.stack(
        [
            chunk_a.frames[frame_index_a[frame_id]].world_points
            for frame_id in overlap_frame_ids
        ]
    )
    conf_a = np.stack(
        [
            chunk_a.frames[frame_index_a[frame_id]].world_points_conf
            for frame_id in overlap_frame_ids
        ]
    )
    points_b = np.stack(
        [
            chunk_b.frames[frame_index_b[frame_id]].world_points
            for frame_id in overlap_frame_ids
        ]
    )
    conf_b = np.stack(
        [
            chunk_b.frames[frame_index_b[frame_id]].world_points_conf
            for frame_id in overlap_frame_ids
        ]
    )

    conf_a = _normalize_conf_shape(conf_a, points_a)
    conf_b = _normalize_conf_shape(conf_b, points_b)

    valid = (
        np.isfinite(points_a).all(axis=-1)
        & np.isfinite(points_b).all(axis=-1)
        & (conf_a > ALIGN_CONF_THRESHOLD)
        & (conf_b > ALIGN_CONF_THRESHOLD)
    )

    target = points_a[valid]
    source = points_b[valid]
    weights = np.sqrt(conf_a[valid] * conf_b[valid])
    return target, source, weights


def align(*args, **kwargs):
    """Placeholder for alignment logic."""
    raise NotImplementedError("Alignment logic has not been specified yet")


def _frame_id_to_index(chunk: ChunkToProcess) -> dict[int, int]:
    if len(chunk.frame_ids) != len(chunk.frames):
        raise ValueError(
            f"chunk {chunk.chunk_id} frame_ids length does not match frames length"
        )

    frame_index: dict[int, int] = {}
    for index, frame_id in enumerate(chunk.frame_ids):
        if frame_id in frame_index:
            raise ValueError(f"chunk {chunk.chunk_id} has duplicate frame_id {frame_id}")
        frame_index[frame_id] = index

    return frame_index


def _normalize_conf_shape(
    conf: NDArray[np.floating],
    points: NDArray[np.floating],
) -> NDArray[np.floating]:
    if conf.shape == points.shape[:-1]:
        return conf
    if conf.shape == points.shape[:-1] + (1,):
        return np.squeeze(conf, axis=-1)

    raise ValueError(
        f"world_points_conf shape {conf.shape} does not match world_points shape "
        f"{points.shape}"
    )
