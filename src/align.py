from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config import (
    ALIGN_CONF_THRESHOLD,
    ALIGN_IRLS_DELTA,
    ALIGN_IRLS_MAX_ITERS,
    ALIGN_MIN_POINTS,
)
from src.datatype import ChunkToProcess, FrameInChunk
from src.rebuild import npz_load


logger = logging.getLogger(__name__)


Sim3 = tuple[float, NDArray[np.floating], NDArray[np.floating]]


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


def estimate_sim3(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> Sim3:
    """Estimate weighted Sim3 that maps source points to target points."""
    _validate_point_pairs(source, target, weights)

    normalized_weights = weights.astype(np.float64, copy=False)
    normalized_weights = normalized_weights / (normalized_weights.sum() + 1e-12)
    source = source.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)

    mu_src = np.sum(normalized_weights[:, None] * source, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target, axis=0)

    src_centered = source - mu_src
    tgt_centered = target - mu_tgt

    scale_src = np.sqrt(
        np.sum(normalized_weights * np.sum(src_centered**2, axis=1))
    )
    scale_tgt = np.sqrt(
        np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1))
    )
    if scale_src <= 1e-12:
        raise ValueError("source points are degenerate; cannot estimate Sim3")

    scale = float(scale_tgt / scale_src)
    weighted_src = scale * src_centered * np.sqrt(normalized_weights[:, None])
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights[:, None])

    h_matrix = weighted_src.T @ weighted_tgt
    u_matrix, _, vt_matrix = np.linalg.svd(h_matrix)
    rotation = vt_matrix.T @ u_matrix.T

    if np.linalg.det(rotation) < 0:
        vt_matrix[2, :] *= -1
        rotation = vt_matrix.T @ u_matrix.T

    translation = mu_tgt - scale * rotation @ mu_src
    return scale, rotation, translation


def robust_estimate_sim3(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> Sim3:
    """Estimate Sim3 with Huber IRLS for outlier robustness."""
    scale, rotation, translation = estimate_sim3(source, target, weights)

    for _ in range(ALIGN_IRLS_MAX_ITERS):
        transformed = apply_sim3_to_points(source, scale, rotation, translation)
        residuals = np.linalg.norm(target - transformed, axis=1)

        huber = np.ones_like(residuals)
        large = residuals > ALIGN_IRLS_DELTA
        huber[large] = ALIGN_IRLS_DELTA / residuals[large]

        scale, rotation, translation = estimate_sim3(
            source,
            target,
            weights * huber,
        )

    return scale, rotation, translation


def apply_sim3_to_points(
    points: NDArray[np.floating],
    scale: float,
    rotation: NDArray[np.floating],
    translation: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply a Sim3 transform to point arrays with last dimension 3."""
    return scale * np.einsum("ij,...j->...i", rotation, points) + translation


def estimate_chunk_pair_sim3(
    chunk_a: ChunkToProcess,
    chunk_b: ChunkToProcess,
) -> Sim3:
    """Estimate chunk_b -> chunk_a Sim3 from overlapping frames."""
    target, source, weights = extract_overlap_points(chunk_a, chunk_b)
    if len(weights) < ALIGN_MIN_POINTS:
        raise ValueError(
            f"not enough valid overlap points: got {len(weights)}, "
            f"need at least {ALIGN_MIN_POINTS}"
        )

    return robust_estimate_sim3(source, target, weights)


def accumulate_sim3_transforms(transforms: list[Sim3]) -> list[Sim3]:
    """Accumulate adjacent transforms so every transform maps a chunk to chunk0."""
    cumulative_transforms: list[Sim3] = []
    for scale_next, rotation_next, translation_next in transforms:
        if not cumulative_transforms:
            cumulative_transforms.append(
                (scale_next, rotation_next, translation_next)
            )
            continue

        scale_prev, rotation_prev, translation_prev = cumulative_transforms[-1]
        scale_new = scale_prev * scale_next
        rotation_new = rotation_prev @ rotation_next
        translation_new = (
            scale_prev * (rotation_prev @ translation_next) + translation_prev
        )
        cumulative_transforms.append((scale_new, rotation_new, translation_new))

    return cumulative_transforms


def align_chunk_to_processes(
    chunk_to_processes: list[ChunkToProcess],
) -> list[ChunkToProcess]:
    """Align all ChunkToProcess objects into the first chunk coordinate system."""
    if not chunk_to_processes:
        return []
    if len(chunk_to_processes) == 1:
        return [chunk_to_processes[0]]

    adjacent_transforms = []
    for index in range(len(chunk_to_processes) - 1):
        chunk_a = chunk_to_processes[index]
        chunk_b = chunk_to_processes[index + 1]
        transform = estimate_chunk_pair_sim3(chunk_a, chunk_b)
        adjacent_transforms.append(transform)
        logger.info(
            "每组分块拼接完成: source_chunk_id=%s, target_chunk_id=%s, transform_matrix=%s",
            chunk_b.chunk_id,
            chunk_a.chunk_id,
            np.array2string(_sim3_to_matrix(transform), precision=6, suppress_small=False),
        )
    cumulative_transforms = accumulate_sim3_transforms(adjacent_transforms)

    aligned_chunks = [chunk_to_processes[0]]
    for chunk, (scale, rotation, translation) in zip(
        chunk_to_processes[1:],
        cumulative_transforms,
    ):
        aligned_chunk = ChunkToProcess(chunk_id=chunk.chunk_id)
        for frame in chunk.frames:
            aligned_world_points = apply_sim3_to_points(
                frame.world_points,
                scale,
                rotation,
                translation,
            ).astype(frame.world_points.dtype, copy=False)
            aligned_frame = FrameInChunk(
                id=frame.id,
                image=frame.image,
                gps_location=frame.gps_location,
                chunk_id=frame.chunk_id,
                world_points=aligned_world_points,
                world_points_conf=frame.world_points_conf,
            )
            aligned_chunk.add_frame(aligned_frame)
        aligned_chunks.append(aligned_chunk)

    return aligned_chunks


def save_aligned_chunks_npz(
    chunk_to_processes: list[ChunkToProcess],
    output_path: str | Path,
) -> Path:
    """Align, concatenate, and save ChunkToProcess objects into one npz file."""
    aligned_chunks = align_chunk_to_processes(chunk_to_processes)
    predictions = _chunks_to_npz_predictions(aligned_chunks)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **predictions)
    return output_path


def align_npz_files(
    npz_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    """Load chunk npz files, align them, and save one concatenated npz file."""
    chunks = [npz_load(npz_path) for npz_path in npz_paths]
    return save_aligned_chunks_npz(chunks, output_path)


def align(
    chunk_to_processes: list[ChunkToProcess],
    output_path: str | Path,
) -> Path:
    """Align a ChunkToProcess array and save one concatenated npz file."""
    return save_aligned_chunks_npz(chunk_to_processes, output_path)


def align_chunks(chunks: list[ChunkToProcess]) -> list[ChunkToProcess]:
    """Backward-compatible alias for align_chunk_to_processes."""
    return align_chunk_to_processes(chunks)


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


def _validate_point_pairs(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> None:
    if source.shape != target.shape:
        raise ValueError(
            f"source shape {source.shape} does not match target shape {target.shape}"
        )
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source and target must have shape (N, 3), got {source.shape}"
        )
    if weights.shape != (source.shape[0],):
        raise ValueError(
            f"weights must have shape ({source.shape[0]},), got {weights.shape}"
        )
    if source.shape[0] < ALIGN_MIN_POINTS:
        raise ValueError(
            f"not enough point pairs: got {source.shape[0]}, "
            f"need at least {ALIGN_MIN_POINTS}"
        )
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target must contain only finite points")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if weights.sum() <= 1e-12:
        raise ValueError("weights sum is zero; cannot estimate Sim3")


def _chunks_to_npz_predictions(chunks: list[ChunkToProcess]) -> dict[str, Any]:
    frames = []
    seen_frame_ids: set[int] = set()
    for chunk in chunks:
        for frame in chunk.frames:
            if frame.id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame.id)
            frames.append(frame)

    if not frames:
        raise ValueError("no frames to save")

    world_points = np.stack([frame.world_points for frame in frames])
    depth_conf = np.stack([frame.world_points_conf for frame in frames])
    images = np.stack([frame.image for frame in frames])
    frame_ids = np.array([frame.id for frame in frames], dtype=np.int64)
    frame_count = len(frames)
    gps_locations = np.array(
        [
            frame.gps_location
            if frame.gps_location is not None
            else (np.nan, np.nan, np.nan)
            for frame in frames
        ],
        dtype=np.float64,
    )

    return {
        "world_points_from_depth": world_points,
        "depth_conf": depth_conf,
        "images": images,
        "pose_enc": np.empty((frame_count, 0), dtype=np.float32),
        "depth": np.zeros(depth_conf.shape, dtype=np.float32),
        "extrinsic": _identity_extrinsics(frame_count),
        "intrinsic": _identity_intrinsics(frame_count),
        "chunk_id": np.array(chunks[0].chunk_id if chunks else -1),
        "frame_ids": frame_ids,
        "gps_locations": gps_locations,
        "world_points": world_points,
        "world_points_conf": depth_conf,
    }


def _identity_extrinsics(frame_count: int) -> NDArray[np.float32]:
    extrinsic = np.zeros((frame_count, 3, 4), dtype=np.float32)
    extrinsic[:, :3, :3] = np.eye(3, dtype=np.float32)
    return extrinsic


def _identity_intrinsics(frame_count: int) -> NDArray[np.float32]:
    intrinsic = np.zeros((frame_count, 3, 3), dtype=np.float32)
    intrinsic[:, :, :] = np.eye(3, dtype=np.float32)
    return intrinsic


def _sim3_to_matrix(transform: Sim3) -> NDArray[np.float64]:
    scale, rotation, translation = transform
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix
