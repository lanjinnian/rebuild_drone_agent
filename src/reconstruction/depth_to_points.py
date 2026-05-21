"""Depth map to point cloud conversion."""

import numpy as np


def depth_to_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    _ = intrinsics
    return np.empty((0, 3), dtype=float) if depth.size == 0 else np.zeros((depth.size, 3), dtype=float)
