"""Geometry helpers."""

import numpy as np


def homogeneous(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    return np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
