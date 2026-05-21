"""Point cloud fusion."""

import numpy as np


def fuse_pointclouds(pointclouds: list[np.ndarray]) -> np.ndarray:
    if not pointclouds:
        return np.empty((0, 3), dtype=float)
    return np.concatenate(pointclouds, axis=0)
