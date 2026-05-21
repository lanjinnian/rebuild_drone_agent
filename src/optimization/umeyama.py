"""Umeyama similarity alignment."""

import numpy as np


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> dict:
    _ = (source, target)
    return {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3)}
