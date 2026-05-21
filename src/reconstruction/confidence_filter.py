"""Confidence-based filtering."""

import numpy as np


def filter_by_confidence(points: np.ndarray, confidence: np.ndarray, threshold: float) -> np.ndarray:
    if points.size == 0:
        return points
    return points[confidence >= threshold]
