"""Sim3 alignment utilities."""

import numpy as np

from src.optimization.umeyama import estimate_similarity


def align_sim3(source: np.ndarray, target: np.ndarray) -> dict:
    return estimate_similarity(source, target)
