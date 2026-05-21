import numpy as np

from src.optimization.sim3_align import align_sim3


def test_align_sim3_returns_identity_placeholder():
    result = align_sim3(np.zeros((3, 3)), np.zeros((3, 3)))
    assert result["scale"] == 1.0
    assert result["rotation"].shape == (3, 3)
    assert result["translation"].shape == (3,)
