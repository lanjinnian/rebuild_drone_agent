import numpy as np

from src.utils.geometry import homogeneous


def test_homogeneous_appends_one_column():
    points = np.array([[1.0, 2.0, 3.0]])
    result = homogeneous(points)
    assert result.tolist() == [[1.0, 2.0, 3.0, 1.0]]
