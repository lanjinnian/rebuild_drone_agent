"""Align reconstructed chunks."""

import numpy as np

from src.optimization.sim3_align import align_sim3


if __name__ == "__main__":
    align_sim3(np.empty((0, 3)), np.empty((0, 3)))
