"""Run VGGT reconstruction for one chunk."""

from src.reconstruction.vggt_runner import run_vggt


if __name__ == "__main__":
    run_vggt([], "data/outputs/chunk_0000")
