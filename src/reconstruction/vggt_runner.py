"""VGGT/VGGT-Long runner wrapper."""

from pathlib import Path
from typing import Iterable, List


class VGGTRunner:
    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        self.checkpoint = checkpoint
        self.device = device

    def reconstruct(self, image_paths: Iterable[str | Path], output_dir: str | Path) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"images": [str(path) for path in image_paths], "output_dir": str(output_dir)}


def run_vggt(image_paths: Iterable[str | Path], output_dir: str | Path) -> dict:
    return VGGTRunner().reconstruct(image_paths, output_dir)
