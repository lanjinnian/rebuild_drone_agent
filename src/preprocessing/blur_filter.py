"""Blur filtering."""

from pathlib import Path
from typing import Iterable, List


def filter_blurry_images(image_paths: Iterable[str | Path], threshold: float = 80.0) -> List[Path]:
    _ = threshold
    return [Path(path) for path in image_paths]
