"""Dynamic object filtering."""

from pathlib import Path
from typing import Iterable, List


def filter_dynamic_frames(image_paths: Iterable[str | Path]) -> List[Path]:
    return [Path(path) for path in image_paths]
