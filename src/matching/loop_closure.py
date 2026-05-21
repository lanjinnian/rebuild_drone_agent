"""Loop closure detection."""

from pathlib import Path
from typing import Iterable, List


def detect_loop_closures(image_paths: Iterable[str | Path]) -> List[tuple[int, int]]:
    _ = list(image_paths)
    return []
