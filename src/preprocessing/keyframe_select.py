"""Keyframe selection."""

from pathlib import Path
from typing import Iterable, List


def select_keyframes(frame_paths: Iterable[str | Path], output_dir: str | Path) -> List[Path]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return [Path(path) for path in frame_paths]
