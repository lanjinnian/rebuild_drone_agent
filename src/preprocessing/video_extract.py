"""Video frame extraction."""

from pathlib import Path
from typing import List


def extract_frames(video_path: str | Path, output_dir: str | Path, stride: int = 1) -> List[Path]:
    """Placeholder frame extraction entry point."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _ = (video_path, stride)
    return []
