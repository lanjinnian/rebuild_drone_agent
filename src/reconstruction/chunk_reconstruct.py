"""Chunk-level reconstruction."""

from pathlib import Path
from typing import Iterable, List

from src.reconstruction.vggt_runner import VGGTRunner


def reconstruct_chunks(chunks: Iterable[Iterable[str | Path]], output_dir: str | Path) -> List[dict]:
    runner = VGGTRunner()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return [runner.reconstruct(chunk, Path(output_dir) / f"chunk_{idx:04d}") for idx, chunk in enumerate(chunks)]
