from __future__ import annotations

from pathlib import Path

import torch

from src.rebuild import load_chunk, load_vggt_model, rebuild_chunk_to_npz


def run_rebuild(
    chunk_path: str | Path,
    model: torch.nn.Module | None = None,
    model_name_or_path: str = "facebook/VGGT-1B",
    device: str | torch.device | None = None,
) -> Path:
    """Load one Chunk file, run reconstruction, and save npz beside it."""
    chunk_path = Path(chunk_path)
    chunk = load_chunk(chunk_path)
    output_path = chunk_path.with_suffix(".npz")

    if model is None:
        model = load_vggt_model(
            model_name_or_path=model_name_or_path,
            device=device,
        )

    return rebuild_chunk_to_npz(
        chunk=chunk,
        output_path=output_path,
        model=model,
        model_name_or_path=model_name_or_path,
        device=device,
    )
