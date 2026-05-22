from __future__ import annotations

from pathlib import Path

import torch

from src.rebuild import load_chunk, load_vggt_model, rebuild_chunk_to_npz, rebuild_npz_to_glb


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


def run_rebuild_to_glb(
    npz_path: str | Path,
    output_path: str | Path | None = None,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    prediction_mode: str = "Predicted Pointmap",
) -> Path:
    """Convert one reconstruction npz file to GLB beside the npz by default."""
    return rebuild_npz_to_glb(
        npz_path=npz_path,
        output_path=output_path,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        prediction_mode=prediction_mode,
    )
