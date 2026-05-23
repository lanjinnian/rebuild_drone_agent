from __future__ import annotations

from pathlib import Path

from src.align import align
from src.rebuild import npz_load, rebuild_npz_to_glb


def run_align_and_merge(
    storage_dir: str | Path,
    chunk_count: int,
    output_npz_name: str = "aligned.npz",
    output_glb_name: str = "aligned.glb",
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = False,
    mask_sky: bool = False,
    prediction_mode: str = "Predicted Pointmap",
) -> Path:
    """Align numbered chunk npz files in one folder and save a merged GLB there."""
    if chunk_count <= 0:
        raise ValueError(f"chunk_count must be positive, got {chunk_count}")

    storage_dir = Path(storage_dir)
    chunk_paths = [storage_dir / f"{index}.npz" for index in range(chunk_count)]
    missing_paths = [path for path in chunk_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"missing chunk npz files: {missing_paths}")

    chunk_to_processes = [npz_load(path) for path in chunk_paths]
    merged_npz_path = storage_dir / output_npz_name
    merged_glb_path = storage_dir / output_glb_name

    align(chunk_to_processes, merged_npz_path)
    rebuild_npz_to_glb(
        npz_path=merged_npz_path,
        output_path=merged_glb_path,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        prediction_mode=prediction_mode,
    )
    return merged_glb_path
