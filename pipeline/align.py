from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path

import torch

from config import ALIGN_GLB_CONF_THRES, ALIGN_LOOP_MODEL_NAME_OR_PATH
from src.align import align
from src.logging_utils import configure_logging, configure_task_file_logging
from src.rebuild import npz_load, rebuild_npz_to_glb
from src.task_summary import ResourceStepMonitor, StepSummary


logger = logging.getLogger(__name__)
ALIGN_GLB_MIN_CONF_PERCENTILE = 50.0


def run_align_and_merge(
    storage_dir: str | Path,
    chunk_count: int,
    output_npz_name: str = "aligned.npz",
    output_glb_name: str = "aligned.glb",
    conf_thres: float = ALIGN_GLB_CONF_THRES,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = False,
    mask_sky: bool = False,
    prediction_mode: str = "Predicted Pointmap",
    loop_pairs: list[tuple[int, int]] | None = None,
    enable_loop: bool = True,
    loop_model: torch.nn.Module | None = None,
    loop_model_name_or_path: str = ALIGN_LOOP_MODEL_NAME_OR_PATH,
    device: str | torch.device | None = None,
    summary_steps: list[StepSummary] | None = None,
) -> Path:
    """Align numbered chunk npz files in one folder and save a merged GLB there."""
    configure_logging()
    if chunk_count <= 0:
        raise ValueError(f"chunk_count must be positive, got {chunk_count}")

    storage_dir = Path(storage_dir)
    configure_task_file_logging(storage_dir)
    chunk_paths = [storage_dir / f"{index}.npz" for index in range(chunk_count)]
    logger.info(
        "开始拼接分块: storage_dir=%s, chunk_count=%d, chunk_paths=%s",
        storage_dir,
        chunk_count,
        chunk_paths,
    )
    missing_paths = [path for path in chunk_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"missing chunk npz files: {missing_paths}")

    logger.info("开始读取重建结果: chunk_count=%d", chunk_count)
    with _summary_step(summary_steps, "align_read_npz"):
        chunk_to_processes = [npz_load(path) for path in chunk_paths]
    merged_npz_path = storage_dir / output_npz_name
    merged_glb_path = storage_dir / output_glb_name

    logger.info("开始计算并保存拼接结果: output_path=%s", merged_npz_path)
    with _summary_step(summary_steps, "align_compute_and_save"):
        align(
            chunk_to_processes,
            merged_npz_path,
            loop_pairs=loop_pairs,
            enable_loop=enable_loop,
            loop_model=loop_model,
            loop_model_name_or_path=loop_model_name_or_path,
            device=str(device) if device is not None else None,
        )
    logger.info("拼接结果保存完成: output_path=%s", merged_npz_path)
    logger.info("开始导出最终 GLB: npz_path=%s, output_path=%s", merged_npz_path, merged_glb_path)
    with _summary_step(summary_steps, "align_export_glb"):
        rebuild_npz_to_glb(
            npz_path=merged_npz_path,
            output_path=merged_glb_path,
            conf_thres=max(conf_thres, ALIGN_GLB_MIN_CONF_PERCENTILE),
            filter_by_frames=filter_by_frames,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            prediction_mode=prediction_mode,
        )
    logger.info("最终输出模型: %s", merged_glb_path)
    return merged_glb_path


@contextmanager
def _summary_step(summary_steps: list[StepSummary] | None, name: str):
    if summary_steps is None:
        yield
        return

    with ResourceStepMonitor(name) as monitor:
        yield
    summary_steps.append(monitor.summary())
