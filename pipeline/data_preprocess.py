from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import pickle
from contextlib import contextmanager

from config import (
    IMAGE_SAMPLE_INTERVAL,
    IMAGE_SEGMENT_SIZE,
    READ_TYPE,
    READ_TYPE_CHOICES,
    READ_TYPE_VIDEO,
    RESULT_DIR,
    REMOVE_MASK_CLASS_IDS,
    REMOVE_MASK_ENABLE,
    REMOVE_MASK_MODEL_PATH,
)
from src.datatype import Chunk
from src.frame_chunk import split_original_frames_into_chunks
from src.image_load import iter_original_frame_segments_from_image
from src.image_preprocess import preprocess_original_frames
from src.key_frame_select import select_key_frames
from src.logging_utils import configure_logging, configure_task_file_logging
from src.remove_mask import add_remove_masks_to_chunk_files, load_remove_mask_model
from src.task_summary import ResourceStepMonitor, StepSummary
from src.video_load import iter_original_frame_segments_from_video


logger = logging.getLogger(__name__)


def run_data_preprocess(
    input_path: str | Path,
    result_dir: str | Path = RESULT_DIR,
    read_type: str = READ_TYPE,
    image_segment_size: int = IMAGE_SEGMENT_SIZE,
    image_sample_interval: int = IMAGE_SAMPLE_INTERVAL,
    remove_mask_enable: bool = REMOVE_MASK_ENABLE,
    remove_mask_model_path: str | Path | None = None,
    remove_mask_device: str | None = None,
    remove_mask_class_ids: tuple[int, ...] | None = None,
    summary_steps: list[StepSummary] | None = None,
) -> list[Chunk]:
    """Run video loading, preprocessing, key-frame selection, chunking, and saving."""
    chunks, _ = run_data_preprocess_with_task_dir(
        input_path,
        result_dir,
        read_type=read_type,
        image_segment_size=image_segment_size,
        image_sample_interval=image_sample_interval,
        remove_mask_enable=remove_mask_enable,
        remove_mask_model_path=remove_mask_model_path,
        remove_mask_device=remove_mask_device,
        remove_mask_class_ids=remove_mask_class_ids,
        summary_steps=summary_steps,
    )
    return chunks


def run_data_preprocess_with_task_dir(
    input_path: str | Path,
    result_dir: str | Path = RESULT_DIR,
    read_type: str = READ_TYPE,
    image_segment_size: int = IMAGE_SEGMENT_SIZE,
    image_sample_interval: int = IMAGE_SAMPLE_INTERVAL,
    remove_mask_enable: bool = REMOVE_MASK_ENABLE,
    remove_mask_model_path: str | Path | None = None,
    remove_mask_device: str | None = None,
    remove_mask_class_ids: tuple[int, ...] | None = None,
    summary_steps: list[StepSummary] | None = None,
) -> tuple[list[Chunk], Path]:
    """Run preprocessing and return the generated chunks with their task directory."""
    configure_logging()
    task_id = _create_task_id()
    task_dir = Path(result_dir) / task_id
    log_path = configure_task_file_logging(task_dir)
    logger.info("任务日志文件: %s", log_path)
    frame_segments = iter(_iter_original_frame_segments(
        input_path,
        read_type,
        image_segment_size=image_segment_size,
        image_sample_interval=image_sample_interval,
    ))

    chunks: list[Chunk] = []
    remove_mask_model = None
    effective_remove_mask_model_path = (
        remove_mask_model_path if remove_mask_model_path is not None else REMOVE_MASK_MODEL_PATH
    )
    effective_remove_mask_class_ids = (
        remove_mask_class_ids if remove_mask_class_ids is not None else REMOVE_MASK_CLASS_IDS
    )
    segment_index = 0
    while True:
        read_step_name = (
            "read_video_segment"
            if read_type == READ_TYPE_VIDEO
            else "read_image_segment"
        )
        logger.info(
            "开始读取输入片段: segment_index=%d, read_type=%s",
            segment_index,
            read_type,
        )
        with _summary_step(summary_steps, f"{read_step_name}_{segment_index}"):
            try:
                original_frames = next(frame_segments)
            except StopIteration:
                logger.info("输入片段读取结束: next_segment_index=%d", segment_index)
                break

        original_frame_ids = _frame_id_range(original_frames.frames)
        logger.info(
            "输入片段读取完成: segment_index=%d, frame_count=%d, frame_id_range=%s",
            segment_index,
            len(original_frames.frames),
            original_frame_ids,
        )
        logger.info(
            "开始预处理输入片段: segment_index=%d, frame_count=%d",
            segment_index,
            len(original_frames.frames),
        )
        with _summary_step(summary_steps, f"image_preprocess_segment_{segment_index}"):
            preprocessed_frames = preprocess_original_frames(original_frames)
        logger.info(
            "输入片段预处理完成: segment_index=%d, frame_count=%d, frame_id_range=%s",
            segment_index,
            len(preprocessed_frames.frames),
            _frame_id_range(preprocessed_frames.frames),
        )

        logger.info(
            "开始关键帧选取: segment_index=%d, frame_count=%d",
            segment_index,
            len(preprocessed_frames.frames),
        )
        with _summary_step(summary_steps, f"key_frame_select_segment_{segment_index}"):
            selected_frames = select_key_frames(preprocessed_frames)
        logger.info(
            "关键帧选取完成: segment_index=%d, input_frames=%d, selected_frames=%d, "
            "frame_id_range=%s",
            segment_index,
            len(preprocessed_frames.frames),
            len(selected_frames.frames),
            _frame_id_range(selected_frames.frames),
        )

        logger.info(
            "开始分块: segment_index=%d, selected_frames=%d",
            segment_index,
            len(selected_frames.frames),
        )
        with _summary_step(summary_steps, f"chunk_split_segment_{segment_index}"):
            segment_chunks = split_original_frames_into_chunks(selected_frames)
        logger.info(
            "分块完成: segment_index=%d, segment_chunk_count=%d, frame_counts=%s",
            segment_index,
            len(segment_chunks),
            [len(chunk.frames) for chunk in segment_chunks],
        )

        chunk_start_id = len(chunks)
        chunks_to_save: list[Chunk] = []
        for chunk in segment_chunks:
            chunks_to_save.append(
                Chunk(id=chunk_start_id + len(chunks_to_save), frames=chunk.frames)
            )
        logger.info(
            "开始保存分块: segment_index=%d, chunk_ids=%s",
            segment_index,
            [chunk.id for chunk in chunks_to_save],
        )
        with _summary_step(summary_steps, f"chunk_save_segment_{segment_index}"):
            task_dir = save_chunks(chunks_to_save, task_id, result_dir)
        logger.info(
            "保存分块完成: segment_index=%d, task_dir=%s, chunk_ids=%s",
            segment_index,
            task_dir,
            [chunk.id for chunk in chunks_to_save],
        )

        if remove_mask_enable:
            if remove_mask_model is None:
                logger.info(
                    "开始加载去除遮罩模型: segment_index=%d, model_path=%s, device=%s",
                    segment_index,
                    effective_remove_mask_model_path,
                    remove_mask_device,
                )
                with _summary_step(summary_steps, "remove_mask_model_load"):
                    remove_mask_model = load_remove_mask_model(
                        effective_remove_mask_model_path,
                        remove_mask_device,
                    )
                logger.info("去除遮罩模型加载完成: segment_index=%d", segment_index)
            chunk_paths = [task_dir / f"{chunk.id}.pkl" for chunk in chunks_to_save]
            logger.info(
                "开始处理遮罩: segment_index=%d, chunk_paths=%s",
                segment_index,
                chunk_paths,
            )
            with _summary_step(summary_steps, f"remove_mask_segment_{segment_index}"):
                masked_chunks = add_remove_masks_to_chunk_files(
                    chunk_paths,
                    model=remove_mask_model,
                    device=remove_mask_device,
                    remove_class_ids=effective_remove_mask_class_ids,
                )
            chunks_to_save = masked_chunks
            logger.info(
                "处理遮罩完成: segment_index=%d, chunk_ids=%s",
                segment_index,
                [chunk.id for chunk in chunks_to_save],
            )
        else:
            logger.info("跳过去除遮罩: segment_index=%d", segment_index)

        chunks.extend(chunks_to_save)
        logger.info(
            "输入片段处理完成: segment_index=%d, segment_chunk_count=%d, total_chunk_count=%d",
            segment_index,
            len(segment_chunks),
            len(chunks),
        )
        segment_index += 1

    logger.info(
        "全部输入片段分块完成: task_id=%s, task_dir=%s, chunk_count=%d",
        task_id,
        task_dir,
        len(chunks),
    )
    return chunks, task_dir


def _iter_original_frame_segments(
    input_path: str | Path,
    read_type: str,
    image_segment_size: int = IMAGE_SEGMENT_SIZE,
    image_sample_interval: int = IMAGE_SAMPLE_INTERVAL,
):
    if read_type not in READ_TYPE_CHOICES:
        raise ValueError(
            f"read_type must be one of {READ_TYPE_CHOICES}, got {read_type!r}"
        )

    if read_type == READ_TYPE_VIDEO:
        return iter_original_frame_segments_from_video(input_path)

    return iter_original_frame_segments_from_image(
        input_path,
        segment_size=image_segment_size,
        sample_interval=image_sample_interval,
    )


@contextmanager
def _summary_step(summary_steps: list[StepSummary] | None, name: str):
    if summary_steps is None:
        yield
        return

    with ResourceStepMonitor(name) as monitor:
        yield
    summary_steps.append(monitor.summary())


def _frame_id_range(frames) -> str:
    if not frames:
        return "empty"

    return f"{frames[0].id}-{frames[-1].id}"


def save_chunks(
    chunks: list[Chunk],
    task_id: str,
    result_dir: str | Path = RESULT_DIR,
) -> Path:
    """Save Chunk objects under result/<task_id>/<chunk_id>.pkl."""
    task_dir = Path(result_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    if not chunks:
        return task_dir

    for chunk in chunks:
        chunk_path = task_dir / f"{chunk.id}.pkl"
        with chunk_path.open("wb") as file:
            pickle.dump(chunk, file)

    logger.info(
        "分块完成并保存: task_id=%s, task_dir=%s, chunk_count=%d, chunk_ids=%s",
        task_id,
        task_dir,
        len(chunks),
        [chunk.id for chunk in chunks],
    )
    return task_dir


def _create_task_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
