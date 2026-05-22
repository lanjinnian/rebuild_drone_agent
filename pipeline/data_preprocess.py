from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pickle

from config import RESULT_DIR
from src.datatype import Chunk
from src.frame_chunk import split_original_frames_into_chunks
from src.image_preprocess import preprocess_original_frames
from src.key_frame_select import select_key_frames
from src.video_load import load_original_frames_from_video


def run_data_preprocess(
    video_path: str | Path,
    result_dir: str | Path = RESULT_DIR,
) -> list[Chunk]:
    """Run video loading, preprocessing, key-frame selection, chunking, and saving."""
    task_id = _create_task_id()

    original_frames = load_original_frames_from_video(video_path)
    preprocessed_frames = preprocess_original_frames(original_frames)
    selected_frames = select_key_frames(preprocessed_frames)
    chunks = split_original_frames_into_chunks(selected_frames)

    save_chunks(chunks, task_id, result_dir)
    return chunks


def save_chunks(
    chunks: list[Chunk],
    task_id: str,
    result_dir: str | Path = RESULT_DIR,
) -> Path:
    """Save Chunk objects under result/<task_id>/<chunk_id>.pkl."""
    task_dir = Path(result_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        chunk_path = task_dir / f"{chunk.id}.pkl"
        with chunk_path.open("wb") as file:
            pickle.dump(chunk, file)

    return task_dir


def _create_task_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
