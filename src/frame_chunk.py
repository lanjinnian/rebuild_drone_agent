from config import CHUNK_OVERLAP_SIZE, CHUNK_SIZE
from src.datatype import Chunk, OriginalFrames


def split_original_frames_into_chunks(
    original_frames: OriginalFrames,
    chunk_size: int = CHUNK_SIZE,
    overlap_size: int = CHUNK_OVERLAP_SIZE,
) -> list[Chunk]:
    """Split OriginalFrames into time-ordered chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")

    if not 0 <= overlap_size < chunk_size:
        raise ValueError(
            "overlap_size must be greater than or equal to 0 and smaller than "
            f"chunk_size, got overlap_size={overlap_size}, chunk_size={chunk_size}"
        )

    if not original_frames.frames:
        return []

    chunks: list[Chunk] = []
    step_size = chunk_size - overlap_size
    start_index = 0
    chunk_id = 0

    while start_index < len(original_frames.frames):
        chunk_frames = original_frames.frames[start_index : start_index + chunk_size]
        chunks.append(Chunk(id=chunk_id, frames=chunk_frames))

        if start_index + chunk_size >= len(original_frames.frames):
            break

        start_index += step_size
        chunk_id += 1

    return chunks
