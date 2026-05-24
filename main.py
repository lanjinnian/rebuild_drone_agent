from __future__ import annotations

import argparse
from pathlib import Path

from config import RESULT_DIR
from pipeline import run_align_and_merge, run_data_preprocess, run_rebuild
from src.logging_utils import configure_logging


def run_pipeline(video_path: str | Path, result_dir: str | Path = RESULT_DIR) -> Path:
    """Run preprocess, rebuild, and align workflows for one input video."""
    video_path = Path(video_path)
    result_dir = Path(result_dir)

    chunks = run_data_preprocess(video_path, result_dir=result_dir)
    if not chunks:
        raise ValueError(f"no chunks generated from video: {video_path}")

    task_dir = _find_task_dir(result_dir, [chunk.id for chunk in chunks])
    for chunk in chunks:
        run_rebuild(task_dir / f"{chunk.id}.pkl")

    return run_align_and_merge(
        storage_dir=task_dir,
        chunk_count=len(chunks),
    )


def _find_task_dir(result_dir: Path, chunk_ids: list[int]) -> Path:
    """Find the latest result directory containing the generated chunk files."""
    if not result_dir.exists():
        raise FileNotFoundError(f"result directory does not exist: {result_dir}")

    candidates = []
    for path in result_dir.iterdir():
        if not path.is_dir():
            continue

        if all((path / f"{chunk_id}.pkl").exists() for chunk_id in chunk_ids):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"could not find task directory under {result_dir} for chunks: {chunk_ids}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run video-to-GLB reconstruction pipeline.",
    )
    parser.add_argument("video_path", help="Input video path.")
    parser.add_argument(
        "--result-dir",
        default=RESULT_DIR,
        help=f"Directory for intermediate and output files. Default: {RESULT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    glb_path = run_pipeline(args.video_path, result_dir=args.result_dir)
    print(glb_path)


if __name__ == "__main__":
    main()
