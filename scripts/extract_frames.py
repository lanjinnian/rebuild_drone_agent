"""Extract frames from a video."""

from src.preprocessing.video_extract import extract_frames


if __name__ == "__main__":
    extract_frames("data/raw/input.mp4", "data/processed")
