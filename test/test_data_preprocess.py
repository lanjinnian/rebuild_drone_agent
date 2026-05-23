import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data_preprocess import run_data_preprocess

video_path = "data/example/01.mp4"

run_data_preprocess(video_path)
