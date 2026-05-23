import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import run_rebuild

chunk_path = "result/20260523_144647/0.pkl"
run_rebuild(chunk_path)
chunk_path = "result/20260523_144647/1.pkl"
run_rebuild(chunk_path)
chunk_path = "result/20260523_144647/2.pkl"
run_rebuild(chunk_path)
chunk_path = "result/20260523_144647/3.pkl"
run_rebuild(chunk_path)
chunk_path = "result/20260523_144647/4.pkl"
run_rebuild(chunk_path)