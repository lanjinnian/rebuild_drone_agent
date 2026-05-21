"""Command-line pipeline runner."""

from pathlib import Path
from typing import Any, Dict

import yaml

from src.pipeline.stages import build_default_stages, run_stages
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def run_pipeline(config_path: str | Path = "configs/default.yaml") -> Dict[str, dict]:
    config = load_config(config_path)
    stage_names = config.get("pipeline", {}).get("stages", [])
    stages = build_default_stages(stage_names)
    LOGGER.info("Running pipeline with %d stages", len(stages))
    return run_stages(stages, config)
