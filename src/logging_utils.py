from __future__ import annotations

import logging
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure project logging once when no application logging exists."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )


def configure_task_file_logging(
    task_dir: str | Path,
    filename: str = "pipeline.log",
    level: int = logging.INFO,
) -> Path:
    """Add a file log handler under the task directory."""
    configure_logging(level)

    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_dir / filename
    resolved_log_path = log_path.resolve()

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler_path = Path(handler.baseFilename).resolve()
            if handler_path == resolved_log_path:
                return log_path

    for handler in root_logger.handlers[:]:
        if getattr(handler, "_rebuild_drone_task_log", False):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler._rebuild_drone_task_log = True
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    )
    root_logger.addHandler(file_handler)
    return log_path
