"""Satellite image matching."""

from pathlib import Path


def match_satellite(image_path: str | Path, satellite_path: str | Path) -> dict:
    return {"image": str(image_path), "satellite": str(satellite_path), "matches": []}
