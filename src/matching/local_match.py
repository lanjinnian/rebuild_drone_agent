"""Local image matching."""

from pathlib import Path


def match_pair(image_a: str | Path, image_b: str | Path, method: str = "orb") -> dict:
    return {"image_a": str(image_a), "image_b": str(image_b), "method": method, "matches": []}
