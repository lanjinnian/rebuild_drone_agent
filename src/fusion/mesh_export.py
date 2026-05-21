"""Mesh export helpers."""

from pathlib import Path


def export_mesh(mesh: object, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# mesh export placeholder\n", encoding="utf-8")
    _ = mesh
    return path
