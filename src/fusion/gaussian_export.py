"""3D Gaussian Splatting export helpers."""

from pathlib import Path


def export_gaussians(gaussians: object, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# gaussian export placeholder\n", encoding="utf-8")
    _ = gaussians
    return path
