from __future__ import annotations

from pathlib import Path
import pickle
import sys
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
import torch

from src.datatype import Chunk


VGGT_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference" / "vggt"
VGGT_TARGET_SIZE = 518


def load_vggt_model(
    model_name_or_path: str = "facebook/VGGT-1B",
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Load a VGGT model using the local reference implementation."""
    _ensure_vggt_reference_importable()

    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained(model_name_or_path)
    model = model.to(_resolve_device(device))
    model.eval()
    return model


def rebuild_chunk_to_npz(
    chunk: Chunk,
    output_path: str | Path,
    model: torch.nn.Module | None = None,
    model_name_or_path: str = "facebook/VGGT-1B",
    device: str | torch.device | None = None,
) -> Path:
    """Run VGGT reconstruction for a Chunk and save predictions to an npz file."""
    model = model if model is not None else load_vggt_model(model_name_or_path, device)
    predictions = rebuild_chunk(chunk, model, device=device)
    return save_rebuild_predictions(predictions, output_path)


def load_chunk(chunk_path: str | Path) -> Chunk:
    """Load a Chunk object from a pickle file."""
    chunk_path = Path(chunk_path)
    with chunk_path.open("rb") as file:
        chunk = pickle.load(file)

    if not isinstance(chunk, Chunk):
        raise TypeError(f"{chunk_path} does not contain a Chunk object")

    return chunk


def rebuild_chunk(
    chunk: Chunk,
    model: torch.nn.Module,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run VGGT reconstruction for a Chunk and return a VGGT-style prediction dict."""
    if not chunk.frames:
        raise ValueError(f"chunk {chunk.id} has no frames")

    _ensure_vggt_reference_importable()
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    images = chunk_to_vggt_images(chunk).to(resolved_device)
    dtype = _select_autocast_dtype(resolved_device)

    with torch.no_grad():
        with _autocast_context(resolved_device, dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        images.shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    np_predictions = _tensor_predictions_to_numpy(predictions)
    np_predictions["pose_enc_list"] = None

    depth_map = np_predictions["depth"]
    np_predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map,
        np_predictions["extrinsic"],
        np_predictions["intrinsic"],
    )
    np_predictions["chunk_id"] = np.array(chunk.id)
    np_predictions["frame_ids"] = np.array(chunk.frame_ids, dtype=np.int64)
    np_predictions["gps_locations"] = _chunk_gps_locations(chunk)

    return np_predictions


def save_rebuild_predictions(
    predictions: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save VGGT-style reconstruction predictions to npz."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **predictions)
    return output_path


def chunk_to_vggt_images(chunk: Chunk, target_size: int = VGGT_TARGET_SIZE) -> torch.Tensor:
    """Convert Chunk frames to a VGGT input tensor with shape [S, 3, H, W]."""
    if not chunk.frames:
        raise ValueError(f"chunk {chunk.id} has no frames")

    tensors = []
    for frame in chunk.frames:
        image = _prepare_frame_image(frame.image, target_size)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)

    return torch.stack(tensors)


def _ensure_vggt_reference_importable() -> None:
    if VGGT_REFERENCE_DIR.exists():
        vggt_reference_path = str(VGGT_REFERENCE_DIR)
        if vggt_reference_path not in sys.path:
            sys.path.insert(0, vggt_reference_path)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=False)


def _tensor_predictions_to_numpy(predictions: dict[str, Any]) -> dict[str, Any]:
    np_predictions: dict[str, Any] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            np_predictions[key] = value.detach().cpu().numpy().squeeze(0)
        else:
            np_predictions[key] = value
    return np_predictions


def _prepare_frame_image(image: NDArray[np.uint8], target_size: int) -> NDArray[np.uint8]:
    image = _ensure_rgb_image(image)
    image = _resize_for_vggt(image, target_size)
    return np.ascontiguousarray(image)


def _ensure_rgb_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)

    if image.shape[2] != 3:
        raise ValueError(f"frame image must have 1, 3, or 4 channels, got {image.shape[2]}")

    return image


def _resize_for_vggt(image: NDArray[np.uint8], target_size: int) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"frame image has invalid shape {image.shape}")

    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14
    new_height = max(new_height, 14)

    image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )

    if new_height > target_size:
        start_y = (new_height - target_size) // 2
        image = image[start_y : start_y + target_size, :, :]

    return _pad_to_shape(image, target_size, target_size)


def _pad_to_shape(
    image: NDArray[np.uint8],
    target_height: int,
    target_width: int,
) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    height_padding = target_height - height
    width_padding = target_width - width

    if height_padding <= 0 and width_padding <= 0:
        return image

    pad_top = height_padding // 2
    pad_bottom = height_padding - pad_top
    pad_left = width_padding // 2
    pad_right = width_padding - pad_left

    return cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _chunk_gps_locations(chunk: Chunk) -> NDArray[np.float32]:
    locations = []
    for frame in chunk.frames:
        if frame.gps_location is None:
            locations.append((np.nan, np.nan, np.nan))
        else:
            locations.append(frame.gps_location)
    return np.array(locations, dtype=np.float32)
