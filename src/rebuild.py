from __future__ import annotations

from pathlib import Path
import pickle
import sys
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import trimesh

from src.datatype import Chunk, ChunkToProcess, FrameInChunk


VGGT_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference" / "vggt"
VGGT_TARGET_SIZE = 518
VGGT_GLB_KEYS = [
    "pose_enc",
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
    "images",
    "extrinsic",
    "intrinsic",
    "world_points_from_depth",
]
NPZ_LOAD_KEYS = [
    "world_points_from_depth",
    "depth_conf",
    "images",
    "chunk_id",
    "frame_ids",
    "gps_locations",
]


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
    ).astype(np.float32)
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


def npz_load(npz_path: str | Path) -> ChunkToProcess:
    """Load reconstruction npz data into a ChunkToProcess object."""
    npz_path = Path(npz_path)
    with np.load(npz_path) as loaded:
        missing_keys = [key for key in NPZ_LOAD_KEYS if key not in loaded.files]
        if missing_keys:
            raise KeyError(f"{npz_path} is missing ChunkToProcess keys: {missing_keys}")

        world_points_from_depth = np.array(loaded["world_points_from_depth"])
        depth_conf = np.array(loaded["depth_conf"])
        images = np.array(loaded["images"])
        chunk_id = int(np.array(loaded["chunk_id"]).item())
        frame_ids = np.array(loaded["frame_ids"])
        gps_locations = np.array(loaded["gps_locations"])

    frame_count = len(frame_ids)
    _validate_npz_load_frame_count(
        npz_path,
        frame_count,
        world_points_from_depth=world_points_from_depth,
        depth_conf=depth_conf,
        images=images,
        gps_locations=gps_locations,
    )

    chunk_to_process = ChunkToProcess(chunk_id=chunk_id, frames=[], frame_ids=[])
    for frame_index, frame_id in enumerate(frame_ids):
        frame = FrameInChunk(
            id=int(frame_id),
            image=images[frame_index],
            gps_location=tuple(gps_locations[frame_index]),
            chunk_id=chunk_id,
            world_points=world_points_from_depth[frame_index],
            world_points_conf=depth_conf[frame_index],
        )
        chunk_to_process.add_frame(frame)

    return chunk_to_process


def rebuild_npz_to_glb(
    npz_path: str | Path,
    output_path: str | Path | None = None,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    prediction_mode: str = "Predicted Pointmap",
) -> Path:
    """Convert a VGGT-style npz prediction file to GLB."""
    npz_path = Path(npz_path)
    if output_path is None:
        output_path = npz_path.with_suffix(".glb")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = load_rebuild_predictions_for_glb(npz_path)
    glb_scene = rebuild_predictions_to_glb_scene(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        prediction_mode=prediction_mode,
    )
    glb_scene.export(file_obj=str(output_path))
    return output_path


def load_rebuild_predictions_for_glb(npz_path: str | Path) -> dict[str, NDArray[Any]]:
    """Load the subset of VGGT predictions required for GLB conversion."""
    npz_path = Path(npz_path)
    with np.load(npz_path) as loaded:
        missing_keys = [key for key in VGGT_GLB_KEYS if key not in loaded.files]
        if missing_keys:
            raise KeyError(f"{npz_path} is missing GLB prediction keys: {missing_keys}")
        return {key: np.array(loaded[key]) for key in VGGT_GLB_KEYS}


def _validate_npz_load_frame_count(
    npz_path: Path,
    frame_count: int,
    **frame_arrays: NDArray[Any],
) -> None:
    for key, value in frame_arrays.items():
        if value.shape[0] != frame_count:
            raise ValueError(
                f"{npz_path} has {frame_count} frame_ids but {key} has first dimension {value.shape[0]}"
            )


def rebuild_predictions_to_glb_scene(
    predictions: dict[str, NDArray[Any]],
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    prediction_mode: str = "Predicted Pointmap",
) -> trimesh.Scene:
    """Build a previewable GLB scene from VGGT-style predictions."""
    if mask_sky:
        raise NotImplementedError("mask_sky requires the reference VGGT sky segmentation assets")

    world_points, confidence = _select_glb_points_and_confidence(
        predictions,
        prediction_mode,
    )
    images = _images_to_nhwc(predictions["images"])

    world_points, confidence, images = _select_glb_frame(
        world_points,
        confidence,
        images,
        filter_by_frames,
    )

    vertices = world_points.reshape(-1, 3)
    colors = (np.clip(images.reshape(-1, 3), 0.0, 1.0) * 255.0).astype(np.uint8)
    confidence = confidence.reshape(-1)

    mask = _build_glb_point_mask(
        confidence,
        colors,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
    )
    vertices = vertices[mask]
    colors = colors[mask]

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))

    if show_cam:
        camera_meshes = _build_camera_markers(predictions["extrinsic"], vertices)
        for camera_mesh in camera_meshes:
            scene.add_geometry(camera_mesh)

    return scene


def _select_glb_points_and_confidence(
    predictions: dict[str, NDArray[Any]],
    prediction_mode: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if "Pointmap" in prediction_mode:
        points = predictions["world_points"]
        confidence = predictions["world_points_conf"]
    else:
        points = predictions["world_points_from_depth"]
        confidence = predictions["depth_conf"]
    return points, confidence


def _images_to_nhwc(images: NDArray[np.floating]) -> NDArray[np.floating]:
    if images.ndim != 4:
        raise ValueError(f"images must have shape [S, C, H, W] or [S, H, W, C], got {images.shape}")
    if images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    if images.shape[-1] == 3:
        return images
    raise ValueError(f"images must have 3 color channels, got {images.shape}")


def _select_glb_frame(
    points: NDArray[np.floating],
    confidence: NDArray[np.floating],
    images: NDArray[np.floating],
    filter_by_frames: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    if filter_by_frames in ("all", "All"):
        return points, confidence, images

    frame_index = int(filter_by_frames.split(":")[0])
    return points[frame_index][None], confidence[frame_index][None], images[frame_index][None]


def _build_glb_point_mask(
    confidence: NDArray[np.floating],
    colors: NDArray[np.uint8],
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
) -> NDArray[np.bool_]:
    if conf_thres == 0.0:
        threshold = 0.0
    else:
        threshold = np.percentile(confidence, conf_thres)

    mask = (confidence >= threshold) & (confidence > 1e-5)

    if mask_black_bg:
        mask = mask & (colors.sum(axis=1) >= 16)

    if mask_white_bg:
        white_mask = (colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240)
        mask = mask & ~white_mask

    return mask


def _build_camera_markers(
    extrinsic: NDArray[np.floating],
    points: NDArray[np.floating],
) -> list[trimesh.Trimesh]:
    if points.size == 0:
        scene_scale = 1.0
    else:
        lower = np.percentile(points, 5, axis=0)
        upper = np.percentile(points, 95, axis=0)
        scene_scale = float(np.linalg.norm(upper - lower))
        if scene_scale <= 0:
            scene_scale = 1.0

    marker_radius = scene_scale * 0.01
    markers = []
    for camera_extrinsic in extrinsic:
        camera_to_world = np.linalg.inv(_to_homogeneous_extrinsic(camera_extrinsic))
        marker = trimesh.creation.icosphere(subdivisions=1, radius=marker_radius)
        marker.apply_translation(camera_to_world[:3, 3])
        marker.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
        markers.append(marker)
    return markers


def _to_homogeneous_extrinsic(extrinsic: NDArray[np.floating]) -> NDArray[np.float64]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = extrinsic
    return matrix


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
        return torch.amp.autocast("cuda", dtype=dtype)
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
