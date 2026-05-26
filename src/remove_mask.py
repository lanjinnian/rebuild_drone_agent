from __future__ import annotations

import logging
from pathlib import Path
import pickle
import sys
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np
from numpy.typing import NDArray
import torch

from config import (
    REMOVE_MASK_CLASS_IDS,
    REMOVE_MASK_DEVICE,
    REMOVE_MASK_MEAN,
    REMOVE_MASK_MODEL_PATH,
    REMOVE_MASK_MODEL_URL,
    REMOVE_MASK_SOURCE_PATH,
    REMOVE_MASK_SOURCE_URL,
    REMOVE_MASK_STD,
)
from src.datatype import Chunk


logger = logging.getLogger(__name__)


def load_sky_segmentation_model(
    model_path: str | Path = REMOVE_MASK_MODEL_PATH,
    device: str | torch.device | None = REMOVE_MASK_DEVICE,
) -> torch.nn.Module:
    """Load a runnable BiSeNetV2 sky segmentation model."""
    resolved_device = _resolve_device(device)
    model = _load_custom_segmentation_model(model_path, resolved_device)

    model = model.to(resolved_device)
    model.eval()
    logger.info(
        "BiSeNetV2天空分割模型加载完成: model_path=%s, device=%s",
        model_path,
        resolved_device,
    )
    return model


def load_bisenetv2_model(
    model_path: str | Path = REMOVE_MASK_MODEL_PATH,
    device: str | torch.device | None = REMOVE_MASK_DEVICE,
) -> torch.nn.Module:
    """Load a runnable BiSeNetV2 sky segmentation model."""
    return load_sky_segmentation_model(model_path=model_path, device=device)


def load_remove_mask_model(
    model_path: str | Path = REMOVE_MASK_MODEL_PATH,
    device: str | torch.device | None = REMOVE_MASK_DEVICE,
) -> torch.nn.Module:
    """Load a runnable BiSeNetV2 remove-mask segmentation model."""
    return load_sky_segmentation_model(model_path=model_path, device=device)


def _load_custom_segmentation_model(
    model_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    model_path = _ensure_local_file(model_path, REMOVE_MASK_MODEL_URL)
    try:
        model = torch.jit.load(str(model_path), map_location=device)
    except RuntimeError:
        loaded = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(loaded, torch.nn.Module):
            model = loaded
        elif isinstance(loaded, dict) and isinstance(loaded.get("model"), torch.nn.Module):
            model = loaded["model"]
        elif isinstance(loaded, dict):
            model = _load_bisenetv2_from_state_dict(loaded, device)
        else:
            raise TypeError(
                "sky segmentation model must be a TorchScript file, a torch-saved "
                "nn.Module, or a BiSeNetV2 state_dict"
            )
    return model.to(device)


def _ensure_local_file(path: str | Path, url: str) -> Path:
    path = Path(path)
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    if not url:
        raise FileNotFoundError(f"sky segmentation model file does not exist: {path}")

    logger.info("开始下载天空分割模型资源: url=%s, path=%s", url, path)
    urlretrieve(url, path)
    logger.info("天空分割模型资源下载完成: path=%s", path)
    return path


def _load_bisenetv2_from_state_dict(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    source_path = _ensure_local_file(REMOVE_MASK_SOURCE_PATH, REMOVE_MASK_SOURCE_URL)
    source_dir = str(source_path.parent.resolve())
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    from bisenetv2 import BiSeNetV2

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint

    state_dict = _strip_state_dict_prefix(state_dict)
    model = BiSeNetV2(n_classes=_infer_class_count(state_dict)).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        logger.warning(
            "BiSeNetV2权重加载存在不匹配: missing_keys=%d, unexpected_keys=%d",
            len(missing_keys),
            len(unexpected_keys),
        )
    return model


def _strip_state_dict_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.")
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        stripped[new_key] = value
    return stripped


def _infer_class_count(state_dict: dict[str, Any]) -> int:
    for key in (
        "conv_out.conv_out.2.weight",
        "module.conv_out.conv_out.2.weight",
        "head.conv_out.conv_out.2.weight",
    ):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            return int(weight.shape[0])
    return 19


def add_remove_masks_to_chunk_files(
    chunk_paths: list[str | Path],
    model: torch.nn.Module | None = None,
    model_path: str | Path = REMOVE_MASK_MODEL_PATH,
    device: str | torch.device | None = REMOVE_MASK_DEVICE,
    remove_class_ids: tuple[int, ...] = REMOVE_MASK_CLASS_IDS,
) -> list[Chunk]:
    """Load chunk files one by one, write remove masks, and save them back."""
    paths = [Path(chunk_path) for chunk_path in chunk_paths]
    if not paths:
        return []

    resolved_device = _resolve_device(device)
    model = model if model is not None else load_sky_segmentation_model(model_path, resolved_device)
    model = model.to(resolved_device)
    model.eval()

    processed_chunks = []
    for chunk_path in paths:
        chunk = load_chunk_pickle(chunk_path)
        apply_remove_masks_to_chunk(
            chunk,
            model=model,
            device=resolved_device,
            remove_class_ids=remove_class_ids,
        )
        save_chunk_pickle(chunk, chunk_path)
        processed_chunks.append(chunk)
        logger.info(
            "去除遮罩分块处理完成: chunk_path=%s, chunk_id=%s, frame_count=%d",
            chunk_path,
            chunk.id,
            len(chunk.frames),
        )

    return processed_chunks


def add_sky_masks_to_chunk_files(
    chunk_paths: list[str | Path],
    model: torch.nn.Module | None = None,
    model_path: str | Path = REMOVE_MASK_MODEL_PATH,
    device: str | torch.device | None = REMOVE_MASK_DEVICE,
    sky_class_id: int = 10,
) -> list[Chunk]:
    """Backward-compatible wrapper for previous sky-mask calls."""
    return add_remove_masks_to_chunk_files(
        chunk_paths,
        model=model,
        model_path=model_path,
        device=device,
        remove_class_ids=(sky_class_id,),
    )


def apply_remove_masks_to_chunk(
    chunk: Chunk,
    model: torch.nn.Module,
    device: str | torch.device | None = REMOVE_MASK_DEVICE,
    remove_class_ids: tuple[int, ...] = REMOVE_MASK_CLASS_IDS,
) -> Chunk:
    """Run BiSeNetV2 on every frame and store the remove mask in frame.mask."""
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    for frame in chunk.frames:
        frame.mask = predict_remove_mask(
            frame.image,
            model=model,
            device=resolved_device,
            remove_class_ids=remove_class_ids,
        )

    return chunk


def apply_sky_masks_to_chunk(
    chunk: Chunk,
    model: torch.nn.Module,
    device: str | torch.device | None = REMOVE_MASK_DEVICE,
    sky_class_id: int = 10,
) -> Chunk:
    """Backward-compatible wrapper for previous sky-mask calls."""
    return apply_remove_masks_to_chunk(
        chunk,
        model=model,
        device=device,
        remove_class_ids=(sky_class_id,),
    )


def predict_remove_mask(
    image: NDArray[np.uint8],
    model: torch.nn.Module,
    device: str | torch.device | None = REMOVE_MASK_DEVICE,
    remove_class_ids: tuple[int, ...] = REMOVE_MASK_CLASS_IDS,
) -> NDArray[np.uint8]:
    """Return a uint8 remove mask with the same shape as image."""
    image = _ensure_rgb_image(image)
    resolved_device = _resolve_device(device)
    padded_image, original_shape = _pad_image_to_multiple(image, multiple=32)
    input_tensor = _image_to_input_tensor(padded_image).to(resolved_device)

    with torch.no_grad():
        output = model(input_tensor)

    logits = _extract_logits(output)
    logits = torch.nn.functional.interpolate(
        logits,
        size=padded_image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    classes = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()
    classes = classes[: original_shape[0], : original_shape[1]]
    remove_mask = np.isin(classes, remove_class_ids).astype(np.uint8) * np.uint8(255)
    return np.repeat(remove_mask[:, :, None], image.shape[2], axis=2)


def predict_sky_mask(
    image: NDArray[np.uint8],
    model: torch.nn.Module,
    device: str | torch.device | None = REMOVE_MASK_DEVICE,
    sky_class_id: int = 10,
) -> NDArray[np.uint8]:
    """Backward-compatible wrapper for previous sky-mask calls."""
    return predict_remove_mask(
        image,
        model=model,
        device=device,
        remove_class_ids=(sky_class_id,),
    )


def load_chunk_pickle(chunk_path: str | Path) -> Chunk:
    chunk_path = Path(chunk_path)
    with chunk_path.open("rb") as file:
        chunk = pickle.load(file)

    if not isinstance(chunk, Chunk):
        raise TypeError(f"{chunk_path} does not contain a Chunk object")
    return chunk


def save_chunk_pickle(chunk: Chunk, chunk_path: str | Path) -> Path:
    chunk_path = Path(chunk_path)
    with chunk_path.open("wb") as file:
        pickle.dump(chunk, file)
    return chunk_path


def _image_to_input_tensor(image: NDArray[np.uint8]) -> torch.Tensor:
    image_float = image.astype(np.float32) / 255.0
    mean = np.asarray(REMOVE_MASK_MEAN, dtype=np.float32)
    std = np.asarray(REMOVE_MASK_STD, dtype=np.float32)
    normalized = (image_float - mean) / std
    return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()


def _pad_image_to_multiple(
    image: NDArray[np.uint8],
    multiple: int,
) -> tuple[NDArray[np.uint8], tuple[int, int]]:
    height, width = image.shape[:2]
    padded_height = ((height + multiple - 1) // multiple) * multiple
    padded_width = ((width + multiple - 1) // multiple) * multiple
    if padded_height == height and padded_width == width:
        return image, (height, width)

    padded_image = np.zeros(
        (padded_height, padded_width, image.shape[2]),
        dtype=image.dtype,
    )
    padded_image[:height, :width] = image
    return padded_image, (height, width)


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, dict):
        for key in ("out", "logits", "pred", "preds"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                logits = value
                break
        else:
            raise TypeError("segmentation output dict does not contain tensor logits")
    elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        logits = output[0]
    else:
        raise TypeError(f"unsupported segmentation output type: {type(output)!r}")

    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    if logits.ndim != 4:
        raise ValueError(f"segmentation logits must have shape [N,C,H,W], got {tuple(logits.shape)}")
    return logits


def _ensure_rgb_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3:
        raise ValueError(f"frame image must be 2D or 3D, got shape {image.shape}")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.shape[2] != 3:
        raise ValueError(f"frame image must have 1, 3, or 4 channels, got {image.shape[2]}")
    return image


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
