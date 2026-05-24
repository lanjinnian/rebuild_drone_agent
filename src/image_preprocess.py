import logging

import cv2

from src.datatype import BaseFrame, OriginalFrames


logger = logging.getLogger(__name__)


def preprocess_original_frames(frames: OriginalFrames) -> OriginalFrames:
    preprocessed_frames = OriginalFrames()
    for frame in frames.frames:
        image = _ensure_rgb_image(frame.image)
        image = _equalize_illumination(image)
        image = _resize_for_vggt(image)
        image = _pad_to_shape(image, 518, 518)
        preprocessed_frames.add_frame(
            BaseFrame(
                id=frame.id,
                image=image,
                gps_location=frame.gps_location,
            )
        )

    logger.info("所有视频帧处理完成: count=%d", len(preprocessed_frames.frames))
    return preprocessed_frames


def _ensure_rgb_image(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype("float32") / 255.0
        rgb = image[:, :, :3].astype("float32")
        white = 255.0
        return (rgb * alpha + white * (1.0 - alpha)).astype("uint8")

    return image


def _equalize_illumination(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    lab_image = cv2.merge((lightness, channel_a, channel_b))
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


def _resize_for_vggt(image, target_size: int = 518):
    height, width = image.shape[:2]
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

    return image


def _pad_to_shape(image, target_height: int, target_width: int):
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
