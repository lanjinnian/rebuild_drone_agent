from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


ImageArray = NDArray[np.uint8]


def calculate_optical_flow_score(
    previous_image: ImageArray,
    current_image: ImageArray,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: int = 7,
    block_size: int = 7,
) -> float:
    """Calculate optical-flow movement score from two image arrays.

    Args:
        previous_image: Previous frame image data, such as BaseFrame.image.
        current_image: Current frame image data, such as BaseFrame.image.

    Returns:
        Average tracked feature-point movement distance in pixels. Higher
        scores indicate larger frame-to-frame motion.
    """
    previous_gray = _to_gray(previous_image)
    current_gray = _to_gray(current_image)
    _ensure_same_size(previous_gray, current_gray)

    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
    )
    if points is None or len(points) == 0:
        return 0.0

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
    )
    if next_points is None or status is None:
        return 0.0

    valid_mask = status.reshape(-1) == 1
    if not np.any(valid_mask):
        return 0.0

    valid_previous = points.reshape(-1, 2)[valid_mask]
    valid_next = next_points.reshape(-1, 2)[valid_mask]
    distances = np.linalg.norm(valid_next - valid_previous, axis=1)
    return float(np.mean(distances))


def calculate_overlap_score(
    first_image: ImageArray,
    second_image: ImageArray,
    max_features: int = 1000,
    lowe_ratio: float = 0.75,
) -> float:
    """Calculate image overlap score from two image arrays.

    Args:
        first_image: First frame image data, such as BaseFrame.image.
        second_image: Second frame image data, such as BaseFrame.image.

    Returns:
        Overlap score in the range [0.0, 1.0]. Higher scores indicate more
        reliable matched feature points between the two images.
    """
    first_gray = _to_gray(first_image)
    second_gray = _to_gray(second_image)

    detector = cv2.ORB_create(nfeatures=max_features)
    first_keypoints, first_descriptors = detector.detectAndCompute(first_gray, None)
    second_keypoints, second_descriptors = detector.detectAndCompute(second_gray, None)

    if (
        first_descriptors is None
        or second_descriptors is None
        or not first_keypoints
        or not second_keypoints
    ):
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(first_descriptors, second_descriptors, k=2)
    good_matches = []
    for match_group in raw_matches:
        if len(match_group) < 2:
            continue
        best_match, second_best_match = match_group
        if best_match.distance < lowe_ratio * second_best_match.distance:
            good_matches.append(best_match)

    denominator = min(len(first_keypoints), len(second_keypoints))
    if denominator == 0:
        return 0.0

    overlap = len(good_matches) / denominator
    return float(np.clip(overlap, 0.0, 1.0))


def calculate_sharpness_score(image: ImageArray) -> float:
    """Calculate image sharpness score from one image array.

    Args:
        image: Frame image data, such as BaseFrame.image.

    Returns:
        Laplacian variance score. Higher scores usually indicate a sharper
        image, while very low scores indicate blur or lack of texture.
    """
    gray = _to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


calculate_average_optical_flow_distance = calculate_optical_flow_score
calculate_feature_overlap = calculate_overlap_score
calculate_laplacian_variance = calculate_sharpness_score


def _to_gray(image: ImageArray) -> ImageArray:
    if image.ndim == 2:
        return image

    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    raise ValueError(f"Expected image with 1, 3, or 4 channels, got {image.shape[2]}")


def _ensure_same_size(first_image: ImageArray, second_image: ImageArray) -> None:
    if first_image.shape[:2] != second_image.shape[:2]:
        raise ValueError(
            "Optical flow requires images with the same height and width: "
            f"{first_image.shape[:2]} != {second_image.shape[:2]}"
        )
