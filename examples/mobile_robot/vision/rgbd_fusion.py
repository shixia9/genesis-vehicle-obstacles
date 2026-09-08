"""Explicit, inspectable RGB-D helper functions.

The current Genesis example has separate RGB and Depth sensors. This module
therefore labels normalized bbox sampling as an approximate alignment mode;
calibrated pixel alignment should replace it before safety-critical use.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import Detection, VisionResult


def sample_bbox_depth(
    depth: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    rgb_shape: tuple[int, int],
    max_range_m: float = 6.0,
) -> float | None:
    """Return a robust approximate depth using normalized bbox coordinates."""

    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim != 2:
        raise ValueError("depth must be a HxW array")
    rgb_height, rgb_width = rgb_shape
    depth_height, depth_width = depth_array.shape
    x1, y1, x2, y2 = bbox_xyxy
    sx = depth_width / max(1, rgb_width)
    sy = depth_height / max(1, rgb_height)
    left = max(0, min(depth_width - 1, int(round(x1 * sx))))
    right = max(left + 1, min(depth_width, int(round(x2 * sx))))
    top = max(0, min(depth_height - 1, int(round(y1 * sy))))
    bottom = max(top + 1, min(depth_height, int(round(y2 * sy))))
    roi = depth_array[top:bottom, left:right]
    valid = roi[np.isfinite(roi) & (roi > 0.05) & (roi <= max_range_m)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def enrich_approximate_depth(
    result: VisionResult,
    depth: np.ndarray,
    *,
    rgb_shape: tuple[int, int],
    max_range_m: float = 6.0,
) -> VisionResult:
    """Attach normalized-coordinate depth samples to missing detections."""

    detections: list[Detection] = []
    for detection in result.detections:
        if detection.distance_m is not None:
            detections.append(detection)
            continue
        distance = sample_bbox_depth(
            depth,
            detection.bbox_xyxy,
            rgb_shape=rgb_shape,
            max_range_m=max_range_m,
        )
        detections.append(replace(detection, distance_m=distance))
    return replace(result, detections=tuple(detections))
