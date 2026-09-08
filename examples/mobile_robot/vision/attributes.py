"""Lightweight RGB ROI attributes used after object detection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import Detection, FramePacket, VisionResult


def dominant_color(image: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> tuple[str | None, float]:
    """Estimate a coarse color from an inner RGB bounding-box region.

    This is deliberately conservative and returns ``None`` for low saturation
    or mixed-color regions. It is an attribute helper, not a replacement for a
    calibrated color classifier.
    """

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    roi = rgb[top:bottom, left:right].astype(np.float32)
    if roi.size == 0:
        return None, 0.0

    red, green, blue = roi[..., 0], roi[..., 1], roi[..., 2]
    brightness = np.max(roi, axis=-1)
    saturation = brightness - np.min(roi, axis=-1)
    valid = (brightness > 45.0) & (saturation > 20.0)
    if not np.any(valid):
        return None, 0.0

    # Genesis materials can appear considerably darker than their nominal
    # colors under the attached camera lighting (the yellow car is commonly
    # around RGB=(80, 70, 13)).  Use chroma ratios for yellow instead of fixed
    # channel floors, while keeping a brightness/saturation gate above.
    masks = {
        "yellow": (
            valid
            & (red > blue * 1.8)
            & (green > blue * 1.8)
            & (red >= green * 0.65)
            & (green >= red * 0.55)
        ),
        "red": valid & (red > green * 1.35) & (red > blue * 1.35),
        "green": valid & (green > red * 1.25) & (green > blue * 1.15),
        "blue": valid & (blue > red * 1.25) & (blue > green * 1.10),
    }
    counts = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    color, count = max(counts.items(), key=lambda item: item[1])
    confidence = count / max(1, int(np.count_nonzero(valid)))
    if confidence < 0.35:
        return None, float(confidence)
    return color, float(min(1.0, confidence))


def enrich_colors(frame: FramePacket, result: VisionResult) -> VisionResult:
    """Fill missing detection colors from the RGB ROI without changing boxes."""

    detections: list[Detection] = []
    for detection in result.detections:
        if detection.color is not None:
            detections.append(detection)
            continue
        color, confidence = dominant_color(frame.image, detection.bbox_xyxy)
        detections.append(replace(detection, color=color, color_confidence=confidence or None))
    return replace(result, detections=tuple(detections))
