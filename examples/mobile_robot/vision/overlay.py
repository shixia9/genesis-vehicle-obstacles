"""Simple RGB detection visualization for offline and runtime results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .types import Detection, VisionResult, normalize_rgb_image


_PALETTE = (
    (255, 80, 80),
    (80, 180, 255),
    (80, 220, 120),
    (255, 190, 60),
    (210, 110, 255),
    (255, 120, 190),
)


def _label_for_detection(detection: Detection) -> str:
    label = f"{detection.label} {detection.confidence:.2f}"
    if detection.track_id:
        label += f" #{detection.track_id}"
    if detection.color:
        label += f" {detection.color}"
    if detection.shape:
        label += f" {detection.shape}"
    if detection.distance_m is not None:
        label += f" {detection.distance_m:.2f}m"
    return label


def annotate_rgb(image: np.ndarray, result: VisionResult) -> np.ndarray:
    """Draw detection boxes and labels and return an RGB uint8 image."""

    rgb = normalize_rgb_image(image)
    canvas = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for detection in result.detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        box = (
            max(0, min(width - 1, int(round(x1)))),
            max(0, min(height - 1, int(round(y1)))),
            max(0, min(width - 1, int(round(x2)))),
            max(0, min(height - 1, int(round(y2)))),
        )
        color = _PALETTE[detection.class_id % len(_PALETTE)]
        draw.rectangle(box, outline=color, width=3)
        label = _label_for_detection(detection)
        text_bbox = draw.textbbox((0, 0), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = min(max(0, box[0]), max(0, width - text_width))
        text_y = box[1] - text_height - 2
        if text_y < 0:
            text_y = min(height - text_height, box[3] + 2)
        text_box = (text_x, text_y, text_x + text_width, text_y + text_height)
        draw.rectangle(text_box, fill=color)
        draw.text((text_x, text_y), label, fill=(0, 0, 0))
    return np.asarray(canvas, dtype=np.uint8)


def iter_image_paths(directory: str | Path) -> list[Path]:
    """Yield supported images in deterministic filename order."""

    root = Path(directory)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [path for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in suffixes]
