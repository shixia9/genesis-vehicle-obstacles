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
        text_box = draw.textbbox((box[0], box[1]), label)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), label, fill=(0, 0, 0))
    return np.asarray(canvas, dtype=np.uint8)


def iter_image_paths(directory: str | Path) -> list[Path]:
    """Yield supported images in deterministic filename order."""

    root = Path(directory)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [path for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in suffixes]
