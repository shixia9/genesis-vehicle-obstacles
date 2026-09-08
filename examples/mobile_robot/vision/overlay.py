"""Simple RGB detection visualization for offline and runtime results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class AnnotatedRgbView:
    """Display runtime RGB frames with perception boxes in an OpenCV window.

    Genesis owns the texture used by its ``Camera(GUI=True)`` window.  That
    texture is intentionally left untouched here: this view is a separate,
    explicit display sink for the image returned by :func:`annotate_rgb`.
    OpenCV is imported lazily so importing the vision package remains safe in
    headless jobs and in tests that only use the data contracts.
    """

    def __init__(self, window_name: str = "YOLO annotated robot camera") -> None:
        if not window_name:
            raise ValueError("window_name must not be empty")
        self.window_name = window_name
        self._cv2: Any | None = None
        self._opened = False
        self._closed = False

    def _load_cv2(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "--annotated-view requires the opencv-python package; "
                    "install the project's vision dependencies first"
                ) from exc
            self._cv2 = cv2
        return self._cv2

    def show(
        self,
        image: np.ndarray,
        result: VisionResult | None = None,
    ) -> bool:
        """Show one RGB frame and return whether the window remains active.

        ``result`` may be ``None`` for image-only capture ticks (for example,
        when ``--image-every`` is slower than ``--vision-every``).  In that
        case the current frame is shown without reusing a stale bounding box.
        Pressing ``q`` or ``Esc`` closes the window and disables subsequent
        updates without stopping the vehicle controller.
        """

        if self._closed:
            return False

        cv2 = self._load_cv2()
        rgb = annotate_rgb(image, result) if result is not None else normalize_rgb_image(image)
        try:
            if not self._opened:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._opened = True
            bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, bgr)
            key = int(cv2.waitKey(1)) & 0xFF
            if key in (27, ord("q")):
                self.close()
                return False
            try:
                visible = float(cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE))
            except (AttributeError, cv2.error):
                # Some OpenCV GUI backends do not implement window-property
                # queries.  The window is still usable in that case.
                visible = 1.0
            if visible < 0.5:
                self.close()
                return False
        except cv2.error as exc:  # pragma: no cover - backend/display dependent
            self.close()
            raise RuntimeError(
                "unable to display --annotated-view; check that a GUI display "
                "is available (DISPLAY/WindowServer)"
            ) from exc
        return True

    def close(self) -> None:
        """Close the OpenCV window, if it was opened."""

        if self._closed:
            return
        self._closed = True
        if self._cv2 is not None and self._opened:
            try:
                self._cv2.destroyWindow(self.window_name)
                self._cv2.waitKey(1)
            except self._cv2.error:  # pragma: no cover - backend/display dependent
                pass
        self._opened = False


def iter_image_paths(directory: str | Path) -> list[Path]:
    """Yield supported images in deterministic filename order."""

    root = Path(directory)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [path for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in suffixes]
