"""Ultralytics YOLO adapter for the mobile-robot vision contract.

This adapter is deliberately local-only. ``model_path`` must point to an
existing weight file; a model name such as ``yolo11n.pt`` is rejected instead
of allowing Ultralytics to download it at runtime.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from .types import Detection, FramePacket, VisionResult


class YoloDetector:
    """Run a locally stored Ultralytics YOLO model on RGB frame packets."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        confidence: float = 0.25,
        image_size: int = 640,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"YOLO model file does not exist: {path}. "
                "Provide a local .pt/.onnx weight file; runtime downloads are disabled."
            )
        if path.suffix.lower() not in {".pt", ".onnx", ".engine"}:
            raise ValueError("model_path must be a local .pt, .onnx or .engine weight file")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if int(image_size) <= 0:
            raise ValueError("image_size must be greater than zero")

        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - depends on environment setup.
            raise RuntimeError(
                "Ultralytics is not installed in the active environment. "
                "Install the project's pinned visual dependencies first."
            ) from error

        self.model_path = path.resolve()
        self.device = str(device)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        # Passing an existing local path is the only model-loading operation.
        self.model = YOLO(str(self.model_path))

    def detect(self, frame: FramePacket) -> VisionResult:
        """Detect objects in one RGB packet and preserve its timestamp."""

        started = time.perf_counter()
        # Ultralytics' numpy input path expects OpenCV-style BGR. The public
        # packet remains RGB so all camera and controller code uses one format.
        image_bgr = np.ascontiguousarray(frame.image[:, :, ::-1])
        predictions = self.model.predict(
            source=image_bgr,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        result = predictions[0]
        detections = tuple(self._convert_boxes(result))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return VisionResult(
            frame_id=frame.frame_id,
            sim_time=frame.sim_time,
            model_name=self.model_path.name,
            latency_ms=latency_ms,
            detections=detections,
        )

    @staticmethod
    def _convert_boxes(result: Any) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        names = result.names
        detections: list[Detection] = []
        for bbox, confidence, class_id in zip(coordinates, confidences, class_ids):
            class_id_int = int(class_id)
            if isinstance(names, dict):
                label = str(names.get(class_id_int, class_id_int))
            else:
                label = str(names[class_id_int])
            detections.append(
                Detection(
                    class_id=class_id_int,
                    label=label,
                    confidence=float(confidence),
                    bbox_xyxy=tuple(float(value) for value in bbox),
                )
            )
        return detections
