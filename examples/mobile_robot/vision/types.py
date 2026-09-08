"""Data structures shared by the simulator and visual detectors.

The first implementation phase only establishes the contract. It does not
depend on ``ultralytics`` or download model weights. Images entering this
layer are normalized to RGB, uint8, HWC format so that a later YOLO adapter
does not need to know how Genesis produced the image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


def normalize_rgb_image(image: Any) -> np.ndarray:
    """Return an image in contiguous ``uint8`` RGB HWC format.

    Genesis camera output is normally already uint8, but keeping the
    conversion here makes the boundary explicit and also handles float images
    in either ``[0, 1]`` or ``[0, 255]`` form. Channel order is deliberately
    not guessed: callers must pass an RGB image when constructing a packet.
    """

    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"expected one image, got batch shape {array.shape}")
        array = array[0]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        finite_array = array[np.isfinite(array)]
        if finite_array.size and float(np.max(finite_array)) <= 1.0:
            array = array * 255.0
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _optional_vector(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class FramePacket:
    """One timestamped image delivered from a camera to a detector."""

    frame_id: int
    sim_time: float
    image: np.ndarray = field(repr=False)
    encoding: str = "rgb8"
    camera_name: str = "robot_rgb_camera"
    camera_pose: tuple[float, ...] | None = None
    intrinsics: tuple[float, ...] | None = None
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        if not math.isfinite(float(self.sim_time)) or float(self.sim_time) < 0.0:
            raise ValueError("sim_time must be a finite non-negative value")
        if self.encoding != "rgb8":
            raise ValueError(f"unsupported image encoding: {self.encoding!r}")
        if not self.camera_name:
            raise ValueError("camera_name must not be empty")

        normalized = normalize_rgb_image(self.image)
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "sim_time", float(self.sim_time))
        object.__setattr__(self, "image", normalized)
        object.__setattr__(self, "camera_pose", _optional_vector(self.camera_pose))
        object.__setattr__(self, "intrinsics", _optional_vector(self.intrinsics))
        object.__setattr__(self, "height", int(normalized.shape[0]))
        object.__setattr__(self, "width", int(normalized.shape[1]))

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe metadata without copying the image into a log."""

        return {
            "frame_id": self.frame_id,
            "sim_time": self.sim_time,
            "width": self.width,
            "height": self.height,
            "encoding": self.encoding,
            "camera_name": self.camera_name,
            "camera_pose": list(self.camera_pose) if self.camera_pose is not None else None,
            "intrinsics": list(self.intrinsics) if self.intrinsics is not None else None,
        }


@dataclass(frozen=True)
class Detection:
    """One object returned by a detector.

    ``color``, ``shape`` and 3-D positions are optional because a plain YOLO
    detector does not provide them by itself. Later stages can fill them via
    RGB ROI analysis, segmentation and depth/LiDAR fusion.
    """

    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    color: str | None = None
    color_confidence: float | None = None
    shape: str | None = None
    distance_m: float | None = None
    position_camera: tuple[float, ...] | None = None
    position_robot: tuple[float, ...] | None = None
    position_world: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        bbox = tuple(float(value) for value in self.bbox_xyxy)
        if len(bbox) != 4:
            raise ValueError("bbox_xyxy must contain four values")
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError("bbox_xyxy must contain finite values")
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("bbox_xyxy must be ordered as x1, y1, x2, y2")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.color_confidence is not None and not 0.0 <= float(self.color_confidence) <= 1.0:
            raise ValueError("color_confidence must be in [0, 1]")
        if self.distance_m is not None and (not math.isfinite(float(self.distance_m)) or float(self.distance_m) < 0.0):
            raise ValueError("distance_m must be a finite non-negative value")

        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(
            self,
            "color_confidence",
            None if self.color_confidence is None else float(self.color_confidence),
        )
        object.__setattr__(self, "distance_m", None if self.distance_m is None else float(self.distance_m))
        object.__setattr__(self, "position_camera", _optional_vector(self.position_camera))
        object.__setattr__(self, "position_robot", _optional_vector(self.position_robot))
        object.__setattr__(self, "position_world", _optional_vector(self.position_world))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "color": self.color,
            "color_confidence": self.color_confidence,
            "shape": self.shape,
            "distance_m": self.distance_m,
            "position_camera": list(self.position_camera) if self.position_camera is not None else None,
            "position_robot": list(self.position_robot) if self.position_robot is not None else None,
            "position_world": list(self.position_world) if self.position_world is not None else None,
        }


@dataclass(frozen=True)
class VisionResult:
    """Timestamped detector output associated with one :class:`FramePacket`."""

    frame_id: int
    sim_time: float
    model_name: str
    latency_ms: float
    detections: tuple[Detection, ...] = ()
    status: str = "ok"
    available: bool = True

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        if not math.isfinite(float(self.sim_time)) or float(self.sim_time) < 0.0:
            raise ValueError("sim_time must be a finite non-negative value")
        if not math.isfinite(float(self.latency_ms)) or float(self.latency_ms) < 0.0:
            raise ValueError("latency_ms must be a finite non-negative value")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not self.status:
            raise ValueError("status must not be empty")

        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "sim_time", float(self.sim_time))
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        object.__setattr__(self, "detections", tuple(self.detections))
        object.__setattr__(self, "available", bool(self.available))

    def is_stale(self, current_sim_time: float, max_age_s: float) -> bool:
        """Return whether this result is too old for control use."""

        if max_age_s < 0.0:
            raise ValueError("max_age_s must be non-negative")
        return float(current_sim_time) - self.sim_time > max_age_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "sim_time": self.sim_time,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "available": self.available,
            "detections": [detection.to_dict() for detection in self.detections],
        }
