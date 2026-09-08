"""Detector interface used by the mobile-robot simulation.

This module intentionally has no YOLO dependency. A later Ultralytics, ONNX,
TensorRT or test detector can implement the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import FramePacket, VisionResult


class VisionDetector(Protocol):
    """Minimal synchronous detector contract."""

    def detect(self, frame: FramePacket) -> VisionResult:
        """Run inference for exactly one frame and return a timestamped result."""


@dataclass(frozen=True)
class DisabledDetector:
    """Explicit no-model detector for baseline and fallback tests.

    The URDF example does not invoke this class by default; it keeps
    ``observation['vision']`` as ``None`` when visual inference is disabled.
    It is provided for callers that need a detector-shaped object without
    pretending that an empty result came from a loaded model.
    """

    reason: str = "disabled"

    def detect(self, frame: FramePacket) -> VisionResult:
        return VisionResult(
            frame_id=frame.frame_id,
            sim_time=frame.sim_time,
            model_name="none",
            latency_ms=0.0,
            detections=(),
            status=self.reason,
            available=False,
        )
