"""Stable data contracts for mobile-robot visual perception.

The package is intentionally independent from Genesis and from a particular
YOLO implementation. The simulator, an offline image reader, and a real
camera can therefore all produce the same :class:`FramePacket`.
"""

from .detector import DisabledDetector, VisionDetector
from .types import Detection, FramePacket, VisionResult, normalize_rgb_image

__all__ = [
    "Detection",
    "DisabledDetector",
    "FramePacket",
    "VisionDetector",
    "VisionResult",
    "normalize_rgb_image",
]
