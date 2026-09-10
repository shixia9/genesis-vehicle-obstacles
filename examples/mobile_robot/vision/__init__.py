"""Stable data contracts for mobile-robot visual perception.

The package is intentionally independent from Genesis and from a particular
YOLO implementation. The simulator, an offline image reader, and a real
camera can therefore all produce the same :class:`FramePacket`.
"""

from .detector import DisabledDetector, VisionDetector
from .ground_truth import GroundTruthDetector
from .attributes import dominant_color, enrich_colors
from .overlay import AnnotatedRgbView, annotate_rgb
from .rgbd_calibration import (
    PinholeIntrinsics,
    RgbdCalibration,
    build_genesis_rgbd_calibration,
    enrich_calibrated_depth,
    sample_aligned_depth,
)
from .rgbd_fusion import enrich_approximate_depth, sample_bbox_depth
from .tracker import ObjectTracker
from .types import Detection, FramePacket, VisionResult, normalize_rgb_image
from .yolo_detector import YoloDetector
from .open_vocab_grounder import (
    GroundingCandidate,
    OpenVocabularyGrounder,
    OpenVocabularyDetector,
    YoloWorldGrounder,
    resolve_open_vocab_device,
)
from .owlv2_grounder import Owlv2Grounder
from .open_vocab_validation import RegionEvidence, candidate_rank_score, region_evidence
from .demo_language import instruction_to_visual_prompt

__all__ = [
    "Detection",
    "DisabledDetector",
    "GroundTruthDetector",
    "FramePacket",
    "ObjectTracker",
    "dominant_color",
    "enrich_colors",
    "enrich_approximate_depth",
    "sample_bbox_depth",
    "PinholeIntrinsics",
    "RgbdCalibration",
    "build_genesis_rgbd_calibration",
    "enrich_calibrated_depth",
    "sample_aligned_depth",
    "annotate_rgb",
    "AnnotatedRgbView",
    "YoloDetector",
    "GroundingCandidate",
    "OpenVocabularyGrounder",
    "OpenVocabularyDetector",
    "YoloWorldGrounder",
    "resolve_open_vocab_device",
    "Owlv2Grounder",
    "RegionEvidence",
    "candidate_rank_score",
    "region_evidence",
    "instruction_to_visual_prompt",
    "VisionDetector",
    "VisionResult",
    "normalize_rgb_image",
]
