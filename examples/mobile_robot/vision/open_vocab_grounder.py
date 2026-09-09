"""Open-vocabulary text grounding adapters for mobile-robot RGB frames.

The existing :class:`YoloDetector` remains a closed-set detector and is not
modified by this module.  ``YoloWorldGrounder`` is an optional, local-only
adapter used to evaluate natural-language referring expressions such as
``"a green pillar"`` or ``"the platform behind the pillar"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable, Protocol

import numpy as np

from .types import Detection, FramePacket, VisionResult, normalize_rgb_image


@dataclass(frozen=True)
class GroundingCandidate:
    """One text-conditioned image region returned by an open-vocabulary model."""

    prompt: str
    prompt_index: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_id: int
    sim_time: float
    model_name: str
    latency_ms: float

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if int(self.prompt_index) < 0:
            raise ValueError("prompt_index must be non-negative")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        bbox = tuple(float(value) for value in self.bbox_xyxy)
        if len(bbox) != 4 or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("bbox_xyxy must be ordered as x1, y1, x2, y2")
        object.__setattr__(self, "prompt_index", int(self.prompt_index))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "sim_time", float(self.sim_time))
        object.__setattr__(self, "latency_ms", float(self.latency_ms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_index": self.prompt_index,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "frame_id": self.frame_id,
            "sim_time": self.sim_time,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
        }


class OpenVocabularyGrounder(Protocol):
    """Stable protocol shared by text-conditioned visual backends.

    A backend must return every plausible region for the supplied prompts;
    selection, ambiguity handling, multi-frame confirmation and navigation
    remain outside the model adapter.
    """

    model_name: str
    device: str
    device_reason: str
    last_latency_ms: float | None

    def ground(self, frame: FramePacket, prompts: Iterable[str]) -> tuple[GroundingCandidate, ...]:
        """Return text-conditioned candidates for one RGB frame."""


def resolve_open_vocab_device(requested: str = "auto") -> tuple[str, str]:
    """Resolve ``auto`` as MPS first and CPU second.

    The second return value records why a fallback happened.  Explicit
    ``mps`` and ``cuda`` requests also fall back to CPU when the backend is not
    available, which keeps evaluation and demos runnable on CPU-only hosts.
    """

    requested = str(requested).strip().lower()
    if requested not in {"auto", "mps", "cpu", "cuda", "cuda:0"}:
        raise ValueError("device must be one of auto, mps, cpu, cuda or cuda:0")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project already depends on torch
        if requested == "cpu":
            return "cpu", "torch_unavailable_cpu"
        raise RuntimeError("open-vocabulary grounding requires torch") from exc

    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return "mps", "requested_mps" if requested == "mps" else "auto_mps"
    if requested in {"auto", "mps"}:
        fallback = "mps_unavailable_cpu" if requested == "mps" else "auto_cpu"
    else:
        fallback = "requested_cpu"
    # The mobile-robot deployment policy is deliberately MPS -> CPU.  CUDA is
    # still accepted when explicitly requested for Linux benchmark hosts, but
    # it must never silently change the default device policy.
    if requested in {"cuda", "cuda:0"} and torch.cuda.is_available():
        return "cuda:0", "auto_cuda" if requested == "auto" else "requested_cuda"
    if requested in {"cuda", "cuda:0"}:
        fallback = "cuda_unavailable_cpu"
    return "cpu", fallback


class YoloWorldGrounder:
    """Local YOLO-World adapter with prompt-conditioned candidate output.

    The model and CLIP text weights must already exist locally.  This class
    never downloads weights during inference, preserving the explicit local
    weight policy used by the closed-set YOLO adapter.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        image_size: int = 640,
        confidence: float = 0.05,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"open-vocabulary model does not exist: {path}")
        if path.suffix.lower() not in {".pt", ".onnx", ".engine"}:
            raise ValueError("model_path must be a local .pt, .onnx or .engine file")
        if int(image_size) <= 0:
            raise ValueError("image_size must be greater than zero")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("install ultralytics to use YoloWorldGrounder") from exc

        self.model_path = path.resolve()
        self.device, self.device_reason = resolve_open_vocab_device(device)
        self.image_size = int(image_size)
        self.confidence = float(confidence)
        self.model = YOLOWorld(str(self.model_path), verbose=False)
        self._prompts: tuple[str, ...] = ()
        self.last_latency_ms: float | None = None

    @property
    def model_name(self) -> str:
        return self.model_path.name

    def set_prompts(self, prompts: Iterable[str]) -> tuple[str, ...]:
        """Set the text vocabulary used by subsequent detections."""

        normalized = tuple(str(prompt).strip() for prompt in prompts if str(prompt).strip())
        if not normalized:
            raise ValueError("at least one non-empty prompt is required")
        if normalized != self._prompts:
            # Ultralytics' text model otherwise attempts to install CLIP and
            # download ViT-B/32 on demand.  Runtime inference is explicitly
            # offline, so fail before that code path when local assets are
            # missing.
            try:
                import clip  # noqa: F401
                from ultralytics.utils import WEIGHTS_DIR
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO-World prompts require the local CLIP package; "
                    "install the optional vision dependencies before inference"
                ) from exc
            clip_weights = Path(WEIGHTS_DIR) / "clip" / "ViT-B-32.pt"
            if not clip_weights.is_file():
                raise FileNotFoundError(
                    f"local CLIP weights do not exist: {clip_weights}; "
                    "runtime downloads are disabled"
                )
            try:
                self.model.set_classes(list(normalized))
            except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
                raise RuntimeError(
                    "YOLO-World text prompts require the optional CLIP package and "
                    "its local ViT-B/32 weights"
                ) from exc
            self._prompts = normalized
        return self._prompts

    def ground(self, frame: FramePacket, prompts: Iterable[str]) -> tuple[GroundingCandidate, ...]:
        """Ground prompts in one RGB frame and return all model candidates."""

        active_prompts = self.set_prompts(prompts)
        started = time.perf_counter()
        # Ultralytics' numpy inference path expects BGR; FramePacket remains RGB.
        image_bgr = np.ascontiguousarray(normalize_rgb_image(frame.image)[:, :, ::-1])
        try:
            prediction = self.model.predict(
                source=image_bgr,
                imgsz=self.image_size,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )[0]
        except RuntimeError as exc:
            # MPS can fail for an operator unsupported by the installed torch
            # build.  Retry the same frame on CPU while preserving the model.
            if self.device != "mps":
                raise
            self.device = "cpu"
            self.device_reason = "mps_runtime_error_cpu"
            prediction = self.model.predict(
                source=image_bgr,
                imgsz=self.image_size,
                conf=self.confidence,
                device="cpu",
                verbose=False,
            )[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.last_latency_ms = latency_ms
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return ()
        candidates: list[GroundingCandidate] = []
        for bbox, confidence, class_id in zip(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
        ):
            index = int(class_id)
            if index < 0 or index >= len(active_prompts):
                continue
            candidates.append(
                GroundingCandidate(
                    prompt=active_prompts[index],
                    prompt_index=index,
                    confidence=float(confidence),
                    bbox_xyxy=tuple(float(value) for value in bbox),
                    frame_id=frame.frame_id,
                    sim_time=frame.sim_time,
                    model_name=self.model_name,
                    latency_ms=latency_ms,
                )
            )
        return tuple(candidates)


class OpenVocabularyDetector:
    """Bridge an open-vocabulary grounder into the runtime vision contract.

    This adapter is intentionally perception-only.  It produces a
    :class:`VisionResult` for live camera logging/visualization, while the
    existing waypoint controller and LiDAR safety layer remain unchanged.
    """

    def __init__(
        self,
        model_path: str | Path,
        prompts: Iterable[str],
        *,
        backend: str = "yolo-world",
        device: str = "auto",
        image_size: int = 640,
        confidence: float = 0.001,
        decision_confidence: float = 0.05,
    ) -> None:
        normalized = tuple(str(prompt).strip() for prompt in prompts if str(prompt).strip())
        if not normalized:
            raise ValueError("open-vocabulary mode requires at least one prompt")
        if backend not in {"yolo-world", "owlv2"}:
            raise ValueError("backend must be yolo-world or owlv2")
        if not 0.0 <= float(decision_confidence) <= 1.0:
            raise ValueError("decision_confidence must be in [0, 1]")
        if backend == "yolo-world":
            self.grounder: OpenVocabularyGrounder = YoloWorldGrounder(
                model_path,
                device=device,
                image_size=image_size,
                confidence=confidence,
            )
        else:
            from .owlv2_grounder import Owlv2Grounder

            self.grounder = Owlv2Grounder(model_path, device=device, confidence=confidence)
        self.prompts = normalized
        self.decision_confidence = float(decision_confidence)

    @property
    def model_path(self) -> Path:
        return Path(getattr(self.grounder, "model_path"))

    @property
    def device(self) -> str:
        return self.grounder.device

    def detect(self, frame: FramePacket) -> VisionResult:
        candidates = self.grounder.ground(frame, self.prompts)
        best_by_prompt: dict[str, float] = {}
        for candidate in candidates:
            best_by_prompt[candidate.prompt] = max(
                best_by_prompt.get(candidate.prompt, 0.0), candidate.confidence
            )
        best_scores = sorted(best_by_prompt.values(), reverse=True)
        if not candidates:
            status = "no_visual_match"
        elif best_scores and best_scores[0] < self.decision_confidence:
            status = "low_confidence"
        elif len(best_scores) > 1 and best_scores[0] - best_scores[1] <= 0.05:
            status = "ambiguous"
        else:
            status = "candidate"
        detections = tuple(
            Detection(
                class_id=candidate.prompt_index,
                label=candidate.prompt,
                confidence=candidate.confidence,
                bbox_xyxy=candidate.bbox_xyxy,
            )
            for candidate in candidates
        )
        return VisionResult(
            frame_id=frame.frame_id,
            sim_time=frame.sim_time,
            model_name=self.grounder.model_name,
            latency_ms=float(self.grounder.last_latency_ms or 0.0),
            detections=detections,
            status=status,
            available=True,
        )
