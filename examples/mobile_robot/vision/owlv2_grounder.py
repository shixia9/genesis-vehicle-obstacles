"""Optional local OWLv2 backend implementing the open-vocabulary contract.

The adapter is intentionally local-only.  Point ``model_path`` at a directory
created with ``Owlv2ForObjectDetection.save_pretrained`` and
``Owlv2Processor.save_pretrained``; ``local_files_only=True`` prevents a
navigation run from unexpectedly contacting Hugging Face.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable

from .open_vocab_grounder import GroundingCandidate, resolve_open_vocab_device
from .types import FramePacket, normalize_rgb_image


class Owlv2Grounder:
    """Hugging Face OWLv2 adapter with MPS-first/CPU-fallback execution."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        confidence: float = 0.05,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"OWLv2 local model directory does not exist: {path}. "
                "Download and save the model before running inference."
            )
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install transformers to use Owlv2Grounder") from exc

        self.model_path = path.resolve()
        self.device, self.device_reason = resolve_open_vocab_device(device)
        self.confidence = float(confidence)
        try:
            self.processor = Owlv2Processor.from_pretrained(str(self.model_path), local_files_only=True)
            self.model = Owlv2ForObjectDetection.from_pretrained(
                str(self.model_path), local_files_only=True
            ).to(self.device)
        except OSError as exc:
            raise RuntimeError(
                f"incomplete OWLv2 local model at {self.model_path}; "
                "expected processor and model files"
            ) from exc
        self.model.eval()
        self.last_latency_ms: float | None = None

    @property
    def model_name(self) -> str:
        return self.model_path.name

    def ground(self, frame: FramePacket, prompts: Iterable[str]) -> tuple[GroundingCandidate, ...]:
        active_prompts = tuple(str(prompt).strip() for prompt in prompts if str(prompt).strip())
        if not active_prompts:
            raise ValueError("at least one non-empty prompt is required")
        started = time.perf_counter()
        image = normalize_rgb_image(frame.image)
        try:
            import torch

            inputs = self.processor(text=[list(active_prompts)], images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
            target_sizes = torch.tensor([(frame.height, frame.width)], device=self.device)
            result = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.confidence,
            )[0]
        except RuntimeError:
            if self.device != "mps":
                raise
            self.device = "cpu"
            self.device_reason = "mps_runtime_error_cpu"
            import torch

            inputs = self.processor(text=[list(active_prompts)], images=image, return_tensors="pt")
            inputs = {key: value.to("cpu") if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model.to("cpu")(**inputs)
            target_sizes = torch.tensor([(frame.height, frame.width)])
            result = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.confidence,
            )[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.last_latency_ms = latency_ms
        candidates: list[GroundingCandidate] = []
        for bbox, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            prompt_index = int(label.detach().cpu().item())
            if prompt_index < 0 or prompt_index >= len(active_prompts):
                continue
            candidates.append(
                GroundingCandidate(
                    prompt=active_prompts[prompt_index],
                    prompt_index=prompt_index,
                    confidence=float(score.detach().cpu().item()),
                    bbox_xyxy=tuple(float(value) for value in bbox.detach().cpu().tolist()),
                    frame_id=frame.frame_id,
                    sim_time=frame.sim_time,
                    model_name=self.model_name,
                    latency_ms=latency_ms,
                )
            )
        return tuple(candidates)
