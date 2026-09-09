"""Evaluate a local open-vocabulary grounder on saved robot-camera frames.

This is an offline, model-only benchmark.  It deliberately does not read
Genesis semantic objects or truth segmentation, so the output is suitable for
checking whether a text prompt produces usable candidates before navigation is
allowed to consume them.  Ground-truth scoring is a separate step because an
image directory alone does not establish which object a phrase refers to.

Example::

    .venv/bin/python examples/mobile_robot/vision/evaluate_open_vocab.py \
      --model models/mobile_robot/open_vocab/yolov8s-world.pt \
      --images out/mobile_robot_yolo_annotated_view/rgb_robot \
      --prompt "yellow car" --prompt "green pillar" --prompt platform \
      --device auto --imgsz 640 --infer-conf 0.001 --decision-conf 0.05 \
      --output-dir out/open_vocab_benchmark

The model path must already exist.  No weight download is performed by this
command or by :class:`YoloWorldGrounder`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from PIL import Image

try:  # Package import when invoked with ``python -m``.
    from .open_vocab_grounder import GroundingCandidate, YoloWorldGrounder
    from .owlv2_grounder import Owlv2Grounder
    from .overlay import annotate_rgb, iter_image_paths
    from .types import Detection, FramePacket, VisionResult
except ImportError:  # Direct script invocation from the repository root.
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from examples.mobile_robot.vision.open_vocab_grounder import GroundingCandidate, YoloWorldGrounder
    from examples.mobile_robot.vision.owlv2_grounder import Owlv2Grounder
    from examples.mobile_robot.vision.overlay import annotate_rgb, iter_image_paths
    from examples.mobile_robot.vision.types import Detection, FramePacket, VisionResult


def _build_grounder(args: argparse.Namespace) -> Any:
    """Construct the requested local backend without changing the evaluator contract."""

    if args.backend == "yolo-world":
        return YoloWorldGrounder(
            args.model,
            device=args.device,
            image_size=args.imgsz,
            confidence=args.infer_conf,
        )
    return Owlv2Grounder(args.model, device=args.device, confidence=args.infer_conf)


_FRAME_ID_RE = re.compile(r"(?:frame[_-])?(\d+)", re.IGNORECASE)


def _frame_id(path: Path, fallback: int) -> int:
    match = _FRAME_ID_RE.search(path.stem)
    return int(match.group(1)) if match else fallback


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _read_prompts(values: Iterable[str], prompt_file: Path | None) -> list[str]:
    prompts = [str(value).strip() for value in values if str(value).strip()]
    if prompt_file is not None:
        if not prompt_file.is_file():
            raise FileNotFoundError(f"prompt file does not exist: {prompt_file}")
        prompts.extend(line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines())
    normalized: list[str] = []
    for prompt in prompts:
        if prompt and prompt not in normalized:
            normalized.append(prompt)
    if not normalized:
        raise ValueError("provide at least one --prompt or --prompts-file")
    return normalized


def _candidate_status(
    candidates: tuple[GroundingCandidate, ...],
    *,
    decision_confidence: float,
    ambiguity_margin: float,
) -> str:
    """Classify a frame without pretending that a candidate is correct.

    ``no_visual_match`` means the model returned no boxes at its inference
    threshold.  ``low_confidence`` preserves the distinction when boxes exist
    but are below the navigation acceptance threshold.  ``ambiguous`` is only
    emitted when two prompts have similarly strong candidates; otherwise the
    frame is a normal ``candidate`` observation.
    """

    if not candidates:
        return "no_visual_match"
    above = [candidate for candidate in candidates if candidate.confidence >= decision_confidence]
    if not above:
        return "low_confidence"
    best_by_prompt: dict[str, float] = {}
    for candidate in above:
        best_by_prompt[candidate.prompt] = max(best_by_prompt.get(candidate.prompt, 0.0), candidate.confidence)
    best_scores = sorted(best_by_prompt.values(), reverse=True)
    if len(best_scores) >= 2 and best_scores[0] - best_scores[1] <= ambiguity_margin:
        return "ambiguous"
    return "candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("yolo-world", "owlv2"), default="yolo-world")
    parser.add_argument("--model", type=Path, required=True, help="existing local model file/directory")
    parser.add_argument("--images", type=Path, required=True, help="directory containing saved RGB frames")
    parser.add_argument("--prompt", action="append", default=[], help="text prompt; repeat for multiple prompts")
    parser.add_argument("--prompts-file", type=Path, help="UTF-8 file with one prompt per line")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda", "cuda:0"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--infer-conf", type=float, default=0.001, help="model pre-filter confidence")
    parser.add_argument("--decision-conf", type=float, default=0.05, help="candidate acceptance confidence")
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="only evaluate the first N frames; 0 means all")
    parser.add_argument("--annotate", action="store_true", help="write annotated RGB frames")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    model_exists = args.model.is_file() if args.backend == "yolo-world" else args.model.is_dir()
    if not model_exists:
        kind = "file" if args.backend == "yolo-world" else "directory"
        raise FileNotFoundError(f"open-vocabulary model {kind} does not exist: {args.model}")
    if not args.images.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {args.images}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if not 0.0 <= args.infer_conf <= 1.0:
        raise ValueError("--infer-conf must be in [0, 1]")
    if not 0.0 <= args.decision_conf <= 1.0:
        raise ValueError("--decision-conf must be in [0, 1]")
    if not 0.0 <= args.ambiguity_margin <= 1.0:
        raise ValueError("--ambiguity-margin must be in [0, 1]")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    prompts = _read_prompts(args.prompt, args.prompts_file)
    paths = iter_image_paths(args.images)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"no supported images found in {args.images}")

    grounder = _build_grounder(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "annotated"
    if args.annotate:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "candidates.jsonl"
    latencies: list[float] = []
    status_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    frame_records: list[dict[str, Any]] = []

    with jsonl_path.open("w", encoding="utf-8") as stream:
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                frame = FramePacket(
                    frame_id=_frame_id(path, index),
                    sim_time=float(index),
                    image=rgb,
                    camera_name="saved_robot_rgb",
                )
            candidates = grounder.ground(frame, prompts)
            status = _candidate_status(
                candidates,
                decision_confidence=args.decision_conf,
                ambiguity_margin=args.ambiguity_margin,
            )
            latency = float(grounder.last_latency_ms or 0.0)
            latencies.append(latency)
            status_counts[status] += 1
            for candidate in candidates:
                candidate_counts[candidate.prompt] += 1
            result = {
                "image": str(path.resolve()),
                "frame_id": frame.frame_id,
                "sim_time": frame.sim_time,
                "status": status,
                "candidate_count": len(candidates),
                "candidates": [candidate.to_dict() for candidate in candidates],
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            frame_records.append(result)
            if args.annotate:
                # Reuse the stable Vision overlay by translating candidates to
                # its Detection contract only at this offline boundary.
                detections = tuple(
                    Detection(
                        class_id=candidate.prompt_index,
                        label=candidate.prompt,
                        confidence=candidate.confidence,
                        bbox_xyxy=candidate.bbox_xyxy,
                    )
                    for candidate in candidates
                )
                annotated = annotate_rgb(
                    frame.image,
                    VisionResult(
                        frame_id=frame.frame_id,
                        sim_time=frame.sim_time,
                        model_name=grounder.model_name,
                        latency_ms=latency,
                        detections=detections,
                    ),
                )
                Image.fromarray(annotated).save(annotated_dir / path.name)

    summary = {
        "model": str(args.model.resolve()),
        "backend": args.backend,
        "model_name": grounder.model_name,
        "images": len(paths),
        "prompts": prompts,
        "device_requested": args.device,
        "device_used": grounder.device,
        "device_reason": grounder.device_reason,
        "image_size": args.imgsz,
        "infer_confidence": args.infer_conf,
        "decision_confidence": args.decision_conf,
        "ambiguity_margin": args.ambiguity_margin,
        "status_counts": dict(status_counts),
        "candidate_counts_by_prompt": dict(candidate_counts),
        "latency_ms": {
            "count_with_candidates": len(latencies),
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
            "mean": statistics.fmean(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "candidates_jsonl": str(jsonl_path.resolve()),
        "annotated_dir": str(annotated_dir.resolve()) if args.annotate else None,
        "note": (
            "This benchmark reports model candidates only; it does not prove target correctness. "
            "Use Genesis truth or a manually labelled OOD set for Recall/Top-1 scoring."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    try:
        summary = run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
