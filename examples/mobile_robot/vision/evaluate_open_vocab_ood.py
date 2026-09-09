"""Score YOLO-World on the evaluation-only Genesis OOD manifest.

The dataset contains Genesis truth masks and prompt manifests, but this tool
never feeds truth into the model.  It compares text-conditioned candidates to
the held-out instance boxes and depth/world positions after inference.  It is
therefore an evaluator, not a closed-set training pipeline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any

from PIL import Image
import numpy as np

try:
    from .open_vocab_grounder import GroundingCandidate, YoloWorldGrounder
    from .overlay import annotate_rgb
    from .rgbd_calibration import (
        PinholeIntrinsics,
        RgbdCalibration,
        enrich_calibrated_depth,
    )
    from .types import Detection, FramePacket, VisionResult
    from .open_vocab_validation import candidate_rank_score
except ImportError:  # pragma: no cover - direct script execution
    import sys

    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from examples.mobile_robot.vision.open_vocab_grounder import GroundingCandidate, YoloWorldGrounder
    from examples.mobile_robot.vision.overlay import annotate_rgb
    from examples.mobile_robot.vision.rgbd_calibration import (
        PinholeIntrinsics,
        RgbdCalibration,
        enrich_calibrated_depth,
    )
    from examples.mobile_robot.vision.types import Detection, FramePacket, VisionResult
    from examples.mobile_robot.vision.open_vocab_validation import candidate_rank_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda", "cuda:0"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--infer-conf", type=float, default=0.001)
    parser.add_argument("--decision-conf", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--split",
        choices=("all", "dev", "negative_test", "ood_test"),
        default="all",
        help="evaluate only one manifest split; use dev for tuning and freeze ood_test",
    )
    parser.add_argument(
        "--prompt-variants-file",
        type=Path,
        help=(
            "optional JSON object mapping canonical prompts to equivalent prompt variants; "
            "variants are ensembled and scored against the canonical prompt"
        ),
    )
    parser.add_argument(
        "--roi-rerank",
        action="store_true",
        help="use truth-free RGB colour/shape evidence only to rank accepted candidates",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _calibration_from_dict(data: dict[str, Any]) -> RgbdCalibration:
    def intrinsics(key: str) -> PinholeIntrinsics:
        values = data[key]
        return PinholeIntrinsics(
            width=int(values["width"]),
            height=int(values["height"]),
            fx=float(values["fx"]),
            fy=float(values["fy"]),
            cx=float(values["cx"]),
            cy=float(values["cy"]),
        )

    return RgbdCalibration(
        rgb=intrinsics("rgb"),
        depth=intrinsics("depth"),
        depth_from_rgb=np.asarray(data["depth_from_rgb"], dtype=np.float64),
        robot_from_depth=np.asarray(data["robot_from_depth"], dtype=np.float64),
        source=str(data.get("source", "genesis_known_intrinsics_extrinsics")),
    )


def _iou(left: tuple[float, float, float, float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _status(candidates: list[GroundingCandidate], decision_confidence: float) -> str:
    if not candidates:
        return "no_visual_match"
    accepted = [item for item in candidates if item.confidence >= decision_confidence]
    if not accepted:
        return "low_confidence"
    # Candidates are already grouped by the canonical prompt and de-duplicated
    # across prompt variants.  Two remaining accepted regions therefore mean
    # two plausible referents, even when they came from the same text phrase.
    if len(accepted) >= 2:
        return "ambiguous"
    by_prompt: dict[str, float] = {}
    for item in accepted:
        by_prompt[item.prompt] = max(by_prompt.get(item.prompt, 0.0), item.confidence)
    scores = sorted(by_prompt.values(), reverse=True)
    return "ambiguous" if len(scores) >= 2 and scores[0] - scores[1] <= 0.05 else "candidate"


def _position_world(
    candidate: GroundingCandidate,
    *,
    image: np.ndarray,
    depth: np.ndarray,
    calibration: RgbdCalibration,
    robot_pose: list[float],
) -> tuple[float, float, float] | None:
    detection = Detection(
        class_id=candidate.prompt_index,
        label=candidate.prompt,
        confidence=candidate.confidence,
        bbox_xyxy=candidate.bbox_xyxy,
    )
    result = VisionResult(
        frame_id=candidate.frame_id,
        sim_time=candidate.sim_time,
        model_name=candidate.model_name,
        latency_ms=candidate.latency_ms,
        detections=(detection,),
    )
    fused = enrich_calibrated_depth(result, depth, calibration, robot_pose=robot_pose)
    position = fused.detections[0].position_world
    return tuple(float(value) for value in position) if position is not None else None


def _best_candidate(
    candidates: list[GroundingCandidate],
    target_boxes: list[list[float]],
    *,
    decision_confidence: float,
    iou_threshold: float,
    rank_key: Any | None = None,
) -> tuple[GroundingCandidate | None, float, bool]:
    accepted = [item for item in candidates if item.confidence >= decision_confidence]
    if not accepted:
        return None, 0.0, False
    ranked = sorted(accepted, key=rank_key or (lambda item: item.confidence), reverse=True)
    best = ranked[0]
    best_iou = max((_iou(best.bbox_xyxy, box) for box in target_boxes), default=0.0)
    recall_hit = any(
        _iou(candidate.bbox_xyxy, box) >= iou_threshold
        for candidate in accepted
        for box in target_boxes
    )
    return best, best_iou, recall_hit


def _deduplicate_candidates(
    candidates: list[GroundingCandidate],
    *,
    iou_threshold: float = 0.7,
) -> list[GroundingCandidate]:
    """Collapse duplicate boxes emitted by equivalent prompt variants.

    Prompt ensembling can produce the same physical region once per wording.
    Keeping those copies would manufacture ambiguity and inflate the candidate
    count.  This is ordinary box-level NMS over the ensemble output; it does
    not use Genesis truth or a closed-set label list.
    """

    kept: list[GroundingCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if any(_iou(candidate.bbox_xyxy, list(previous.bbox_xyxy)) >= iou_threshold for previous in kept):
            continue
        kept.append(candidate)
    return kept


def _new_metric_bucket() -> dict[str, Any]:
    """Create a serializable metric accumulator for one evaluation slice."""

    return {
        "visible_total": 0,
        "recall_hits": 0,
        "top1_hits": 0,
        "out_of_view_total": 0,
        "out_of_view_false_positives": 0,
        "absent_total": 0,
        "absent_false_positives": 0,
        "ambiguous_total": 0,
        "ambiguous_detected": 0,
        "position_errors_m": [],
    }


def _metric_rates(bucket: dict[str, Any]) -> dict[str, Any]:
    """Convert an internal bucket into stable rates for JSON/reporting."""

    def rate(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    errors = [float(value) for value in bucket["position_errors_m"]]
    return {
        "phrase_grounding_recall": {
            "hits": bucket["recall_hits"],
            "total": bucket["visible_total"],
            "rate": rate(bucket["recall_hits"], bucket["visible_total"]),
        },
        "top1_selection": {
            "hits": bucket["top1_hits"],
            "total": bucket["visible_total"],
            "rate": rate(bucket["top1_hits"], bucket["visible_total"]),
        },
        "target_out_of_view_false_positive_rate": {
            "false_positives": bucket["out_of_view_false_positives"],
            "total": bucket["out_of_view_total"],
            "rate": rate(bucket["out_of_view_false_positives"], bucket["out_of_view_total"]),
        },
        "target_absent_false_positive_rate": {
            "false_positives": bucket["absent_false_positives"],
            "total": bucket["absent_total"],
            "rate": rate(bucket["absent_false_positives"], bucket["absent_total"]),
        },
        "ambiguity_detection_rate": {
            "detected": bucket["ambiguous_detected"],
            "total": bucket["ambiguous_total"],
            "rate": rate(bucket["ambiguous_detected"], bucket["ambiguous_total"]),
        },
        "rgbd_world_position_error_m": {
            "count": len(errors),
            "mean": statistics.fmean(errors) if errors else None,
            "p50": float(np.percentile(errors, 50)) if errors else None,
            "p95": float(np.percentile(errors, 95)) if errors else None,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model.is_file():
        raise FileNotFoundError(f"model does not exist: {args.model}")
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"dataset does not exist: {args.dataset}")
    if args.imgsz <= 0 or args.limit < 0:
        raise ValueError("imgsz must be positive and limit must be non-negative")
    if not 0.0 <= args.infer_conf <= 1.0 or not 0.0 <= args.decision_conf <= 1.0:
        raise ValueError("confidence thresholds must be in [0, 1]")
    if not 0.0 < args.iou_threshold <= 1.0:
        raise ValueError("iou-threshold must be in (0, 1]")

    metadata = [json.loads(line) for line in (args.dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.split != "all":
        metadata = [item for item in metadata if item.get("split") == args.split]
        if not metadata:
            raise ValueError(f"dataset contains no frames for split {args.split!r}")
    prompts_by_image = {
        item["image"]: item["prompts"]
        for item in (
            json.loads(line)
            for line in (args.dataset / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    layouts_path = args.dataset / "layouts.json"
    asset_variants_by_layout: dict[int, dict[str, str]] = {}
    if layouts_path.is_file():
        for layout in json.loads(layouts_path.read_text(encoding="utf-8")):
            asset_variants_by_layout[int(layout["layout_index"])] = {
                str(item["object_id"]): str(item.get("asset_variant", "unknown"))
                for item in layout.get("objects", [])
            }
    prompt_variants: dict[str, tuple[str, ...]] = {}
    variant_to_canonical: dict[str, str] = {}
    if args.prompt_variants_file is not None:
        if not args.prompt_variants_file.is_file():
            raise FileNotFoundError(f"prompt variants file does not exist: {args.prompt_variants_file}")
        raw_variants = json.loads(args.prompt_variants_file.read_text(encoding="utf-8"))
        if not isinstance(raw_variants, dict):
            raise ValueError("prompt variants file must contain a JSON object")
        for canonical, values in raw_variants.items():
            canonical_text = str(canonical).strip()
            if not canonical_text or not isinstance(values, list):
                raise ValueError("prompt variants must map non-empty prompts to JSON arrays")
            normalized = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
            if not normalized:
                raise ValueError(f"prompt variants for {canonical!r} must not be empty")
            prompt_variants[canonical_text] = normalized
            for variant in normalized:
                previous = variant_to_canonical.get(variant)
                if previous is not None and previous != canonical_text:
                    raise ValueError(f"prompt variant {variant!r} maps to multiple canonical prompts")
                variant_to_canonical[variant] = canonical_text
    if args.limit:
        metadata = metadata[: args.limit]
    calibration = _calibration_from_dict(json.loads((args.dataset / "camera_calibration.json").read_text(encoding="utf-8")))
    grounder = YoloWorldGrounder(
        args.model,
        device=args.device,
        image_size=args.imgsz,
        confidence=args.infer_conf,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "annotated"
    if args.annotate:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "ood_results.jsonl"

    visible_total = visible_hits = top1_total = top1_hits = 0
    absent_total = absent_false_positive = 0
    out_of_view_total = out_of_view_false_positive = 0
    ambiguous_total = ambiguous_detected = 0
    position_errors: list[float] = []
    latencies: list[float] = []
    status_counts: Counter[str] = Counter()
    prompt_kind_counts: Counter[str] = Counter()
    breakdown_by_prompt: dict[str, dict[str, Any]] = {}
    breakdown_by_split: dict[str, dict[str, Any]] = {}
    breakdown_by_asset_variant: dict[str, dict[str, Any]] = {}
    breakdown_by_visibility: dict[str, dict[str, Any]] = {}

    with result_path.open("w", encoding="utf-8") as stream:
        for index, frame_meta in enumerate(metadata):
            image_path = args.dataset / frame_meta["image"]
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            frame = FramePacket(
                frame_id=index,
                sim_time=float(frame_meta["sim_time"]),
                image=image,
                camera_name="ood_robot_rgb",
            )
            prompt_records = prompts_by_image[frame_meta["image"]]
            canonical_prompts = [record["prompt"] for record in prompt_records]
            prompt_texts = list(
                dict.fromkeys(
                    variant
                    for canonical in canonical_prompts
                    for variant in prompt_variants.get(canonical, (canonical,))
                )
            )
            candidates = grounder.ground(frame, prompt_texts)
            latency = float(grounder.last_latency_ms or 0.0)
            latencies.append(latency)
            by_prompt: dict[str, list[GroundingCandidate]] = {prompt: [] for prompt in canonical_prompts}
            for candidate in candidates:
                canonical = variant_to_canonical.get(candidate.prompt, candidate.prompt)
                by_prompt.setdefault(canonical, []).append(candidate)
            truth_by_id = {item["object_id"]: item for item in frame_meta["objects"]}
            frame_report: dict[str, Any] = {
                "image": frame_meta["image"],
                "split": frame_meta["split"],
                "frame_id": index,
                "candidates": [item.to_dict() for item in candidates],
                "prompts": [],
            }
            for prompt_record in prompt_records:
                prompt = prompt_record["prompt"]
                prompt_candidates = _deduplicate_candidates(by_prompt.get(prompt, []))
                status = _status(prompt_candidates, args.decision_conf)
                status_counts[status] += 1
                kind = str(prompt_record["kind"])
                prompt_kind_counts[kind] += 1
                target_objects = [truth_by_id[item_id] for item_id in prompt_record["target_object_ids"]]
                target_boxes = [item["bbox_xyxy"] for item in target_objects if item["bbox_xyxy"] is not None]
                best, best_iou, recall_hit = _best_candidate(
                    prompt_candidates,
                    target_boxes,
                    decision_confidence=args.decision_conf,
                    iou_threshold=args.iou_threshold,
                    rank_key=(lambda item: candidate_rank_score(image, item)) if args.roi_rerank else None,
                )
                if prompt_record["target_visible"]:
                    visible_total += 1
                    visible_hits += int(recall_hit)
                    top1_total += 1
                    top1_hits += int(best is not None and best_iou >= args.iou_threshold)
                elif prompt_record["target_exists_in_scene"]:
                    out_of_view_total += 1
                    out_of_view_false_positive += int(best is not None)
                else:
                    absent_total += 1
                    absent_false_positive += int(best is not None)
                accepted_count = sum(item.confidence >= args.decision_conf for item in prompt_candidates)
                if kind == "ambiguous":
                    ambiguous_total += 1
                    ambiguous_detected += int(accepted_count >= 2)

                # Keep the same event accounting in multiple, orthogonal
                # slices.  This makes a good aggregate score auditable: a
                # failure can be attributed to a phrase, asset variant,
                # split, or visibility condition without touching test data.
                metric_keys = [
                    (breakdown_by_prompt, prompt),
                    (breakdown_by_split, str(frame_meta["split"])),
                ]
                target_variants = {
                    asset_variants_by_layout.get(int(frame_meta["layout_index"]), {}).get(item["object_id"], "unknown")
                    for item in target_objects
                }
                metric_keys.extend((breakdown_by_asset_variant, key) for key in sorted(target_variants))
                if prompt_record["target_visible"]:
                    visibility_key = "visible_truncated" if any(item.get("truncated") for item in target_objects) else "visible"
                elif prompt_record["target_exists_in_scene"]:
                    visibility_key = "out_of_view"
                else:
                    visibility_key = "absent"
                metric_keys.append((breakdown_by_visibility, visibility_key))
                for collection, key in metric_keys:
                    bucket = collection.setdefault(key, _new_metric_bucket())
                    if prompt_record["target_visible"]:
                        bucket["visible_total"] += 1
                        bucket["recall_hits"] += int(recall_hit)
                        bucket["top1_hits"] += int(best is not None and best_iou >= args.iou_threshold)
                    elif prompt_record["target_exists_in_scene"]:
                        bucket["out_of_view_total"] += 1
                        bucket["out_of_view_false_positives"] += int(best is not None)
                    else:
                        bucket["absent_total"] += 1
                        bucket["absent_false_positives"] += int(best is not None)
                    if kind == "ambiguous":
                        bucket["ambiguous_total"] += 1
                        bucket["ambiguous_detected"] += int(accepted_count >= 2)

                position_error = None
                if best is not None and target_objects:
                    depth_path = args.dataset / frame_meta["depth"]
                    depth = np.load(depth_path)
                    estimated = _position_world(
                        best,
                        image=image,
                        depth=depth,
                        calibration=calibration,
                        robot_pose=frame_meta["robot_pose"],
                    )
                    if estimated is not None:
                        target_positions = [np.asarray(item["position_world"], dtype=np.float64) for item in target_objects]
                        position_error = min(float(np.linalg.norm(np.asarray(estimated) - target)) for target in target_positions)
                        position_errors.append(position_error)
                        for collection, key in metric_keys:
                            collection[key]["position_errors_m"].append(position_error)
                frame_report["prompts"].append(
                    {
                        **prompt_record,
                        "status": status,
                        "candidate_count": len(prompt_candidates),
                        "best_iou": best_iou,
                        "phrase_recall_hit": recall_hit,
                        "top1_hit": bool(best is not None and best_iou >= args.iou_threshold),
                        "best_prompt": best.prompt if best is not None else None,
                        "best_rank_score": (
                            candidate_rank_score(image, best) if best is not None and args.roi_rerank else None
                        ),
                        "position_error_m": position_error,
                    }
                )
            stream.write(json.dumps(frame_report, ensure_ascii=False) + "\n")
            if args.annotate:
                detections = tuple(
                    Detection(
                        class_id=item.prompt_index,
                        label=item.prompt,
                        confidence=item.confidence,
                        bbox_xyxy=item.bbox_xyxy,
                    )
                    for item in candidates
                )
                annotated = annotate_rgb(
                    image,
                    VisionResult(
                        frame_id=index,
                        sim_time=frame.frame_id,
                        model_name=grounder.model_name,
                        latency_ms=latency,
                        detections=detections,
                    ),
                )
                Image.fromarray(annotated).save(annotated_dir / image_path.name)

    def rate(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    summary = {
        "purpose": "open_vocabulary_evaluation_only",
        "closed_set_training": False,
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "images": len(metadata),
        "device_used": grounder.device,
        "device_reason": grounder.device_reason,
        "decision_confidence": args.decision_conf,
        "iou_threshold": args.iou_threshold,
        "prompt_variants_file": str(args.prompt_variants_file.resolve()) if args.prompt_variants_file else None,
        "prompt_variants": {key: list(value) for key, value in prompt_variants.items()},
        "roi_rerank": bool(args.roi_rerank),
        "status_counts": dict(status_counts),
        "prompt_kind_counts": dict(prompt_kind_counts),
        "phrase_grounding_recall": {"hits": visible_hits, "total": visible_total, "rate": rate(visible_hits, visible_total)},
        "top1_selection": {"hits": top1_hits, "total": top1_total, "rate": rate(top1_hits, top1_total)},
        "target_absent_false_positive_rate": {
            "false_positives": absent_false_positive,
            "total": absent_total,
            "rate": rate(absent_false_positive, absent_total),
        },
        "target_out_of_view_false_positive_rate": {
            "false_positives": out_of_view_false_positive,
            "total": out_of_view_total,
            "rate": rate(out_of_view_false_positive, out_of_view_total),
        },
        "ambiguity_detection_rate": {
            "detected": ambiguous_detected,
            "total": ambiguous_total,
            "rate": rate(ambiguous_detected, ambiguous_total),
        },
        "rgbd_world_position_error_m": {
            "count": len(position_errors),
            "mean": statistics.fmean(position_errors) if position_errors else None,
            "p50": float(np.percentile(position_errors, 50)) if position_errors else None,
            "p95": float(np.percentile(position_errors, 95)) if position_errors else None,
        },
        "latency_ms": {
            "count": len(latencies),
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "breakdown_by_prompt": {key: _metric_rates(value) for key, value in sorted(breakdown_by_prompt.items())},
        "breakdown_by_split": {key: _metric_rates(value) for key, value in sorted(breakdown_by_split.items())},
        "breakdown_by_asset_variant": {
            key: _metric_rates(value) for key, value in sorted(breakdown_by_asset_variant.items())
        },
        "breakdown_by_visibility": {
            key: _metric_rates(value) for key, value in sorted(breakdown_by_visibility.items())
        },
        "results_jsonl": str(result_path.resolve()),
        "annotated_dir": str(annotated_dir.resolve()) if args.annotate else None,
        "note": "Truth annotations are used only after inference for scoring; they are never passed to YOLO-World.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    try:
        print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
