"""Evaluate a recorded runtime perception log against an optional ground truth log.

The evaluator is deliberately offline: the running detector never receives
semantic registry data.  When a ground-truth log from the same deterministic
Genesis route is supplied, detections are matched by class and bbox IoU so the
report can quantify recall, precision, color accuracy and depth error.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-log", type=Path, required=True)
    parser.add_argument("--ground-truth-log", type=Path)
    parser.add_argument("--tracked-objects", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--match-mode", choices=("bbox", "position"), default="bbox")
    parser.add_argument("--position-threshold-m", type=float, default=0.75)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"log does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        records.append(value)
    return records


def _detections(record: dict[str, Any]) -> list[dict[str, Any]]:
    result = record.get("result")
    if not isinstance(result, dict):
        return []
    detections = result.get("detections", [])
    return [item for item in detections if isinstance(item, dict)]


def _bbox_iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = area_left + area_right - intersection
    return intersection / union if union > 0.0 else 0.0


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _position_distance(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    left_position = left.get("position_world")
    right_position = right.get("position_world")
    if not isinstance(left_position, list) or not isinstance(right_position, list):
        return None
    if len(left_position) < 3 or len(right_position) < 3:
        return None
    try:
        return math.sqrt(
            sum((float(left_position[index]) - float(right_position[index])) ** 2 for index in range(3))
        )
    except (TypeError, ValueError):
        return None


def _match_frame(
    predictions: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    iou_threshold: float,
    match_mode: str,
    position_threshold_m: float,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], list[dict[str, Any]], list[dict[str, Any]]]:
    # Score is IoU for bbox mode and negative distance for position mode.  In
    # both cases greedy matching starts with the strongest candidate and keeps
    # one prediction per ground-truth object.
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        prediction_label = prediction.get("label")
        for truth_index, expected in enumerate(truth):
            if prediction_label != expected.get("label"):
                continue
            if match_mode == "position":
                distance = _position_distance(prediction, expected)
                if distance is not None and distance <= position_threshold_m:
                    candidates.append((-distance, prediction_index, truth_index))
            else:
                score = _bbox_iou(prediction.get("bbox_xyxy", []), expected.get("bbox_xyxy", []))
                if score >= iou_threshold:
                    candidates.append((score, prediction_index, truth_index))
    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for score, prediction_index, truth_index in candidates:
        if prediction_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(prediction_index)
        used_truth.add(truth_index)
        matches.append((predictions[prediction_index], truth[truth_index], score if match_mode == "bbox" else -score))
    unmatched_predictions = [item for index, item in enumerate(predictions) if index not in used_predictions]
    unmatched_truth = [item for index, item in enumerate(truth) if index not in used_truth]
    return matches, unmatched_predictions, unmatched_truth


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be in (0, 1]")
    if args.position_threshold_m <= 0.0:
        raise ValueError("--position-threshold-m must be positive")
    prediction_records = _load_jsonl(args.prediction_log)
    truth_records = _load_jsonl(args.ground_truth_log) if args.ground_truth_log else []

    latencies = [
        value
        for record in prediction_records
        for value in [_finite_float(record.get("result", {}).get("latency_ms"))]
        if value is not None
    ]
    statuses = Counter(str(record.get("result", {}).get("status", "unknown")) for record in prediction_records)
    prediction_counts = Counter(
        str(detection.get("label")) for record in prediction_records for detection in _detections(record)
    )
    confidence_values = [
        value
        for record in prediction_records
        for detection in _detections(record)
        for value in [_finite_float(detection.get("confidence"))]
        if value is not None
    ]
    metrics: dict[str, Any] = {
        "prediction_frames": len(prediction_records),
        "status_counts": dict(statuses),
        "prediction_detection_counts": dict(prediction_counts),
        "mean_detection_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else None
        ),
        "latency_ms": {
            "count": len(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
            "max": max(latencies) if latencies else None,
        },
        "match_mode": args.match_mode,
        "iou_threshold": args.iou_threshold,
        "position_threshold_m": args.position_threshold_m,
    }

    if truth_records:
        truth_by_frame = {
            int(record.get("frame", {}).get("frame_id", record.get("result", {}).get("frame_id", -1))): record
            for record in truth_records
        }
        counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        color_totals: Counter[str] = Counter()
        color_correct: Counter[str] = Counter()
        distance_errors: defaultdict[str, list[float]] = defaultdict(list)
        position_errors: defaultdict[str, list[float]] = defaultdict(list)
        iou_values: defaultdict[str, list[float]] = defaultdict(list)
        match_distances: defaultdict[str, list[float]] = defaultdict(list)
        matched_frames = 0
        for prediction_record in prediction_records:
            frame = prediction_record.get("frame", {})
            frame_id = int(frame.get("frame_id", prediction_record.get("result", {}).get("frame_id", -1)))
            expected_record = truth_by_frame.get(frame_id)
            if expected_record is None:
                continue
            matched_frames += 1
            matches, unmatched_predictions, unmatched_truth = _match_frame(
                _detections(prediction_record),
                _detections(expected_record),
                args.iou_threshold,
                args.match_mode,
                args.position_threshold_m,
            )
            for prediction in unmatched_predictions:
                counts[str(prediction.get("label"))]["fp"] += 1
            for expected in unmatched_truth:
                counts[str(expected.get("label"))]["fn"] += 1
            for prediction, expected, iou in matches:
                label = str(expected.get("label"))
                counts[label]["tp"] += 1
                if args.match_mode == "bbox":
                    iou_values[label].append(iou)
                else:
                    match_distances[label].append(iou)
                if prediction.get("color") is not None and expected.get("color") is not None:
                    color_totals[label] += 1
                    if prediction.get("color") == expected.get("color"):
                        color_correct[label] += 1
                prediction_distance = _finite_float(prediction.get("distance_m"))
                expected_distance = _finite_float(expected.get("distance_m"))
                if prediction_distance is not None and expected_distance is not None:
                    distance_errors[label].append(abs(prediction_distance - expected_distance))
                prediction_position = prediction.get("position_world")
                expected_position = expected.get("position_world")
                if (
                    isinstance(prediction_position, list)
                    and isinstance(expected_position, list)
                    and len(prediction_position) >= 3
                    and len(expected_position) >= 3
                ):
                    try:
                        position_errors[label].append(
                            math.sqrt(
                                sum(
                                    (float(prediction_position[index]) - float(expected_position[index])) ** 2
                                    for index in range(3)
                                )
                            )
                        )
                    except (TypeError, ValueError):
                        pass
        classes = sorted(set(counts) | set(prediction_counts) | {str(d.get("label")) for r in truth_records for d in _detections(r)})
        per_class: dict[str, Any] = {}
        for label in classes:
            tp = counts[label]["tp"]
            fp = counts[label]["fp"]
            fn = counts[label]["fn"]
            per_class[label] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "mean_iou": sum(iou_values[label]) / len(iou_values[label]) if iou_values[label] else None,
                "mean_match_distance_m": (
                    sum(match_distances[label]) / len(match_distances[label]) if match_distances[label] else None
                ),
                "color_accuracy": (
                    color_correct[label] / color_totals[label] if color_totals[label] else None
                ),
                "color_comparisons": color_totals[label],
                "distance_abs_error_m_mean": (
                    sum(distance_errors[label]) / len(distance_errors[label]) if distance_errors[label] else None
                ),
                "distance_abs_error_m_p95": _percentile(distance_errors[label], 95.0),
                "position_world_error_m_mean": (
                    sum(position_errors[label]) / len(position_errors[label]) if position_errors[label] else None
                ),
                "position_world_error_m_p95": _percentile(position_errors[label], 95.0),
            }
        metrics["ground_truth_frames"] = len(truth_records)
        metrics["matched_frames"] = matched_frames
        metrics["per_class"] = per_class

    if args.tracked_objects:
        tracked = json.loads(args.tracked_objects.read_text(encoding="utf-8"))
        if not isinstance(tracked, list):
            raise ValueError("tracked objects must be a JSON list")
        tracks_by_category = Counter(str(item.get("category")) for item in tracked if isinstance(item, dict))
        metrics["tracked_object_count"] = len(tracked)
        metrics["tracks_by_category"] = dict(tracks_by_category)
        metrics["fragmented_categories"] = {
            category: count for category, count in tracks_by_category.items() if count > 1
        }

    summary = {
        "prediction_log": str(args.prediction_log.resolve()),
        "ground_truth_log": str(args.ground_truth_log.resolve()) if args.ground_truth_log else None,
        "tracked_objects": str(args.tracked_objects.resolve()) if args.tracked_objects else None,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    try:
        summary = run(parse_args())
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
