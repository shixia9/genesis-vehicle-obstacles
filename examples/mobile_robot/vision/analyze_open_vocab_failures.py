"""Summarize high-confidence open-vocabulary failures from an OOD run.

This is a post-inference diagnostic.  It reads model candidates from the
evaluator JSONL and joins Genesis annotations only to label failure cases; no
truth is passed back into inference or control.

Example::

    .venv/bin/python examples/mobile_robot/vision/analyze_open_vocab_failures.py \
      --dataset datasets/mobile_robot_open_vocab_ood \
      --results out/mobile_robot_open_vocab_dev_prompt_ensemble_v2/ood_results.jsonl \
      --output-dir out/mobile_robot_open_vocab_dev_prompt_ensemble_v2/failures
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def _iou(left: list[float] | tuple[float, float, float, float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _load_variants(results_path: Path) -> dict[str, str]:
    summary_path = results_path.with_name("summary.json")
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for canonical, variants in summary.get("prompt_variants", {}).items():
        for variant in variants:
            aliases[str(variant)] = str(canonical)
    return aliases


def _asset_variants(dataset: Path) -> dict[int, dict[str, str]]:
    path = dataset / "layouts.json"
    if not path.is_file():
        return {}
    result: dict[int, dict[str, str]] = {}
    for layout in json.loads(path.read_text(encoding="utf-8")):
        result[int(layout["layout_index"])] = {
            str(item["object_id"]): str(item.get("asset_variant", "unknown"))
            for item in layout.get("objects", [])
        }
    return result


def _failure_record(
    *,
    frame: dict[str, Any],
    prompt_record: dict[str, Any],
    candidates: list[dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    variants_by_layout: dict[int, dict[str, str]],
    failure_type: str,
) -> dict[str, Any]:
    target_objects = [truth[item_id] for item_id in prompt_record.get("target_object_ids", [])]
    target_boxes = [item["bbox_xyxy"] for item in target_objects if item.get("bbox_xyxy") is not None]
    best_iou = max(
        (_iou(candidate["bbox_xyxy"], box) for candidate in candidates for box in target_boxes),
        default=0.0,
    )
    best_confidence = max((float(candidate["confidence"]) for candidate in candidates), default=0.0)
    layout_variants = variants_by_layout.get(int(frame["layout_index"]), {})
    return {
        "failure_type": failure_type,
        "image": frame["image"],
        "split": frame["split"],
        "layout_index": frame["layout_index"],
        "prompt": prompt_record["prompt"],
        "kind": prompt_record["kind"],
        "target_object_ids": list(prompt_record.get("target_object_ids", [])),
        "asset_variants": sorted(
            layout_variants.get(item_id, "unknown")
            for item_id in prompt_record.get("target_object_ids", [])
        ),
        "target_visible": bool(prompt_record["target_visible"]),
        "target_exists_in_scene": bool(prompt_record["target_exists_in_scene"]),
        "candidate_count": len(candidates),
        "best_confidence": best_confidence,
        "best_iou": best_iou,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"dataset does not exist: {args.dataset}")
    if not args.results.is_file():
        raise FileNotFoundError(f"results JSONL does not exist: {args.results}")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    metadata = {
        item["image"]: item
        for item in (
            json.loads(line)
            for line in (args.dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    aliases = _load_variants(args.results)
    variants_by_layout = _asset_variants(args.dataset)
    failures: list[dict[str, Any]] = []
    for line in args.results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame_result = json.loads(line)
        frame = metadata[frame_result["image"]]
        truth = {item["object_id"]: item for item in frame["objects"]}
        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in frame_result.get("candidates", []):
            canonical = aliases.get(candidate["prompt"], candidate["prompt"])
            by_prompt[canonical].append(candidate)
        for prompt_record in frame_result.get("prompts", []):
            prompt = prompt_record["prompt"]
            candidates = [
                candidate
                for candidate in by_prompt.get(prompt, [])
                if float(candidate["confidence"]) >= args.min_confidence
            ]
            visible = bool(prompt_record["target_visible"])
            exists = bool(prompt_record["target_exists_in_scene"])
            if visible and not prompt_record["phrase_recall_hit"]:
                failure_type = "visible_miss"
            elif not visible and exists and candidates:
                failure_type = "out_of_view_false_positive"
            elif not visible and not exists and candidates:
                failure_type = "absent_false_positive"
            elif prompt_record["kind"] == "ambiguous" and len(candidates) < 2:
                failure_type = "ambiguity_missed"
            else:
                continue
            failures.append(
                _failure_record(
                    frame=frame,
                    prompt_record=prompt_record,
                    candidates=candidates,
                    truth=truth,
                    variants_by_layout=variants_by_layout,
                    failure_type=failure_type,
                )
            )

    by_type: Counter[str] = Counter(item["failure_type"] for item in failures)
    by_prompt: Counter[str] = Counter(item["prompt"] for item in failures)
    by_asset: Counter[str] = Counter(
        variant for item in failures for variant in item.get("asset_variants", [])
    )
    failures.sort(key=lambda item: (item["failure_type"], -item["best_confidence"]))
    top_by_type: dict[str, list[dict[str, Any]]] = {}
    for failure_type in sorted(by_type):
        top_by_type[failure_type] = [
            item for item in failures if item["failure_type"] == failure_type
        ][: args.top_k]
    result = {
        "purpose": "open_vocabulary_failure_diagnostics",
        "dataset": str(args.dataset.resolve()),
        "results": str(args.results.resolve()),
        "min_confidence": args.min_confidence,
        "failure_count": len(failures),
        "failure_counts": dict(sorted(by_type.items())),
        "failure_counts_by_prompt": dict(sorted(by_prompt.items())),
        "failure_counts_by_asset_variant": dict(sorted(by_asset.items())),
        "top_failures_by_type": top_by_type,
        "note": "Truth is joined only after inference for diagnostics; it is never passed to the detector or controller.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "failure_analysis.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["output"] = str(output_path.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    try:
        print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()

