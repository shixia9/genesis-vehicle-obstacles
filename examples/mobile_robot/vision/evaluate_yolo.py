"""Evaluate a local YOLO detector on a held-out dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model.is_file():
        raise FileNotFoundError(f"model does not exist: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset yaml does not exist: {args.data}")
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("imgsz and batch must be positive")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=True,
        project=str(args.output_dir.resolve()),
        name=f"{args.split}_eval",
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    names = getattr(metrics, "names", None) or getattr(model, "names", None) or {}
    if isinstance(names, dict):
        class_names = [str(names[index]) for index in range(len(names))]
    else:
        class_names = [str(value) for value in names]

    def _per_class(values: Any) -> list[float]:
        """Convert Ultralytics' per-class arrays to JSON-safe floats."""

        if values is None:
            return []
        return [float(value) for value in values]

    per_class_precision = _per_class(getattr(box, "p", None))
    per_class_recall = _per_class(getattr(box, "r", None))
    per_class_map50 = _per_class(getattr(box, "ap50", None))
    per_class_map50_95 = _per_class(getattr(box, "ap", None))
    split_images = list((args.data.parent / "images" / args.split).glob("*.png"))
    summary = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "images": len(split_images),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "class_names": class_names,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_map50": per_class_map50,
        "per_class_map50_95": per_class_map50_95,
        "per_class_threshold_0_90": {
            class_names[index] if index < len(class_names) else str(index): {
                "precision": per_class_precision[index] if index < len(per_class_precision) else None,
                "recall": per_class_recall[index] if index < len(per_class_recall) else None,
                "map50": per_class_map50[index] if index < len(per_class_map50) else None,
                "passed": (
                    index < len(per_class_precision)
                    and index < len(per_class_recall)
                    and per_class_precision[index] >= 0.90
                    and per_class_recall[index] >= 0.90
                ),
            }
            for index in range(max(len(class_names), len(per_class_precision), len(per_class_recall)))
        },
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.split}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    try:
        summary = run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
