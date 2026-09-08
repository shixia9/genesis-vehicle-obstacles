"""Run local YOLO inference on saved robot-camera images.

Example::

    python -m examples.mobile_robot.vision.offline_infer \
        --model models/obstacle_yolo.pt \
        --input-dir out/mobile_robot_observable/rgb_robot \
        --output-dir out/mobile_robot_vision

The model path is mandatory and must already exist. This command never asks
Ultralytics to download a default model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from PIL import Image

if __package__:
    from .overlay import annotate_rgb, iter_image_paths
    from .types import FramePacket
    from .yolo_detector import YoloDetector
else:  # Support ``python examples/mobile_robot/vision/offline_infer.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vision.overlay import annotate_rgb, iter_image_paths
    from vision.types import FramePacket
    from vision.yolo_detector import YoloDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Existing local YOLO weight file.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing RGB images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/mobile_robot_vision"),
        help="Directory for annotated images and JSONL results.",
    )
    parser.add_argument("--device", default="cpu", help="Inference device, normally cpu on this Mac.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images.")
    return parser.parse_args()


def frame_id_from_path(path: Path, fallback: int) -> int:
    match = re.search(r"frame_(\d+)", path.stem)
    return int(match.group(1)) if match else fallback


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {args.input_dir}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    image_paths = list(iter_image_paths(args.input_dir))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"no supported images found in: {args.input_dir}")

    detector = YoloDetector(
        args.model,
        device=args.device,
        confidence=args.conf,
        image_size=args.imgsz,
    )
    annotated_dir = args.output_dir / "rgb_robot_annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "detections.jsonl"

    detection_count = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for index, image_path in enumerate(image_paths):
            with Image.open(image_path) as image_file:
                rgb = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
            frame = FramePacket(
                frame_id=frame_id_from_path(image_path, index),
                # Offline image folders do not contain a reliable clock. The
                # source frame id remains authoritative; sim_time is 0 by contract.
                sim_time=0.0,
                image=rgb,
                camera_name="robot_rgb_camera_offline",
            )
            result = detector.detect(frame)
            annotated = annotate_rgb(frame.image, result)
            annotated_path = annotated_dir / image_path.name
            Image.fromarray(annotated).save(annotated_path)
            detection_count += len(result.detections)

            record = {
                "source_image": str(image_path),
                "annotated_image": str(annotated_path),
                "offline_sim_time": True,
                **result.to_dict(),
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{image_path.name}: detections={len(result.detections)} "
                f"latency_ms={result.latency_ms:.1f}"
            )

    summary = {
        "model": str(detector.model_path),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "images_processed": len(image_paths),
        "detections": detection_count,
        "jsonl": str(jsonl_path),
        "annotated_dir": str(annotated_dir),
        "device": args.device,
        "confidence": args.conf,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    try:
        summary = run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
