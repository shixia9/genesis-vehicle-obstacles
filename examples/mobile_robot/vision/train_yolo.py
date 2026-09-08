"""Train a local custom YOLO detector without downloading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset.yaml generated locally.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training project output directory.")
    parser.add_argument("--model", default="yolo11n.yaml", help="Bundled local model config, not a weight file.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda:0, ...")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, str | int | bool]:
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset yaml does not exist: {args.data}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("epochs, imgsz and batch must be positive; workers cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import only after argument validation. YOLO resolves yolo11n.yaml from
    # the installed local Ultralytics package; no pretrained weight or network
    # download is requested.
    from ultralytics import YOLO

    model = YOLO(args.model)
    result = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        pretrained=False,
        project=str(args.output_dir.resolve()),
        name="yolo11n_custom",
        exist_ok=True,
        cache=False,
        plots=True,
        verbose=False,
    )
    save_dir = Path(getattr(result, "save_dir", args.output_dir / "yolo11n_custom"))
    best = save_dir / "weights" / "best.pt"
    summary = {
        "data": str(args.data.resolve()),
        "model_config": args.model,
        "pretrained": False,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "save_dir": str(save_dir),
        "best_weights": str(best),
        "weights_exists": best.is_file(),
    }
    (save_dir / "training_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    try:
        summary = run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

