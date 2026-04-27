#!/usr/bin/env python3
"""
Evaluate and inspect a trained YOLOv8 pole/tower model.

Usage examples:
  # 1) Quantitative evaluation on val split
  python3 test_model.py --mode val

  # 2) Prediction on a custom image folder
  python3 test_model.py --mode predict --source path/to/images

  # 3) Both in one run
  python3 test_model.py --mode both --source path/to/images
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test trained YOLOv8 model on val split and/or custom images."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/runs/pole_tower/m4pro_trial_v1/weights/best.pt"),
        help="Path to trained YOLO weights (.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/yolo_trial_dataset/data.generated.yaml"),
        help="YOLO data YAML used for evaluation.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split for evaluation mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["val", "predict", "both"],
        default="val",
        help="Run validation metrics, prediction export, or both.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Image file/folder for prediction mode.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="mps", help="e.g. mps, cpu, 0")
    parser.add_argument("--project", default="runs/eval")
    parser.add_argument("--name", default="m4pro_trial_v1_eval")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if args.mode in {"val", "both"} and not args.data.exists():
        raise SystemExit(f"Data YAML not found: {args.data}")
    if args.mode in {"predict", "both"} and args.source is None:
        raise SystemExit("--source is required for predict/both mode.")
    if args.source is not None and not args.source.exists():
        raise SystemExit(f"Source path not found: {args.source}")


def run_val(model, args: argparse.Namespace) -> None:
    print("\n=== Running validation ===")
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
        save_json=False,
        verbose=True,
    )

    # Ultralytics exposes summary values on metrics.box
    print("\nValidation summary:")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")


def run_predict(model, args: argparse.Namespace) -> None:
    print("\n=== Running predictions ===")
    model.predict(
        source=str(args.source.resolve()),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=f"{args.name}_predict",
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        verbose=True,
    )
    print("Prediction images saved under runs/eval/... with boxes and class labels.")


def main() -> None:
    args = parse_args()
    validate_paths(args)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install with: pip install ultralytics"
        ) from exc

    model = YOLO(str(args.model.resolve()))

    if args.mode in {"val", "both"}:
        run_val(model, args)
    if args.mode in {"predict", "both"}:
        run_predict(model, args)


if __name__ == "__main__":
    main()
