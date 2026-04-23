#!/usr/bin/env python3
"""
Prepare and train a YOLOv8 detector for:
  0 -> pole
  1 -> tower

Only uses:
  - dataset_final/Negative-python-confirmed
  - dataset_final/roboflow_raw_yolo8_format_folders
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_NAMES = ["pole", "tower"]


@dataclass
class Sample:
    image_path: Path
    label_path: Path | None  # None means "negative image / no object"
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare + optionally train YOLOv8 on pole/tower data."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset_final"),
        help="Path containing the two approved source folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/yolo_trial_dataset"),
        help="Where merged dataset + generated YAML are written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used for splitting."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio for merged dataset.",
    )
    parser.add_argument(
        "--include-incomplete-roboflow",
        action="store_true",
        help="Include Roboflow subsets that have many unlabeled positives.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually run YOLO training (default behavior is prepare-only).",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLOv8 model.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/pole_tower")
    parser.add_argument("--name", default="trial_v1")
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_names(names: Iterable[str]) -> list[str]:
    return [str(x).strip().lower() for x in names]


def validate_data_yaml(data_yaml: Path) -> None:
    data = load_yaml(data_yaml)
    names = normalize_names(data.get("names", []))
    nc = int(data.get("nc", -1))
    if nc != 2 or names[:2] != EXPECTED_NAMES:
        raise ValueError(
            f"Unexpected class mapping in {data_yaml}: nc={nc}, names={names}. "
            "Expected index 0='pole', index 1='tower'."
        )


def collect_roboflow_samples(
    roboflow_root: Path, include_incomplete: bool
) -> tuple[list[Sample], dict]:
    samples: list[Sample] = []
    report: dict = {}

    for ds in sorted(p for p in roboflow_root.iterdir() if p.is_dir()):
        data_yaml = ds / "data.yaml"
        if not data_yaml.exists():
            continue
        validate_data_yaml(data_yaml)

        train_images_dir = ds / "train" / "images"
        train_labels_dir = ds / "train" / "labels"
        images = list_images(train_images_dir) if train_images_dir.exists() else []
        label_files = (
            sorted(train_labels_dir.glob("*.txt")) if train_labels_dir.exists() else []
        )
        label_by_stem = {p.stem: p for p in label_files}

        missing_labels = 0
        cls_counts = Counter()
        for lp in label_files:
            txt = lp.read_text(encoding="utf-8").strip()
            if not txt:
                continue
            for line in txt.splitlines():
                parts = line.split()
                if parts:
                    cls_counts[parts[0]] += 1

        ds_samples: list[Sample] = []
        for img in images:
            label = label_by_stem.get(img.stem)
            if label is None:
                missing_labels += 1
            ds_samples.append(Sample(image_path=img, label_path=label, source=ds.name))

        missing_ratio = (missing_labels / len(images)) if images else 0.0
        keep = include_incomplete or missing_ratio <= 0.10

        report[ds.name] = {
            "images": len(images),
            "labels": len(label_files),
            "missing_labels": missing_labels,
            "missing_ratio": round(missing_ratio, 4),
            "class_tokens": dict(cls_counts),
            "kept_for_training": keep,
        }

        if keep:
            samples.extend(ds_samples)

    return samples, report


def collect_negative_samples(neg_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for folder in sorted(p for p in neg_root.iterdir() if p.is_dir()):
        for img in list_images(folder):
            samples.append(Sample(image_path=img, label_path=None, source=folder.name))
    return samples


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_label(dst: Path, src_label: Path | None) -> None:
    if src_label is None:
        dst.write_text("", encoding="utf-8")
        return
    dst.write_text(src_label.read_text(encoding="utf-8"), encoding="utf-8")


def symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def build_merged_dataset(
    samples: list[Sample], output_root: Path, val_ratio: float, seed: int
) -> tuple[Path, dict]:
    random.seed(seed)
    shuffled = samples[:]
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_set = shuffled[:val_size]
    train_set = shuffled[val_size:]

    for split in ("train", "val"):
        ensure_clean_dir(output_root / split / "images")
        ensure_clean_dir(output_root / split / "labels")

    split_map = {"train": train_set, "val": val_set}
    split_stats = defaultdict(lambda: {"images": 0, "negatives": 0})

    for split, split_samples in split_map.items():
        for idx, sample in enumerate(split_samples):
            safe_source = sample.source.replace(" ", "_").replace("+", "_")
            stem = f"{safe_source}__{sample.image_path.stem}__{idx}"
            img_dst = output_root / split / "images" / f"{stem}{sample.image_path.suffix.lower()}"
            lbl_dst = output_root / split / "labels" / f"{stem}.txt"

            symlink_or_copy(sample.image_path, img_dst)
            write_label(lbl_dst, sample.label_path)

            split_stats[split]["images"] += 1
            if sample.label_path is None:
                split_stats[split]["negatives"] += 1

    data_yaml_path = output_root / "data.generated.yaml"
    data = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": EXPECTED_NAMES,
        "nc": 2,
    }
    data_yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml_path, split_stats


def run_training(args: argparse.Namespace, data_yaml_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install with: pip install ultralytics"
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        pretrained=True,
        save=True,
        verbose=True,
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    negative_root = dataset_root / "Negative-python-confirmed"
    roboflow_root = dataset_root / "roboflow_raw_yolo8_format_folders"
    if not negative_root.exists() or not roboflow_root.exists():
        raise SystemExit(
            "Expected dataset directories not found. "
            "Need: dataset_final/Negative-python-confirmed and "
            "dataset_final/roboflow_raw_yolo8_format_folders"
        )

    roboflow_samples, roboflow_report = collect_roboflow_samples(
        roboflow_root, include_incomplete=args.include_incomplete_roboflow
    )
    negative_samples = collect_negative_samples(negative_root)
    all_samples = roboflow_samples + negative_samples

    if not all_samples:
        raise SystemExit("No training samples collected.")

    output_root = args.output_root.resolve()
    data_yaml_path, split_stats = build_merged_dataset(
        all_samples, output_root=output_root, val_ratio=args.val_ratio, seed=args.seed
    )

    print("\n=== Roboflow subset audit ===")
    for name, stats in roboflow_report.items():
        print(f"{name}: {stats}")

    print("\n=== Merged dataset summary ===")
    print(f"Generated data YAML: {data_yaml_path}")
    print(f"Train: {split_stats['train']}")
    print(f"Val:   {split_stats['val']}")
    print(f"Total positives+mixed source images used: {len(roboflow_samples)}")
    print(f"Total explicit negatives used: {len(negative_samples)}")
    print(
        f"Class mapping fixed as: index 0 -> {EXPECTED_NAMES[0]}, "
        f"index 1 -> {EXPECTED_NAMES[1]}"
    )

    if args.train:
        run_training(args, data_yaml_path)
    else:
        print(
            "\nPrepare-only mode complete. Training NOT started.\n"
            "Run with --train to start training once you're ready."
        )


if __name__ == "__main__":
    main()
