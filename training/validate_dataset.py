from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def validate_label(label_path: Path, *, task: str, class_count: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        expected_minimum = 7 if task == "segment" else 5
        if len(parts) < expected_minimum:
            raise ValueError(f"{label_path}:{line_number}: too few values for {task}")
        if task == "detect" and len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: detection labels need exactly 5 values")
        if task == "segment" and (len(parts) - 1) % 2:
            raise ValueError(f"{label_path}:{line_number}: polygon coordinates must be x/y pairs")
        class_id = int(parts[0])
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number}: class {class_id} is outside the schema")
        coordinates = [float(value) for value in parts[1:]]
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: coordinates must be normalized to [0, 1]")
        if task == "detect" and (coordinates[2] <= 0 or coordinates[3] <= 0):
            raise ValueError(f"{label_path}:{line_number}: box width and height must be positive")
        counts[class_id] += 1
    return counts


def audit(root: Path, *, task: str, class_count: int) -> None:
    total_images = 0
    total_negatives = 0
    class_totals: Counter[int] = Counter()
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images/{split} or labels/{split} under {root}")
        split_images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not split_images:
            raise ValueError(f"No images found in {image_dir}")
        split_negatives = 0
        for image_path in split_images:
            with Image.open(image_path) as image:
                image.verify()
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label file (empty is valid for a negative): {label_path}")
            counts = validate_label(label_path, task=task, class_count=class_count)
            if not counts:
                split_negatives += 1
            class_totals.update(counts)
        total_images += len(split_images)
        total_negatives += split_negatives
        print(f"{split}: {len(split_images)} images, {split_negatives} hard negatives")
    if total_negatives == 0:
        raise ValueError("No hard-negative images were found. Non-Aadhaar and mixed-card scenes are required.")
    print(f"total: {total_images} images, {total_negatives} hard negatives, labels {dict(class_totals)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO Aadhaar datasets before training.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--task", choices=("segment", "detect"), required=True)
    parser.add_argument("--classes", type=int, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), task=args.task, class_count=args.classes)


if __name__ == "__main__":
    main()

