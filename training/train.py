from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "backend" / "models"


def train_model(
    *,
    base_model: str,
    dataset: Path,
    run_name: str,
    output_name: str,
    epochs: int,
    image_size: int,
    batch: int,
    device: str,
    workers: int,
    patience: int,
    freeze: int,
    degrees: float,
    translate: float,
    scale: float,
    perspective: float,
    mosaic: float,
) -> Path:
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset}")
    model = YOLO(base_model)
    metrics = model.train(
        data=str(dataset),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        patience=patience,
        device=device,
        workers=workers,
        project=str(ROOT / "runs"),
        name=run_name,
        exist_ok=False,
        seed=42,
        deterministic=True,
        cache=False,
        degrees=degrees,
        translate=translate,
        scale=scale,
        perspective=perspective,
        fliplr=0.0,
        flipud=0.0,
        mosaic=mosaic,
        close_mosaic=min(20, max(1, epochs // 4)) if mosaic > 0 else 0,
        freeze=freeze if freeze > 0 else None,
    )
    best = Path(metrics.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training completed without best.pt at {best}")

    # Validate the exact checkpoint that will be deployed.
    YOLO(str(best)).val(
        data=str(dataset),
        imgsz=image_size,
        batch=batch,
        device=device,
        workers=workers,
        split="test",
        project=str(ROOT / "runs"),
        name=f"{run_name}-test",
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_DIR / output_name
    shutil.copy2(best, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two private Aadhaar masking models.")
    parser.add_argument("--task", choices=("documents", "numbers", "all"), default="all")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze the first N model layers; useful for CPU bootstrap training.")
    parser.add_argument("--document-imgsz", type=int, default=1280)
    parser.add_argument("--number-imgsz", type=int, default=960)
    parser.add_argument("--document-base", default="yolo26n-seg.pt")
    parser.add_argument("--number-base", default="yolo26n.pt")
    parser.add_argument(
        "--document-data",
        type=Path,
        default=ROOT / "training" / "aadhaar-document-seg.yaml",
    )
    parser.add_argument(
        "--number-data",
        type=Path,
        default=ROOT / "training" / "aadhaar-number.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs: list[Path] = []
    if args.task in {"documents", "all"}:
        outputs.append(
            train_model(
                base_model=args.document_base,
                dataset=args.document_data.resolve(),
                run_name="aadhaar-document-seg",
                output_name="aadhaar-document-seg.pt",
                epochs=args.epochs,
                image_size=args.document_imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                patience=args.patience,
                freeze=args.freeze,
                degrees=180.0,
                translate=0.12,
                scale=0.45,
                perspective=0.0008,
                mosaic=0.35,
            )
        )
    if args.task in {"numbers", "all"}:
        outputs.append(
            train_model(
                base_model=args.number_base,
                dataset=args.number_data.resolve(),
                run_name="aadhaar-number-det",
                output_name="aadhaar-number-det.pt",
                epochs=args.epochs,
                image_size=args.number_imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                patience=args.patience,
                freeze=args.freeze,
                # The number detector sees an already-rectified document. Keep
                # text large and nearly upright instead of applying the scene-
                # level segmentation recipe to small printed digits.
                degrees=6.0,
                translate=0.04,
                scale=0.15,
                perspective=0.0001,
                mosaic=0.05,
            )
        )
    for output in outputs:
        print(f"Deployed reviewed checkpoint: {output}")


if __name__ == "__main__":
    main()

