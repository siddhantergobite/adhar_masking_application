from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent


def train_model(
    *,
    base_model: str,
    dataset: Path,
    run_name: str,
    epochs: int,
    image_size: int,
    batch: int,
    device: str,
    workers: int,
    patience: int,
    freeze: int,
    cache: str,
    deterministic: bool,
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
        deterministic=deterministic,
        amp=device != "cpu",
        cache=False if cache == "none" else cache,
        degrees=degrees,
        translate=translate,
        scale=scale,
        perspective=perspective,
        fliplr=0.0,
        flipud=0.0,
        mosaic=mosaic,
        close_mosaic=min(20, max(1, epochs // 4)) if mosaic > 0 else 0,
        freeze=freeze if freeze > 0 else None,
        plots=False,
    )
    best = Path(metrics.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training completed without best.pt at {best}")

    # Use the validation split during iteration. Keep the locked test split for
    # the single final evaluation after a candidate has been selected.
    YOLO(str(best)).val(
        data=str(dataset),
        imgsz=image_size,
        batch=batch,
        device=device,
        workers=workers,
        split="val",
        plots=False,
        project=str(ROOT / "runs"),
        name=f"{run_name}-validation",
    )
    return best


def resolve_device(requested: str) -> str:
    if requested.lower() != "auto":
        return requested
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def resolve_workers(requested: str, device: str) -> int:
    if requested.lower() != "auto":
        workers = int(requested)
        if workers < 0:
            raise ValueError("--workers must be zero or greater")
        return workers
    logical_cores = os.cpu_count() or 2
    divisor = 4 if device == "cpu" else 2
    return min(8, max(0 if device == "cpu" else 2, logical_cores // divisor))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two private Aadhaar masking models.")
    parser.add_argument("--task", choices=("documents", "numbers", "all"), default="all")
    parser.add_argument("--device", default="auto", help="auto, a CUDA device such as 0, or cpu")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", default="auto", help="Data-loader workers, or auto")
    parser.add_argument("--cache", choices=("none", "ram", "disk"), default="none")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic kernels at some speed cost")
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
    device = resolve_device(args.device)
    workers = resolve_workers(args.workers, device)
    outputs: list[Path] = []
    if args.task in {"documents", "all"}:
        outputs.append(
            train_model(
                base_model=args.document_base,
                dataset=args.document_data.resolve(),
                run_name="aadhaar-document-seg",
                epochs=args.epochs,
                image_size=args.document_imgsz,
                batch=args.batch,
                device=device,
                workers=workers,
                patience=args.patience,
                freeze=args.freeze,
                cache=args.cache,
                deterministic=args.deterministic,
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
                epochs=args.epochs,
                image_size=args.number_imgsz,
                batch=args.batch,
                device=device,
                workers=workers,
                patience=args.patience,
                freeze=args.freeze,
                cache=args.cache,
                deterministic=args.deterministic,
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
        print(f"Candidate checkpoint for review: {output}")


if __name__ == "__main__":
    main()

