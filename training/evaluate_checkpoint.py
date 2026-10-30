from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "documents": ROOT / "training" / "aadhaar-document-seg.yaml",
    "numbers": ROOT / "training" / "aadhaar-number.yaml",
}
IMAGE_SIZES = {"documents": 1280, "numbers": 960}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the locked test split once for a selected Aadhaar candidate checkpoint."
    )
    parser.add_argument("task", choices=tuple(DATASETS))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--confirm-final-test", action="store_true", help="Confirm this is the final locked test evaluation")
    parser.add_argument("--device", default="auto", help="auto, a CUDA device such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=0, help="Validation data-loader workers")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int)
    args = parser.parse_args()
    if not args.confirm_final_test:
        parser.error("The locked test set is reserved for one final evaluation; pass --confirm-final-test")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    device = args.device
    if device.lower() == "auto":
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"
    if args.workers < 0 or args.batch < 1:
        parser.error("--workers must be zero or greater and --batch must be positive")

    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(DATASETS[args.task]),
        split="test",
        imgsz=args.imgsz or IMAGE_SIZES[args.task],
        batch=args.batch,
        device=device,
        workers=args.workers,
        plots=False,
        project=str(ROOT / "runs"),
        name=f"{args.task}-locked-test",
        exist_ok=False,
    )
    values = getattr(metrics, "results_dict", {})
    report: dict[str, float] = {}
    for name, value in values.items():
        try:
            report[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    print(json.dumps({"task": args.task, "checkpoint": checkpoint.name, "metrics": report}, indent=2))


if __name__ == "__main__":
    main()
