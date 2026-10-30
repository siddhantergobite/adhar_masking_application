from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "backend" / "models"
EXPECTED_NAMES = {
    "documents": ["aadhaar_front", "aadhaar_back", "aadhaar_letter"],
    "numbers": ["aadhaar_number"],
}
DESTINATIONS = {
    "documents": MODEL_DIR / "aadhaar-document-seg.pt",
    "numbers": MODEL_DIR / "aadhaar-number-det.pt",
}


def checkpoint_names(checkpoint: Path) -> list[str]:
    model = YOLO(str(checkpoint))
    names = model.names
    if isinstance(names, dict):
        return [str(names[index]).strip().lower() for index in sorted(names)]
    return [str(name).strip().lower() for name in names]


def promote(task: str, checkpoint: Path, *, reviewed: bool, allow_replace: bool) -> Path:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if not reviewed:
        raise ValueError("Review the locked test results and pass --confirm-reviewed before promotion")
    destination = DESTINATIONS[task]
    if destination.exists() and not allow_replace:
        raise FileExistsError("Runtime weights already exist; pass --replace-runtime-checkpoint to replace them")
    actual_names = checkpoint_names(checkpoint)
    if actual_names != EXPECTED_NAMES[task]:
        raise ValueError(f"Checkpoint classes {actual_names} do not match the required {EXPECTED_NAMES[task]}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".pt", dir=MODEL_DIR)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a manually reviewed YOLO checkpoint to runtime weights.")
    parser.add_argument("task", choices=tuple(EXPECTED_NAMES))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--confirm-reviewed", action="store_true")
    parser.add_argument("--replace-runtime-checkpoint", action="store_true")
    args = parser.parse_args()
    destination = promote(
        args.task,
        args.checkpoint,
        reviewed=args.confirm_reviewed,
        allow_replace=args.replace_runtime_checkpoint,
    )
    print(f"Promoted reviewed checkpoint: {destination}")


if __name__ == "__main__":
    main()
