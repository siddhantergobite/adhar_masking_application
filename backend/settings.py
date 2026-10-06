from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "backend" / "models"


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    value = int(raw) if raw is not None else default
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    document_model_path: Path
    number_model_path: Path
    yolo_device: str
    document_confidence: float
    number_confidence: float
    document_iou: float
    inference_size: int
    ocr_confidence: float
    verification_confidence: float
    max_upload_bytes: int
    max_pages: int
    max_page_pixels: int
    pdf_scale: float
    min_region_edge: int
    min_blur_score: float
    orientation_attempts: int = 2
    ocr_max_long_edge: int = 1200
    document_fallback_confidence: float = 0.12
    number_fallback_confidence: float = 0.12

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            document_model_path=_env_path(
                "AADHAAR_DOCUMENT_MODEL",
                DEFAULT_MODEL_DIR / "aadhaar-document-seg.pt",
            ),
            number_model_path=_env_path(
                "AADHAAR_NUMBER_MODEL",
                DEFAULT_MODEL_DIR / "aadhaar-number-det.pt",
            ),
            yolo_device=os.getenv("AADHAAR_YOLO_DEVICE", "cpu"),
            document_confidence=_env_float("AADHAAR_DOCUMENT_CONFIDENCE", 0.72),
            number_confidence=_env_float("AADHAAR_NUMBER_CONFIDENCE", 0.62),
            document_iou=_env_float("AADHAAR_DOCUMENT_IOU", 0.45),
            # The document and number bootstrap models were trained at
            # 384/512. A 512-pixel pass is a good CPU/accuracy balance for the
            # local demo while retaining enough detail for the number model.
            inference_size=_env_int("AADHAAR_INFERENCE_SIZE", 512, 320),
            ocr_confidence=_env_float("AADHAAR_OCR_CONFIDENCE", 0.60),
            verification_confidence=_env_float("AADHAAR_VERIFICATION_CONFIDENCE", 0.68),
            max_upload_bytes=_env_int("AADHAAR_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
            max_pages=_env_int("AADHAAR_MAX_PAGES", 10),
            max_page_pixels=_env_int("AADHAAR_MAX_PAGE_PIXELS", 18_000_000),
            pdf_scale=float(os.getenv("AADHAAR_PDF_SCALE", "2.5")),
            min_region_edge=_env_int("AADHAAR_MIN_REGION_EDGE", 220, 80),
            min_blur_score=float(os.getenv("AADHAAR_MIN_BLUR_SCORE", "14.0")),
            # Some phone photos arrive with a physically landscape card whose
            # detected polygon is ordered 90 degrees off. Try all rotations
            # by default; the OCR result is still checksum and layout gated.
            orientation_attempts=min(_env_int("AADHAAR_ORIENTATION_ATTEMPTS", 4), 4),
            # PaddleOCR is CPU-bound on the full rectified page. OCR boxes are
            # mapped back to the full-resolution page before masking, so this
            # only reduces OCR cost and does not reduce output quality.
            ocr_max_long_edge=_env_int("AADHAAR_OCR_MAX_LONG_EDGE", 1200, 320),
            # The bootstrap YOLO weights are calibrated on clean material and
            # can score real phone captures below the strict primary threshold.
            # A second, lower-confidence pass is only used when the primary
            # pass returns no candidates; OCR, checksum and identity checks
            # remain mandatory before anything is emitted.
            document_fallback_confidence=_env_float("AADHAAR_DOCUMENT_FALLBACK_CONFIDENCE", 0.12),
            number_fallback_confidence=_env_float("AADHAAR_NUMBER_FALLBACK_CONFIDENCE", 0.12),
        )

