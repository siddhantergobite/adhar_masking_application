from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np

from .domain import OcrLine, OcrWord
from .errors import OcrFailure
from .geometry import enhance_for_ocr


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def configure_cpu_threads() -> int:
    """Use one consistent, capped thread count for CPU inference."""

    default_threads = min(4, os.cpu_count() or 4)
    requested_threads = _positive_int_env("AADHAAR_OCR_CPU_THREADS", default_threads)
    cpu_threads = min(requested_threads, default_threads, 4)
    # Set these before importing PyTorch/Paddle so their CPU runtimes do not
    # oversubscribe the laptop's cores. They are scoped to this service process.
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    # Apply the same cap to libraries that may otherwise create their own
    # machine-wide thread pools. Import torch only after the environment caps
    # are set, before Ultralytics initializes its models.
    cv2.setNumThreads(cpu_threads)
    try:
        import torch

        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only permits setting inter-op threads before parallel
            # work starts. The intra-op cap above still applies if initialized.
            pass
    except ImportError:
        pass
    return cpu_threads


def load_ocr_engine(device: str = "cpu") -> Any:
    cpu_threads = configure_cpu_threads()
    batch_size = _positive_int_env("AADHAAR_OCR_BATCH_SIZE", 8)

    from paddleocr import PaddleOCR

    return PaddleOCR(
        # The mobile English models keep word boxes but are substantially
        # faster than the default medium models on CPU. Override these names
        # for a higher-accuracy OCR profile when needed.
        text_detection_model_name=os.getenv("AADHAAR_OCR_DETECTION_MODEL", "PP-OCRv5_mobile_det"),
        text_recognition_model_name=os.getenv("AADHAAR_OCR_RECOGNITION_MODEL", "en_PP-OCRv5_mobile_rec"),
        text_recognition_batch_size=batch_size,
        device=device,
        enable_mkldnn=True,
        cpu_threads=cpu_threads,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
    )


def _prediction_payload(prediction: Any) -> dict[str, Any]:
    payload = getattr(prediction, "json", prediction)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise OcrFailure()
    result = payload.get("res", payload)
    if not isinstance(result, dict):
        raise OcrFailure()
    return result


def _word_box(raw_box: object) -> np.ndarray | None:
    values = np.asarray(raw_box, dtype=np.float32)
    if values.size == 4:
        left, top, right, bottom = values.reshape(4)
    elif values.size >= 8:
        points = values.reshape(-1, 2)
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
    else:
        return None
    if right <= left or bottom <= top:
        return None
    return np.array([left, top, right, bottom], dtype=np.float32)


def read_ocr_lines(
    engine: Any,
    image_rgb: np.ndarray,
    *,
    max_long_edge: int | None = None,
    enhance_after_resize: bool = False,
) -> list[OcrLine]:
    try:
        original_height, original_width = image_rgb.shape[:2]
        scale = 1.0
        if max_long_edge is not None:
            if max_long_edge < 1:
                raise ValueError("max_long_edge must be positive")
            longest_edge = max(original_height, original_width)
            if longest_edge > max_long_edge:
                scale = max_long_edge / longest_edge
                image_rgb = cv2.resize(
                    image_rgb,
                    (
                        max(1, int(round(original_width * scale))),
                        max(1, int(round(original_height * scale))),
                    ),
                    interpolation=cv2.INTER_AREA,
                )

        # Keep the OCR-facing dimensions unchanged while avoiding a full-size
        # CLAHE/sharpening pass on large rectified pages that are resized next.
        if enhance_after_resize:
            image_rgb = enhance_for_ocr(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        predictions = engine.predict(input=image_bgr)
        lines: list[OcrLine] = []
        for prediction in predictions:
            payload = _prediction_payload(prediction)
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            polygons = payload.get("rec_polys", payload.get("dt_polys", []))
            words_by_line = payload.get("text_word", [])
            word_boxes_by_line = payload.get("text_word_boxes", [])

            for index, (text, score, polygon) in enumerate(zip(texts, scores, polygons)):
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if len(points) < 4:
                    continue
                if scale != 1.0:
                    points = points / scale
                raw_words = words_by_line[index] if index < len(words_by_line) else []
                raw_word_boxes = word_boxes_by_line[index] if index < len(word_boxes_by_line) else []
                words: list[OcrWord] = []
                for raw_text, raw_box in zip(raw_words, raw_word_boxes):
                    box = _word_box(raw_box)
                    if box is not None:
                        if scale != 1.0:
                            box = box / scale
                        words.append(OcrWord(text=str(raw_text), box=box))
                lines.append(
                    OcrLine(
                        text=str(text),
                        confidence=float(score),
                        polygon=points,
                        words=tuple(words),
                    )
                )
        return lines
    except OcrFailure:
        raise
    except Exception as exc:
        raise OcrFailure() from exc

