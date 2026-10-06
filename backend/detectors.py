from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .domain import NumberDetection, VisionDetection


DOCUMENT_CLASS_ALIASES = {
    "aadhaar_front": "aadhaar_front",
    "aadhar_front": "aadhaar_front",
    "aadhaar_card_front": "aadhaar_front",
    "aadhar_card_front": "aadhaar_front",
    "aadhaar_back": "aadhaar_back",
    "aadhar_back": "aadhaar_back",
    "aadhaar_card_back": "aadhaar_back",
    "aadhar_card_back": "aadhaar_back",
    "aadhaar_letter": "aadhaar_letter",
    "aadhar_letter": "aadhaar_letter",
    "aadhaar_page": "aadhaar_letter",
    "aadhar_page": "aadhaar_letter",
    "aadhaar_full_page": "aadhaar_letter",
    "aadhar_full_page": "aadhaar_letter",
}

NUMBER_CLASS_ALIASES = {
    "aadhaar_number": "aadhaar_number",
    "aadhar_number": "aadhaar_number",
    "uid_number": "aadhaar_number",
    "aadhaar_no": "aadhaar_number",
    "aadhar_no": "aadhaar_number",
}


def _normalise_class_name(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _model_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class DetectorConfigurationError(RuntimeError):
    pass


class UltralyticsDocumentSegmenter:
    engine_name = "Ultralytics YOLO instance segmentation"

    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        confidence: float,
        iou: float,
        image_size: int,
        fallback_confidence: float | None = None,
    ) -> None:
        if not model_path.is_file():
            raise DetectorConfigurationError(f"Missing document segmentation weights: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorConfigurationError(
                "Ultralytics is not installed. Install backend/requirements.txt."
            ) from exc

        try:
            self.model = YOLO(str(model_path), task="segment")
        except Exception as exc:
            raise DetectorConfigurationError(f"Could not load document segmentation weights: {exc}") from exc

        self.names = _model_names(self.model)
        recognised = {
            DOCUMENT_CLASS_ALIASES.get(_normalise_class_name(name))
            for name in self.names.values()
        }
        recognised.discard(None)
        if not recognised:
            expected = ", ".join(sorted(set(DOCUMENT_CLASS_ALIASES.values())))
            actual = ", ".join(self.names.values()) or "none"
            raise DetectorConfigurationError(
                f"Document model classes are [{actual}]. Expected Aadhaar classes such as [{expected}]."
            )

        self.device = device
        self.confidence = confidence
        self.fallback_confidence = max(
            0.01,
            min(
                float(fallback_confidence if fallback_confidence is not None else 0.12),
                confidence,
            ),
        )
        self.iou = iou
        self.image_size = image_size
        self.model_path = model_path

    def detect(self, image_rgb: np.ndarray) -> list[VisionDetection]:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        confidence_passes = [self.confidence]
        if self.fallback_confidence < self.confidence:
            confidence_passes.append(self.fallback_confidence)

        for confidence in confidence_passes:
            results = self.model.predict(
                source=image_bgr,
                conf=confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                retina_masks=True,
                max_det=12,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            if result.masks is None or result.masks.xy is None:
                raise RuntimeError("The document model returned boxes without segmentation masks.")

            classes = _as_numpy(result.boxes.cls).astype(int).reshape(-1)
            confidences = _as_numpy(result.boxes.conf).astype(float).reshape(-1)
            boxes = _as_numpy(result.boxes.xyxy).astype(np.float32).reshape(-1, 4)
            polygons = result.masks.xy
            detections: list[VisionDetection] = []
            for class_id, confidence, box, polygon in zip(classes, confidences, boxes, polygons):
                raw_name = self.names.get(int(class_id), str(class_id))
                canonical_name = DOCUMENT_CLASS_ALIASES.get(_normalise_class_name(raw_name))
                if canonical_name is None:
                    continue
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if len(points) < 4:
                    continue
                detections.append(
                    VisionDetection(
                        class_name=canonical_name,
                        confidence=float(confidence),
                        polygon=points,
                        box=np.asarray(box, dtype=np.float32),
                    )
                )
            if detections:
                return detections
        return []


class UltralyticsNumberDetector:
    engine_name = "Ultralytics YOLO Aadhaar-number detector"

    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        confidence: float,
        iou: float,
        image_size: int,
        fallback_confidence: float | None = None,
    ) -> None:
        if not model_path.is_file():
            raise DetectorConfigurationError(f"Missing Aadhaar-number weights: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorConfigurationError(
                "Ultralytics is not installed. Install backend/requirements.txt."
            ) from exc

        try:
            self.model = YOLO(str(model_path))
        except Exception as exc:
            raise DetectorConfigurationError(f"Could not load Aadhaar-number weights: {exc}") from exc

        self.names = _model_names(self.model)
        recognised = {
            NUMBER_CLASS_ALIASES.get(_normalise_class_name(name))
            for name in self.names.values()
        }
        recognised.discard(None)
        if "aadhaar_number" not in recognised:
            actual = ", ".join(self.names.values()) or "none"
            raise DetectorConfigurationError(
                f"Number model classes are [{actual}]. Expected a class named aadhaar_number."
            )

        self.device = device
        self.confidence = confidence
        self.fallback_confidence = max(
            0.01,
            min(
                float(fallback_confidence if fallback_confidence is not None else 0.12),
                confidence,
            ),
        )
        self.iou = iou
        self.image_size = image_size
        self.model_path = model_path

    def detect(self, image_rgb: np.ndarray) -> list[NumberDetection]:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        confidence_passes = [self.confidence]
        if self.fallback_confidence < self.confidence:
            confidence_passes.append(self.fallback_confidence)

        for confidence in confidence_passes:
            results = self.model.predict(
                source=image_bgr,
                conf=confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                max_det=20,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            classes = _as_numpy(result.boxes.cls).astype(int).reshape(-1)
            confidences = _as_numpy(result.boxes.conf).astype(float).reshape(-1)
            boxes = _as_numpy(result.boxes.xyxy).astype(np.float32).reshape(-1, 4)
            detections: list[NumberDetection] = []
            for class_id, confidence, box in zip(classes, confidences, boxes):
                raw_name = self.names.get(int(class_id), str(class_id))
                canonical_name = NUMBER_CLASS_ALIASES.get(_normalise_class_name(raw_name))
                if canonical_name != "aadhaar_number":
                    continue
                detections.append(NumberDetection(confidence=float(confidence), box=np.asarray(box, dtype=np.float32)))
            if detections:
                return detections
        return []

