from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .domain import VisionDetection
from .geometry import box_iou


@dataclass(frozen=True)
class _Candidate:
    area_fraction: float
    rectangularity: float
    polygon: np.ndarray
    box: np.ndarray

    @property
    def area(self) -> float:
        return max(0.0, float(self.box[2] - self.box[0])) * max(
            0.0, float(self.box[3] - self.box[1])
        )


def _intersection_over_smaller(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0 else 0.0


class OpenCvDocumentProposer:
    """Find rectangular card/page candidates without claiming they are Aadhaar.

    These proposals are only a geometric fallback. The pipeline applies much
    stronger OCR, checksum, identity-text and post-redaction checks before a
    proposal can produce output.
    """

    engine_name = "OpenCV quadrilateral candidate proposal"

    def __init__(
        self,
        *,
        min_area_fraction: float = 0.035,
        max_candidates: int = 8,
        analysis_long_edge: int = 1200,
    ) -> None:
        self.min_area_fraction = min_area_fraction
        self.max_candidates = max_candidates
        self.analysis_long_edge = analysis_long_edge

    def detect(self, image_rgb: np.ndarray) -> list[VisionDetection]:
        image_height, image_width = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        scale = min(1.0, self.analysis_long_edge / max(image_height, image_width))
        if scale < 1.0:
            work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            work = gray

        blurred = cv2.GaussianBlur(work, (5, 5), 0)
        variants = [
            cv2.Canny(blurred, 40, 120),
            cv2.Canny(blurred, 70, 180),
        ]
        for threshold in (110, 150, 185, 210):
            _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
            variants.append(binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        raw: list[_Candidate] = []
        work_area = float(work.shape[0] * work.shape[1])
        for variant in variants:
            closed = cv2.morphologyEx(variant, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_area = abs(float(cv2.contourArea(contour)))
                area_fraction = contour_area / work_area
                if not self.min_area_fraction <= area_fraction <= 0.96:
                    continue

                perimeter = cv2.arcLength(contour, True)
                approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                    continue

                rectangle_width, rectangle_height = cv2.minAreaRect(contour)[1]
                if min(rectangle_width, rectangle_height) < 40:
                    continue
                rectangularity = contour_area / (rectangle_width * rectangle_height)
                if rectangularity < 0.65:
                    continue

                polygon = approximation.reshape(4, 2).astype(np.float32) / scale
                polygon[:, 0] = np.clip(polygon[:, 0], 0, image_width - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, image_height - 1)
                box = np.array(
                    [
                        polygon[:, 0].min(),
                        polygon[:, 1].min(),
                        polygon[:, 0].max(),
                        polygon[:, 1].max(),
                    ],
                    dtype=np.float32,
                )
                candidate = _Candidate(area_fraction, rectangularity, polygon, box)
                duplicate_index = next(
                    (index for index, existing in enumerate(raw) if box_iou(box, existing.box) >= 0.92),
                    None,
                )
                if duplicate_index is None:
                    raw.append(candidate)
                elif rectangularity > raw[duplicate_index].rectangularity:
                    raw[duplicate_index] = candidate

        # Nested near-page boundaries are common in screenshots. When the
        # inner rectangle is a substantial part of the outer one, it is the
        # document; when it is tiny, it is usually a portrait or QR panel.
        keep = [True] * len(raw)
        for first_index, first in enumerate(raw):
            for second_index in range(first_index + 1, len(raw)):
                second = raw[second_index]
                if _intersection_over_smaller(first.box, second.box) < 0.92:
                    continue
                smaller_index, larger_index = (
                    (first_index, second_index) if first.area <= second.area else (second_index, first_index)
                )
                smaller = raw[smaller_index]
                larger = raw[larger_index]
                ratio = smaller.area / larger.area if larger.area > 0 else 0.0
                if 0.45 <= ratio < 0.93:
                    keep[larger_index] = False
                elif ratio < 0.18:
                    keep[smaller_index] = False

        candidates = [candidate for candidate, accepted in zip(raw, keep) if accepted]
        candidates.sort(
            key=lambda item: (item.rectangularity * 0.7 + min(item.area_fraction, 0.7) * 0.3),
            reverse=True,
        )

        detections: list[VisionDetection] = []
        for candidate in candidates[: self.max_candidates]:
            width = float(candidate.box[2] - candidate.box[0])
            height = float(candidate.box[3] - candidate.box[1])
            long_short_ratio = max(width, height) / max(1.0, min(width, height))
            class_name = "aadhaar_front" if long_short_ratio >= 1.45 else "aadhaar_letter"
            geometry_confidence = min(
                0.60,
                0.20 + candidate.rectangularity * 0.25 + min(candidate.area_fraction, 0.75) * 0.15,
            )
            detections.append(
                VisionDetection(
                    class_name=class_name,
                    confidence=float(geometry_confidence),
                    polygon=candidate.polygon,
                    box=candidate.box,
                    source="opencv_proposal",
                )
            )
        return detections
