from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .errors import UnsafeDocument


@dataclass(frozen=True)
class RectifiedRegion:
    image: np.ndarray
    source_quad: np.ndarray
    transform: np.ndarray


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def polygon_to_quad(polygon: np.ndarray, image_shape: tuple[int, ...], expansion: float = 1.015) -> np.ndarray:
    height, width = image_shape[:2]
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        raise UnsafeDocument("The Aadhaar boundary did not contain four usable corners.")

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
    perimeter = cv2.arcLength(hull.reshape(-1, 1, 2), True)

    quad: np.ndarray | None = None
    for fraction in (0.006, 0.01, 0.015, 0.022, 0.032, 0.05, 0.075):
        approximation = cv2.approxPolyDP(hull.reshape(-1, 1, 2), fraction * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            quad = approximation.reshape(4, 2).astype(np.float32)
            break

    if quad is None:
        rectangle = cv2.minAreaRect(hull.reshape(-1, 1, 2))
        quad = cv2.boxPoints(rectangle).astype(np.float32)

    quad = order_quad(quad)
    center = quad.mean(axis=0)
    quad = center + (quad - center) * expansion
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)

    area = abs(cv2.contourArea(quad.reshape(-1, 1, 2)))
    if area < height * width * 0.008:
        raise UnsafeDocument("The detected Aadhaar is too small to read safely.")
    return quad.astype(np.float32)


def rectify_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    *,
    min_long_edge: int = 1200,
    max_long_edge: int = 2600,
) -> RectifiedRegion:
    quad = polygon_to_quad(polygon, image.shape)
    top_left, top_right, bottom_right, bottom_left = quad
    measured_width = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )
    measured_height = max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )
    if measured_width < 2 or measured_height < 2:
        raise UnsafeDocument("The detected Aadhaar boundary is degenerate.")

    scale = max(1.0, min_long_edge / max(measured_width, measured_height))
    if max(measured_width, measured_height) * scale > max_long_edge:
        scale = max_long_edge / max(measured_width, measured_height)
    output_width = max(2, int(round(measured_width * scale)))
    output_height = max(2, int(round(measured_height * scale)))
    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad, destination)
    rectified = cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return RectifiedRegion(image=rectified, source_quad=quad, transform=transform)


def rotate_quarter(image: np.ndarray, turns_counterclockwise: int) -> np.ndarray:
    turns = turns_counterclockwise % 4
    if turns == 0:
        return image.copy()
    if turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced = clahe.apply(lightness)
    restored = cv2.cvtColor(cv2.merge((balanced, channel_a, channel_b)), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(restored, (0, 0), 1.0)
    return cv2.addWeighted(restored, 1.35, blurred, -0.35, 0)


def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    longest = max(gray.shape)
    if longest > 1200:
        scale = 1200.0 / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def validate_region_quality(image: np.ndarray, *, min_edge: int, min_blur_score: float) -> float:
    height, width = image.shape[:2]
    if min(height, width) < min_edge:
        raise UnsafeDocument(
            "The Aadhaar is too small in the image to separate and mask eight digits safely."
        )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if float(gray.std()) < 8.0:
        raise UnsafeDocument("The Aadhaar has too little contrast to read safely.")
    score = blur_score(image)
    if score < min_blur_score:
        raise UnsafeDocument("The Aadhaar is too blurred to mask safely.")
    return score


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32).reshape(4)
    second = np.asarray(second, dtype=np.float32).reshape(4)
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def polygon_box(polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    return np.array(
        [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()],
        dtype=np.float32,
    )


def overlap_over_first(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32).reshape(4)
    second = np.asarray(second, dtype=np.float32).reshape(4)
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    return intersection / first_area if first_area > 0 else 0.0

