from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from .domain import OcrLine
from .errors import UnsafeDocument
from .validation import extract_aadhaar_digits


_SEPARATOR_WIDTH = 0.35
_MASK_LABEL = "XXXX XXXX"


def _token_digit_boxes(
    text: str,
    box: np.ndarray,
    *,
    vertical: bool = False,
) -> list[tuple[str, np.ndarray]]:
    left, top, right, bottom = np.asarray(box, dtype=np.float32).reshape(4)
    units: list[tuple[str | None, float]] = []
    for character in text:
        try:
            digit = str(unicodedata.decimal(character))
            units.append((digit, 1.0))
        except (TypeError, ValueError):
            if character.isspace() or character in "-\u2010\u2011\u2012\u2013\u2014":
                units.append((None, _SEPARATOR_WIDTH))
            else:
                raise UnsafeDocument(
                    "OCR mixed letters with the Aadhaar number, so the first eight digits cannot be isolated safely."
                )
    total_units = sum(width for _, width in units)
    if total_units <= 0:
        return []
    if vertical:
        pixels_per_unit = (bottom - top) / total_units
        cursor = top
    else:
        pixels_per_unit = (right - left) / total_units
        cursor = left
    digit_boxes: list[tuple[str, np.ndarray]] = []
    for digit, width in units:
        next_cursor = cursor + width * pixels_per_unit
        if digit is not None:
            if vertical:
                digit_boxes.append(
                    (digit, np.array([left, cursor, right, next_cursor], dtype=np.float32))
                )
            else:
                digit_boxes.append(
                    (digit, np.array([cursor, top, next_cursor, bottom], dtype=np.float32))
                )
        cursor = next_cursor
    return digit_boxes


def digit_boxes_for_line(line: OcrLine) -> tuple[str, list[np.ndarray]]:
    if not line.words:
        raise UnsafeDocument(
            "OCR did not return word-level positions for the Aadhaar number. No approximate mask was applied."
        )
    digits_with_boxes: list[tuple[str, np.ndarray]] = []
    line_points = np.asarray(line.polygon, dtype=np.float32).reshape(-1, 2)
    line_width = float(line_points[:, 0].max() - line_points[:, 0].min())
    line_height = float(line_points[:, 1].max() - line_points[:, 1].min())
    vertical = line_height > line_width * 1.25
    for word in line.words:
        digits_with_boxes.extend(_token_digit_boxes(word.text, word.box, vertical=vertical))
    digits = "".join(digit for digit, _ in digits_with_boxes)
    boxes = [box for _, box in digits_with_boxes]
    validated = extract_aadhaar_digits(line.text)
    if validated is None or digits != validated or len(boxes) != 12:
        raise UnsafeDocument(
            "The OCR text and its digit positions disagree, so masking was stopped before creating an unsafe copy."
        )
    return validated, boxes


def _sample_mask_background(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int]:
    """Choose a nearby opaque fill so the replacement looks like document text."""

    height, width = image.shape[:2]
    samples: list[np.ndarray] = []
    band = max(2, min(6, (y2 - y1) // 8))
    if y1 > 0:
        samples.append(image[max(0, y1 - band):y1, x1:x2])
    if y2 < height:
        samples.append(image[y2:min(height, y2 + band), x1:x2])
    if x1 > 0:
        samples.append(image[y1:y2, max(0, x1 - band):x1])
    if x2 < width:
        samples.append(image[y1:y2, x2:min(width, x2 + band)])

    usable = [sample.reshape(-1, image.shape[2]) for sample in samples if sample.size]
    if not usable:
        return (245, 245, 245)
    colour = np.median(np.concatenate(usable, axis=0), axis=0)
    return tuple(int(np.clip(channel, 0, 255)) for channel in colour[:3])


def _draw_mask_label(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    background: tuple[int, int, int],
    *,
    vertical: bool = False,
) -> None:
    """Cover the source digits, then render the user-facing masked value."""

    cv2.rectangle(image, (x1, y1), (x2, y2), color=background, thickness=cv2.FILLED)

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    padding_x = max(3, int(round(box_height * 0.10)))
    padding_y = max(2, int(round(box_height * 0.12)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(box_height * 0.075)))
    font_scale = max(0.45, box_height / 30.0)

    if vertical:
        # OCR can return a valid number from a 90-degree text line when a
        # perspective polygon has been ordered the wrong way. Render the
        # replacement label horizontally first, rotate it into the same line
        # direction, and rotate the full document back before returning it.
        padding_x = max(3, int(round(box_width * 0.10)))
        padding_y = max(2, int(round(box_width * 0.12)))
        thickness = max(1, int(round(box_width * 0.075)))
        font_scale = max(0.45, box_width / 30.0)
        available_width = max(1, box_height - 2 * padding_y)
        available_height = max(1, box_width - 2 * padding_x)
        for _ in range(3):
            (text_width, text_height), baseline = cv2.getTextSize(
                _MASK_LABEL,
                font,
                font_scale,
                thickness,
            )
            width_ratio = available_width / max(1, text_width)
            height_ratio = available_height / max(1, text_height + baseline)
            ratio = min(1.0, width_ratio, height_ratio)
            if ratio >= 0.99:
                break
            font_scale *= ratio

        (text_width, text_height), baseline = cv2.getTextSize(
            _MASK_LABEL,
            font,
            font_scale,
            thickness,
        )
        canvas_height = max(1, text_height + baseline + 2 * padding_y)
        canvas_width = max(1, text_width + 2 * padding_x)
        label = np.full((canvas_height, canvas_width, 3), background, dtype=image.dtype)
        background_luma = (
            0.299 * background[0]
            + 0.587 * background[1]
            + 0.114 * background[2]
        )
        text_colour = (38, 66, 48) if background_luma >= 145 else (245, 245, 245)
        cv2.putText(
            label,
            _MASK_LABEL,
            (padding_x, padding_y + text_height),
            font,
            font_scale,
            text_colour,
            thickness,
            lineType=cv2.LINE_AA,
        )
        label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
        label_height, label_width = label.shape[:2]
        paste_x = x1 + max(0, (box_width - label_width) // 2)
        paste_y = y1 + max(0, (box_height - label_height) // 2)
        paste_x2 = min(x2, paste_x + label_width)
        paste_y2 = min(y2, paste_y + label_height)
        if paste_x2 > paste_x and paste_y2 > paste_y:
            image[paste_y:paste_y2, paste_x:paste_x2] = label[
                : paste_y2 - paste_y,
                : paste_x2 - paste_x,
            ]
        return

    for _ in range(3):
        (text_width, text_height), baseline = cv2.getTextSize(
            _MASK_LABEL,
            font,
            font_scale,
            thickness,
        )
        available_width = max(1, box_width - 2 * padding_x)
        available_height = max(1, box_height - 2 * padding_y)
        width_ratio = available_width / max(1, text_width)
        height_ratio = available_height / max(1, text_height + baseline)
        ratio = min(1.0, width_ratio, height_ratio)
        if ratio >= 0.99:
            break
        font_scale *= ratio

    (text_width, text_height), baseline = cv2.getTextSize(
        _MASK_LABEL,
        font,
        font_scale,
        thickness,
    )
    text_x = x1 + max(0, (box_width - text_width) // 2)
    text_y = y1 + max(text_height, (box_height + text_height - baseline) // 2)
    background_luma = (
        0.299 * background[0]
        + 0.587 * background[1]
        + 0.114 * background[2]
    )
    text_colour = (38, 66, 48) if background_luma >= 145 else (245, 245, 245)
    cv2.putText(
        image,
        _MASK_LABEL,
        (text_x, text_y),
        font,
        font_scale,
        text_colour,
        thickness,
        lineType=cv2.LINE_AA,
    )


def mask_first_eight(image: np.ndarray, line: OcrLine) -> np.ndarray:
    _, boxes = digit_boxes_for_line(line)
    first_eight = boxes[:8]
    ninth = boxes[8]
    line_points = np.asarray(line.polygon, dtype=np.float32).reshape(-1, 2)
    line_width = float(line_points[:, 0].max() - line_points[:, 0].min())
    line_height = float(line_points[:, 1].max() - line_points[:, 1].min())
    vertical = line_height > line_width * 1.25
    digit_height = max(float(box[3] - box[1]) for box in boxes)
    digit_width = max(float(box[2] - box[0]) for box in boxes)
    padding_x = max(3.0, (digit_width if vertical else line_height) * 0.16)
    padding_y = max(2.0, (digit_height if vertical else line_height) * 0.16)

    if vertical:
        left = min(
            min(float(box[0]) for box in first_eight),
            float(line_points[:, 0].min()),
        ) - padding_x
        right = max(
            max(float(box[2]) for box in first_eight),
            float(line_points[:, 0].max()),
        ) + padding_x
        top = min(
            min(float(box[1]) for box in first_eight),
            float(line_points[:, 1].min()),
        ) - padding_y
        eighth_bottom = max(float(box[3]) for box in first_eight)
        ninth_top = min(float(box[1]) for box in boxes[8:])
        gap = max(0.0, ninth_top - eighth_bottom)
        bottom = eighth_bottom + gap * 0.35
    else:
        detected_left = min(
            min(float(box[0]) for box in first_eight),
            float(line_points[:, 0].min()),
        )
        left = detected_left - padding_x
        top = min(float(box[1]) for box in boxes) - padding_y
        eighth_right = float(first_eight[-1][2])
        ninth_left = float(ninth[0])
        eighth_center = (float(first_eight[-1][0]) + eighth_right) / 2
        ninth_center = (ninth_left + float(ninth[2])) / 2
        if ninth_left >= eighth_right:
            right = eighth_right + (ninth_left - eighth_right) * 0.35
        else:
            right = (eighth_center + ninth_center) / 2
        bottom = max(float(box[3]) for box in boxes) + padding_y

    height, width = image.shape[:2]
    x1 = int(np.clip(np.floor(left), 0, width - 1))
    y1 = int(np.clip(np.floor(top), 0, height - 1))
    x2 = int(np.clip(np.ceil(right), x1 + 1, width))
    y2 = int(np.clip(np.ceil(bottom), y1 + 1, height))
    if not vertical and x2 >= int(ninth_center):
        x2 = max(x1 + 1, int(np.floor(ninth_center - 1)))
    if x2 <= x1 or y2 <= y1:
        raise UnsafeDocument("The first eight digit coordinates were not geometrically safe to mask.")

    background = _sample_mask_background(image, x1, y1, x2, y2)
    _draw_mask_label(image, x1, y1, x2, y2, background, vertical=vertical)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

