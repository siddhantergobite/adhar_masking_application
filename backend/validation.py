from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np

from .domain import OcrLine, Verification


_D_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_P_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

_INVERSE = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)
_SEPARATORS = frozenset(" \t\r\n-\u2010\u2011\u2012\u2013\u2014")


def verhoeff_is_valid(number: str) -> bool:
    if not number.isascii() or not number.isdigit():
        return False
    checksum = 0
    for index, character in enumerate(reversed(number)):
        checksum = _D_TABLE[checksum][_P_TABLE[index % 8][int(character)]]
    return checksum == 0


def verhoeff_check_digit(prefix: str) -> str:
    if not prefix.isascii() or not prefix.isdigit():
        raise ValueError("prefix must contain ASCII digits only")
    checksum = 0
    for index, character in enumerate(reversed(prefix)):
        checksum = _D_TABLE[checksum][_P_TABLE[(index + 1) % 8][int(character)]]
    return str(_INVERSE[checksum])


def extract_aadhaar_digits(text: str) -> str | None:
    digits: list[str] = []
    for character in text:
        try:
            digits.append(str(unicodedata.decimal(character)))
        except (TypeError, ValueError):
            if character not in _SEPARATORS:
                return None
    candidate = "".join(digits)
    if len(candidate) != 12 or candidate[0] in {"0", "1"}:
        return None
    return candidate if verhoeff_is_valid(candidate) else None


def _normalise_text(lines: list[OcrLine]) -> str:
    text = " ".join(line.text for line in lines).casefold()
    text = re.sub(r"\s+", " ", text)
    # OCR regularly confuses the first character in the Ashoka-emblem
    # heading, or swaps visually similar letters in Aadhaar.  These are
    # conservative corrections for phrases that are only useful as identity
    # anchors when they occur alongside a checksum-valid number.
    for misspelling in (
        "covernment",
        "goverment",
        "governmant",
        "govemment",
        "governrnent",
    ):
        text = text.replace(misspelling, "government")
    text = text.replace("government cf india", "government of india")
    text = text.replace("aaphaar", "aadhaar")
    return text


def qr_present(image_rgb: np.ndarray) -> bool:
    detector = cv2.QRCodeDetector()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    try:
        if hasattr(detector, "detectMulti"):
            detected, points = detector.detectMulti(image_bgr)
            if detected and points is not None and len(points):
                return True
        detected, points = detector.detect(image_bgr)
        return bool(detected and points is not None)
    except cv2.error:
        return False


def verify_aadhaar(
    lines: list[OcrLine],
    *,
    detector_confidence: float,
    has_qr: bool,
    valid_number_count: int,
    threshold: float,
    detection_source: str = "yolo",
    number_detector_supported: bool = True,
) -> Verification:
    text = _normalise_text(lines)
    strong_phrases = {
        "aadhaar": 0.20,
        "aadhar": 0.17,
        "uidai": 0.20,
        "unique identification authority": 0.20,
        "government of india": 0.16,
        "भारत सरकार": 0.16,
        "आधार": 0.20,
    }
    secondary_phrases = {
        "year of birth",
        "date of birth",
        "dob",
        "yob",
        "male",
        "female",
        "address",
        "help@uidai",
        "www.uidai",
        "mera aadhaar",
    }

    signals: list[str] = []
    if detection_source == "yolo":
        signals.append("aadhaar-segmentation-model")
    else:
        signals.append("quadrilateral-candidate")
    strong_weight = 0.0
    strong_hits = 0
    for phrase, weight in strong_phrases.items():
        if phrase in text:
            strong_weight = max(strong_weight, weight)
            strong_hits += 1
            signals.append(f"text:{phrase}")
    secondary_hits = [phrase for phrase in secondary_phrases if phrase in text]
    if secondary_hits:
        signals.append("aadhaar-layout-text")
    if has_qr:
        signals.append("qr-pattern")
    if valid_number_count:
        signals.append("valid-verhoeff-number")
    if number_detector_supported:
        signals.append("number-detector-overlap")

    if detection_source == "yolo":
        score = detector_confidence * 0.58
        score += strong_weight
        score += min(len(secondary_hits) * 0.025, 0.10)
        score += 0.10 if has_qr else 0.0
        score += 0.12 if valid_number_count else 0.0
        score = min(score, 1.0)
        has_identity_anchor = bool(strong_weight or has_qr)
        exceptional_model_and_number = detector_confidence >= 0.90 and valid_number_count > 0
        accepted = score >= threshold and (has_identity_anchor or exceptional_model_and_number)
    else:
        # OpenCV proposes geometry only; it receives no Aadhaar identity
        # credit. Acceptance requires mutually independent content signals.
        score = 0.08
        # A checksum-valid 12-digit value is a strong independent signal. The
        # first-digit rule and mandatory Aadhaar identity/layout anchors below
        # still prevent a generic numeric card from passing this fallback.
        score += 0.34 if valid_number_count else 0.0
        score += min(strong_weight * 1.25, 0.25)
        score += min(strong_hits * 0.04, 0.12)
        score += min(len(secondary_hits) * 0.03, 0.12)
        score += 0.15 if has_qr else 0.0
        score += 0.08 if number_detector_supported else 0.0
        score = min(score, 1.0)
        corroborated_layout = has_qr or bool(secondary_hits) or strong_hits >= 2
        accepted = (
            score >= threshold
            and valid_number_count > 0
            and strong_hits > 0
            and corroborated_layout
        )
    return Verification(accepted=accepted, confidence=score, signals=tuple(signals))

