from __future__ import annotations

import unittest

import numpy as np

from backend.domain import OcrLine
from backend.validation import (
    extract_aadhaar_digits,
    verhoeff_check_digit,
    verhoeff_is_valid,
    verify_aadhaar,
)


class VerhoeffTests(unittest.TestCase):
    def setUp(self) -> None:
        prefix = "23456789012"
        self.valid_number = prefix + verhoeff_check_digit(prefix)

    def test_generated_number_validates(self) -> None:
        self.assertTrue(verhoeff_is_valid(self.valid_number))
        self.assertEqual(extract_aadhaar_digits(" ".join(
            (self.valid_number[:4], self.valid_number[4:8], self.valid_number[8:])
        )), self.valid_number)

    def test_invalid_checksum_is_rejected(self) -> None:
        replacement = "0" if self.valid_number[-1] != "0" else "1"
        invalid = self.valid_number[:-1] + replacement
        self.assertFalse(verhoeff_is_valid(invalid))
        self.assertIsNone(extract_aadhaar_digits(invalid))

    def test_payment_card_length_and_letters_are_rejected(self) -> None:
        self.assertIsNone(extract_aadhaar_digits("2222 3333 4444 5555"))
        self.assertIsNone(extract_aadhaar_digits("A234 5678 9012"))

    def test_zero_or_one_prefix_is_rejected(self) -> None:
        for first in ("0", "1"):
            prefix = first + "3456789012"
            candidate = prefix + verhoeff_check_digit(prefix)
            self.assertIsNone(extract_aadhaar_digits(candidate))

    def test_opencv_fallback_requires_checksum_identity_and_layout(self) -> None:
        lines = [
            OcrLine(
                text="Government of India",
                confidence=0.98,
                polygon=np.array([[0, 0], [100, 0], [100, 20], [0, 20]], dtype=np.float32),
            ),
            OcrLine(
                text="MALE",
                confidence=0.98,
                polygon=np.array([[0, 30], [50, 30], [50, 50], [0, 50]], dtype=np.float32),
            ),
        ]
        accepted = verify_aadhaar(
            lines,
            detector_confidence=0.50,
            has_qr=False,
            valid_number_count=1,
            threshold=0.68,
            detection_source="opencv_proposal",
            number_detector_supported=False,
        )
        rejected_without_identity = verify_aadhaar(
            [],
            detector_confidence=0.60,
            has_qr=True,
            valid_number_count=1,
            threshold=0.68,
            detection_source="opencv_proposal",
            number_detector_supported=True,
        )
        rejected_without_number = verify_aadhaar(
            lines,
            detector_confidence=0.60,
            has_qr=True,
            valid_number_count=0,
            threshold=0.68,
            detection_source="opencv_proposal",
            number_detector_supported=True,
        )

        self.assertTrue(accepted.accepted)
        self.assertFalse(rejected_without_identity.accepted)
        self.assertFalse(rejected_without_number.accepted)


if __name__ == "__main__":
    unittest.main()

