from __future__ import annotations

import unittest

import numpy as np

from backend.domain import OcrLine, OcrWord
from backend.redaction import mask_first_eight
from backend.validation import verhoeff_check_digit


def number_line() -> tuple[str, OcrLine]:
    prefix = "23456789012"
    number = prefix + verhoeff_check_digit(prefix)
    text = f"{number[:4]} {number[4:8]} {number[8:]}"
    words = (
        OcrWord(number[:4], np.array([30, 40, 150, 90], dtype=np.float32)),
        OcrWord(number[4:8], np.array([175, 40, 295, 90], dtype=np.float32)),
        OcrWord(number[8:], np.array([320, 40, 440, 90], dtype=np.float32)),
    )
    polygon = np.array([[25, 35], [445, 35], [445, 95], [25, 95]], dtype=np.float32)
    return number, OcrLine(text=text, confidence=0.99, polygon=polygon, words=words)


class RedactionTests(unittest.TestCase):
    def test_only_first_eight_digit_area_is_masked(self) -> None:
        _, line = number_line()
        image = np.full((140, 500, 3), 240, dtype=np.uint8)
        box = mask_first_eight(image, line)
        masked_region = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        self.assertLess(box[2], 320)
        self.assertLessEqual(box[0], 20)
        self.assertGreater(np.unique(masked_region.reshape(-1, 3), axis=0).shape[0], 1)
        self.assertTrue(np.any(np.all(masked_region < 150, axis=2)))
        self.assertTrue(np.all(image[60, 370] == 240))

    def test_vertical_number_line_masks_first_eight_digits(self) -> None:
        number, _ = number_line()
        words = (
            OcrWord(number[:4], np.array([40, 30, 100, 150], dtype=np.float32)),
            OcrWord(" ", np.array([40, 165, 100, 180], dtype=np.float32)),
            OcrWord(number[4:8], np.array([40, 195, 100, 315], dtype=np.float32)),
            OcrWord(" ", np.array([40, 330, 100, 345], dtype=np.float32)),
            OcrWord(number[8:], np.array([40, 360, 100, 480], dtype=np.float32)),
        )
        line = OcrLine(
            text=f"{number[:4]} {number[4:8]} {number[8:]}",
            confidence=0.99,
            polygon=np.array([[35, 25], [105, 25], [105, 485], [35, 485]], dtype=np.float32),
            words=words,
        )
        image = np.full((520, 150, 3), 240, dtype=np.uint8)
        box = mask_first_eight(image, line)

        self.assertLess(box[3], 360)
        self.assertLessEqual(box[0], 35)
        self.assertGreaterEqual(box[2], 105)


if __name__ == "__main__":
    unittest.main()

