from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.ocr import read_ocr_lines


class RecordingOcrEngine:
    def __init__(self) -> None:
        self.input_shape: tuple[int, ...] | None = None

    def predict(self, *, input: np.ndarray) -> list[SimpleNamespace]:
        self.input_shape = input.shape
        payload = {
            "res": {
                "rec_texts": ["12345678"],
                "rec_scores": [0.99],
                "rec_polys": [
                    np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
                ],
                "text_word": [["1234", "5678"]],
                "text_word_boxes": [[
                    np.array([[10, 10], [25, 10], [25, 30], [10, 30]], dtype=np.float32),
                    np.array([[35, 10], [50, 10], [50, 30], [35, 30]], dtype=np.float32),
                ]],
            }
        }
        return [SimpleNamespace(json=payload)]


class OcrPreprocessingTests(unittest.TestCase):
    @patch("backend.ocr.enhance_for_ocr")
    def test_resizes_before_enhancement_and_maps_boxes_to_source(self, enhance) -> None:
        enhance.side_effect = lambda image: image
        engine = RecordingOcrEngine()
        image = np.zeros((1200, 600, 3), dtype=np.uint8)

        lines = read_ocr_lines(
            engine,
            image,
            max_long_edge=600,
            enhance_after_resize=True,
        )

        self.assertEqual(enhance.call_args.args[0].shape, (600, 300, 3))
        self.assertEqual(engine.input_shape, (600, 300, 3))
        np.testing.assert_array_equal(
            lines[0].polygon,
            np.array([[20, 20], [100, 20], [100, 60], [20, 60]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            lines[0].words[0].box,
            np.array([20, 20, 50, 60], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
