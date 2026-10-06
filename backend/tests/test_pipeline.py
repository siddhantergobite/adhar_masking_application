from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.domain import NumberDetection, OcrLine, OcrWord, VisionDetection
from backend.errors import NoAadhaarDetected
from backend.pipeline import AadhaarPipeline
from backend.settings import Settings
from backend.validation import verhoeff_check_digit


class EmptyDocumentSegmenter:
    def detect(self, _image):
        return []


class OneDocumentSegmenter:
    def detect(self, image):
        height, width = image.shape[:2]
        polygon = np.array(
            [[10, 10], [width - 11, 10], [width - 11, height - 11], [10, height - 11]],
            dtype=np.float32,
        )
        return [
            VisionDetection(
                class_name="aadhaar_front",
                confidence=0.96,
                polygon=polygon,
                box=np.array([10, 10, width - 11, height - 11], dtype=np.float32),
            )
        ]


class OneNumberDetector:
    def detect(self, _image):
        return [NumberDetection(0.95, np.array([175, 450, 1030, 620], dtype=np.float32))]


class EmptyNumberDetector:
    def detect(self, _image):
        return []


class EmptyDocumentProposer:
    def detect(self, _image):
        return []


def settings() -> Settings:
    return Settings(
        document_model_path=Path("document.pt"),
        number_model_path=Path("number.pt"),
        yolo_device="cpu",
        document_confidence=0.72,
        number_confidence=0.62,
        document_iou=0.45,
        inference_size=1280,
        ocr_confidence=0.60,
        verification_confidence=0.68,
        max_upload_bytes=1024,
        max_pages=1,
        max_page_pixels=1_000_000,
        pdf_scale=2.5,
        min_region_edge=100,
        min_blur_score=0.0,
    )


def ocr_lines() -> list[OcrLine]:
    prefix = "23456789012"
    number = prefix + verhoeff_check_digit(prefix)
    number_text = f"{number[:4]} {number[4:8]} {number[8:]}"
    number_polygon = np.array([[190, 490], [1010, 490], [1010, 570], [190, 570]], dtype=np.float32)
    words = (
        OcrWord(number[:4], np.array([200, 500, 420, 560], dtype=np.float32)),
        OcrWord(number[4:8], np.array([450, 500, 670, 560], dtype=np.float32)),
        OcrWord(number[8:], np.array([710, 500, 930, 560], dtype=np.float32)),
    )
    return [
        OcrLine(
            text="Government of India Aadhaar",
            confidence=0.98,
            polygon=np.array([[180, 100], [800, 100], [800, 150], [180, 150]], dtype=np.float32),
        ),
        OcrLine(text=number_text, confidence=0.99, polygon=number_polygon, words=words),
    ]


def ocr_lines_with_repeat() -> list[OcrLine]:
    lines = ocr_lines()
    original = lines[1]
    shift = np.array([0, 180, 0, 180], dtype=np.float32)
    repeated_words = tuple(
        OcrWord(word.text, word.box + shift) for word in original.words
    )
    repeated = OcrLine(
        text=original.text,
        confidence=original.confidence,
        polygon=original.polygon + np.array([[0, 180], [0, 180], [0, 180], [0, 180]], dtype=np.float32),
        words=repeated_words,
    )
    return [lines[0], original, repeated]


def ocr_lines_with_split_number() -> list[OcrLine]:
    lines = ocr_lines()
    original = lines[1]
    fragments = [
        OcrLine(
            text=word.text,
            confidence=0.99,
            polygon=np.array(
                [
                    [word.box[0], word.box[1]],
                    [word.box[2], word.box[1]],
                    [word.box[2], word.box[3]],
                    [word.box[0], word.box[3]],
                ],
                dtype=np.float32,
            ),
            words=(word,),
        )
        for word in original.words
    ]
    return [lines[0], *fragments]


class PipelineTests(unittest.TestCase):
    def test_yolo_identity_uses_overlapping_opencv_outer_boundary(self) -> None:
        yolo = VisionDetection(
            class_name="aadhaar_letter",
            confidence=0.94,
            polygon=np.array([[0, 50], [650, 50], [650, 700], [0, 700]], dtype=np.float32),
            box=np.array([0, 50, 650, 700], dtype=np.float32),
        )
        proposal = VisionDetection(
            class_name="aadhaar_letter",
            confidence=0.50,
            polygon=np.array([[35, 90], [625, 90], [625, 750], [35, 750]], dtype=np.float32),
            box=np.array([35, 90, 625, 750], dtype=np.float32),
            source="opencv_proposal",
        )

        refined = AadhaarPipeline._refine_document_boundaries([yolo], [proposal])

        np.testing.assert_array_equal(refined[0].box, proposal.box)
        self.assertEqual(refined[0].confidence, yolo.confidence)
        self.assertEqual(refined[0].source, "yolo")

    def test_debit_only_scene_returns_no_aadhaar(self) -> None:
        pipeline = AadhaarPipeline(EmptyDocumentSegmenter(), OneNumberDetector(), object(), settings())
        image = np.random.default_rng(7).integers(0, 255, (500, 800, 3), dtype=np.uint8)
        with self.assertRaises(NoAadhaarDetected):
            pipeline.process([image])

    @patch("backend.pipeline.qr_present", return_value=False)
    @patch("backend.pipeline.read_ocr_lines")
    def test_verified_aadhaar_is_cropped_and_masked(self, mocked_ocr, _mocked_qr) -> None:
        mocked_ocr.side_effect = [ocr_lines(), []]
        pipeline = AadhaarPipeline(OneDocumentSegmenter(), OneNumberDetector(), object(), settings())
        image = np.random.default_rng(11).integers(60, 235, (500, 800, 3), dtype=np.uint8)
        result = pipeline.process([image])
        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.masked_regions, 1)
        output = result.documents[0].image
        self.assertFalse(np.all(output[530, 300] == np.array([24, 36, 29])))
        self.assertFalse(np.all(output[530, 820] == np.array([24, 36, 29])))

    @patch("backend.pipeline.qr_present", return_value=False)
    @patch("backend.pipeline.read_ocr_lines")
    def test_verified_yolo_aadhaar_uses_exact_ocr_boxes_when_number_detector_misses(
        self, mocked_ocr, _mocked_qr
    ) -> None:
        mocked_ocr.side_effect = [ocr_lines(), []]
        pipeline = AadhaarPipeline(OneDocumentSegmenter(), EmptyNumberDetector(), object(), settings())
        image = np.random.default_rng(13).integers(60, 235, (500, 800, 3), dtype=np.uint8)

        result = pipeline.process([image])

        self.assertEqual(result.masked_regions, 1)
        self.assertFalse(np.all(result.documents[0].image[530, 300] == np.array([24, 36, 29])))

    @patch("backend.pipeline.qr_present", return_value=False)
    @patch("backend.pipeline.read_ocr_lines")
    def test_full_frame_fallback_recovers_when_document_detectors_miss(
        self, mocked_ocr, _mocked_qr
    ) -> None:
        mocked_ocr.side_effect = [ocr_lines(), []]
        pipeline = AadhaarPipeline(
            EmptyDocumentSegmenter(),
            EmptyNumberDetector(),
            object(),
            settings(),
            document_proposer=EmptyDocumentProposer(),
        )
        image = np.random.default_rng(17).integers(60, 235, (500, 800, 3), dtype=np.uint8)

        result = pipeline.process([image])

        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.masked_regions, 1)

    @patch("backend.pipeline.qr_present", return_value=False)
    @patch("backend.pipeline.read_ocr_lines")
    def test_broader_candidate_covers_repeated_number_occurrences(
        self, mocked_ocr, _mocked_qr
    ) -> None:
        mocked_ocr.side_effect = [ocr_lines(), [], ocr_lines_with_repeat(), []]
        pipeline = AadhaarPipeline(
            OneDocumentSegmenter(),
            OneNumberDetector(),
            object(),
            settings(),
            document_proposer=EmptyDocumentProposer(),
        )
        image = np.random.default_rng(19).integers(60, 235, (500, 800, 3), dtype=np.uint8)

        result = pipeline.process([image])

        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.masked_regions, 2)

    @patch("backend.pipeline.qr_present", return_value=False)
    @patch("backend.pipeline.read_ocr_lines")
    def test_split_number_fragments_are_merged_using_word_boxes(
        self, mocked_ocr, _mocked_qr
    ) -> None:
        mocked_ocr.side_effect = [ocr_lines_with_split_number(), []]
        pipeline = AadhaarPipeline(OneDocumentSegmenter(), OneNumberDetector(), object(), settings())
        image = np.random.default_rng(23).integers(60, 235, (500, 800, 3), dtype=np.uint8)

        result = pipeline.process([image])

        self.assertEqual(result.masked_regions, 1)


if __name__ == "__main__":
    unittest.main()

