from __future__ import annotations

import unittest

import numpy as np

from backend.ai_summary import (
    CardSummaryRequest,
    _local_summary,
    confidence_score,
    pipeline_confidence_score,
)
from backend.domain import ProcessedDocument, ProcessingResult


def document(
    *,
    boundary: float,
    scan: float,
    ocr: float,
    verification: float,
    localization: float | None,
) -> ProcessedDocument:
    return ProcessedDocument(
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        document_class="aadhaar_front",
        document_confidence=boundary,
        verification_confidence=verification,
        masked_regions=1,
        boundary_confidence=boundary,
        scan_quality_score=scan,
        ocr_confidence=ocr,
        number_detector_confidence=localization,
    )


class PipelineConfidenceTests(unittest.TestCase):
    def test_missing_optional_number_detector_match_does_not_zero_score(self) -> None:
        result = ProcessingResult((document(
            boundary=0.758,
            scan=1.0,
            ocr=0.994,
            verification=0.770,
            localization=None,
        ),))

        self.assertEqual(pipeline_confidence_score(result), 82)

    def test_low_number_detector_support_contributes_but_does_not_dominate(self) -> None:
        result = ProcessingResult((document(
            boundary=0.635,
            scan=1.0,
            ocr=0.995,
            verification=0.698,
            localization=0.427,
        ),))

        self.assertEqual(pipeline_confidence_score(result), 76)

    def test_file_score_uses_weakest_document(self) -> None:
        strong = document(
            boundary=0.95, scan=1.0, ocr=0.99, verification=0.95, localization=0.9
        )
        weaker = document(
            boundary=0.60, scan=0.8, ocr=0.90, verification=0.70, localization=None
        )

        self.assertEqual(
            pipeline_confidence_score(ProcessingResult((strong, weaker))),
            pipeline_confidence_score(ProcessingResult((weaker,))),
        )

    def test_empty_result_and_rejection_are_not_scored(self) -> None:
        self.assertIsNone(pipeline_confidence_score(ProcessingResult(())))
        self.assertIsNone(confidence_score(CardSummaryRequest(outcome="rejected")))
        self.assertIsNone(confidence_score(CardSummaryRequest(outcome="unavailable")))
        rejected_summary = _local_summary(CardSummaryRequest(outcome="rejected"), None)
        self.assertIn("No confidence score", rejected_summary)
        self.assertNotIn("0/100", rejected_summary)


if __name__ == "__main__":
    unittest.main()
