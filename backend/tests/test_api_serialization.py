from __future__ import annotations

import unittest

from backend.server import serialize_batch


class LegacyScoreManager:
    def get_batch(self, _batch_id: str):
        return {
            "batch": {
                "source_type": "upload",
                "status": "completed",
                "created_at": "2026-10-29T00:00:00+00:00",
                "updated_at": "2026-10-29T00:00:01+00:00",
                "total_files": 1,
                "completed_files": 1,
                "rejected_files": 0,
                "failed_files": 0,
            },
            "files": [{
                "id": "a" * 32,
                "display_name": "synthetic.jpg",
                "status": "completed",
                "error_code": None,
                "error_message": None,
                "document_count": 1,
                "document_classes": "[]",
                "masked_regions": 1,
                "output_path": None,
                "output_s3_uri": None,
                # Legacy database columns may exist after upgrading.
                "masking_confidence": 92,
                "masking_confidence_source": "yolo",
            }],
        }


class BatchResponseContractTests(unittest.TestCase):
    def test_batch_response_does_not_expose_legacy_scores(self) -> None:
        response = serialize_batch(LegacyScoreManager(), "b" * 32)
        file_record = response["files"][0]

        self.assertNotIn("yolo_confidence", file_record)
        self.assertNotIn("masking_confidence", file_record)
        self.assertNotIn("confidence", file_record)


if __name__ == "__main__":
    unittest.main()
