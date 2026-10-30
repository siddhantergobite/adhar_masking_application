from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.batches import BatchManager, _PostgresConnectionAdapter


class FakeConnection:
    def __init__(self) -> None:
        self.query = ""
        self.parameters = ()

    def execute(self, query: str, parameters: tuple = ()):
        self.query = query
        self.parameters = parameters
        return self


class SharedStorageTests(unittest.TestCase):
    def test_postgres_adapter_translates_parameters(self) -> None:
        raw = FakeConnection()
        _PostgresConnectionAdapter(raw).execute(
            "UPDATE batch_files SET status=? WHERE id=?",
            ("queued", "abc"),
        )
        self.assertEqual(raw.query, "UPDATE batch_files SET status=%s WHERE id=%s")
        self.assertEqual(raw.parameters, ("queued", "abc"))

    def test_direct_upload_uses_temporary_shared_prefix_in_postgres_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {"AADHAAR_BATCH_STORAGE_ROOT": temp_dir, "CLOUDFLARE_R2_BUCKET": ""},
                clear=False,
            ):
                manager = BatchManager()
            manager.use_postgres = True
            manager.shared_input_bucket = "private-bucket"
            manager.shared_temp_prefix = "aadhaar/queued-input"
            source = Path(temp_dir) / "source.jpg"
            source.write_bytes(b"synthetic image")
            with patch.object(BatchManager, "upload_s3_object") as upload:
                stored = manager.store_shared_input(
                    "a" * 32,
                    "b" * 32,
                    ".jpg",
                    source,
                    "image/jpeg",
                )

        self.assertEqual(stored["source_type"], "s3_temporary")
        self.assertEqual(stored["source_bucket"], "private-bucket")
        self.assertTrue(stored["source_key"].startswith("aadhaar/queued-input/"))
        upload.assert_called_once_with(
            "private-bucket",
            stored["source_key"],
            source,
            "image/jpeg",
        )

    def test_materialize_s3_output_to_local_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "masked.pdf"

            class FakeS3:
                def download_file(self, bucket: str, key: str, target: str) -> None:
                    self.received = (bucket, key)
                    Path(target).write_bytes(b"synthetic masked pdf")

            client = FakeS3()
            with patch.object(BatchManager, "_s3_client", return_value=client):
                BatchManager.materialize_output("s3://private-bucket/masked/job.pdf", destination)

            self.assertEqual(client.received, ("private-bucket", "masked/job.pdf"))
            self.assertEqual(destination.read_bytes(), b"synthetic masked pdf")


if __name__ == "__main__":
    unittest.main()
