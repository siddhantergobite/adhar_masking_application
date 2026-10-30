from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from backend.batches import BatchManager, BatchSeed


@unittest.skipUnless(os.getenv("AADHAAR_TEST_DATABASE_URL"), "set AADHAAR_TEST_DATABASE_URL to run PostgreSQL integration tests")
class PostgresBatchQueueIntegrationTests(unittest.TestCase):
    def test_parallel_workers_claim_distinct_rows_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = patch.dict(
                os.environ,
                {
                    "AADHAAR_DATABASE_URL": os.environ["AADHAAR_TEST_DATABASE_URL"],
                    "AADHAAR_BATCH_STORAGE_ROOT": temp_dir,
                    "AADHAAR_S3_INPUT_BUCKET": "test-input",
                    "AADHAAR_S3_INPUT_PREFIX": "test-input",
                    "AADHAAR_S3_TEMP_PREFIX": "test-temporary",
                    "AADHAAR_S3_OUTPUT_BUCKET": "test-output",
                    "AADHAAR_S3_OUTPUT_PREFIX": "test-output",
                    "CLOUDFLARE_R2_BUCKET": "",
                },
                clear=False,
            )
            environment.start()
            managers: list[BatchManager] = []
            try:
                first = BatchManager()
                second = BatchManager()
                managers.extend((first, second))
                batch_id = first.new_id()
                file_ids = (first.new_id(), first.new_id())
                first.prepare_batch_dir(batch_id)
                first.create_batch(
                    batch_id,
                    "integration-test",
                    [
                        BatchSeed(file_ids[0], "synthetic-a.jpg", "s3_temporary"),
                        BatchSeed(file_ids[1], "synthetic-b.jpg", "s3_temporary"),
                    ],
                )
                barrier = threading.Barrier(2)

                def claim(manager: BatchManager):
                    barrier.wait()
                    return manager.claim_next_file()

                with ThreadPoolExecutor(max_workers=2) as pool:
                    claimed = list(pool.map(claim, managers))
                claimed_items = [item for item in claimed if item is not None]
                claimed_ids = {item["id"] for item in claimed_items}
                self.assertEqual(claimed_ids, set(file_ids))

                for item in claimed_items:
                    first.finish_file(
                        item["id"],
                        lease_token=item["lease_token"],
                        status="failed",
                        error_code="integration_test",
                    )
                first.delete_batch(batch_id)
            finally:
                for manager in managers:
                    manager.close()
                environment.stop()


if __name__ == "__main__":
    unittest.main()
