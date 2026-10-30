from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from backend.batches import BatchManager, BatchQueueFull, BatchSeed


class DurableBatchClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.environment = patch.dict(
            os.environ,
            {
                "AADHAAR_BATCH_STORAGE_ROOT": self.temp_dir.name,
                "AADHAAR_BATCH_LEASE_SECONDS": "30",
                "AADHAAR_BATCH_MAX_ATTEMPTS": "2",
            },
        )
        self.environment.start()
        self.manager = BatchManager()
        self.batch_id = self.manager.new_id()
        self.file_id = self.manager.new_id()
        self.manager.prepare_batch_dir(self.batch_id)
        self.manager.create_batch(
            self.batch_id,
            "upload",
            [BatchSeed(self.file_id, "synthetic.jpg", "upload")],
        )

    def tearDown(self) -> None:
        self.environment.stop()
        self.temp_dir.cleanup()

    def test_two_managers_cannot_claim_the_same_file(self) -> None:
        second_manager = BatchManager()
        barrier = threading.Barrier(2)

        def claim(manager: BatchManager):
            barrier.wait()
            return manager.claim_next_file()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(claim, (self.manager, second_manager)))

        claimed = [item for item in results if item is not None]
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0]["id"], self.file_id)

    def test_expired_worker_lease_returns_file_to_queue(self) -> None:
        claimed = self.manager.claim_next_file()
        self.assertIsNotNone(claimed)
        with self.manager._connection() as connection:
            connection.execute(
                "UPDATE batch_files SET updated_at=? WHERE id=?",
                ("2000-01-01T00:00:00+00:00", self.file_id),
            )

        self.assertEqual(self.manager.requeue_stale_processing(), 1)
        reclaimed = self.manager.claim_next_file()
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed["id"], self.file_id)

    def test_stale_worker_cannot_heartbeat_or_finish_after_reclaim(self) -> None:
        stale_claim = self.manager.claim_next_file()
        self.assertIsNotNone(stale_claim)
        with self.manager._connection() as connection:
            connection.execute(
                "UPDATE batch_files SET updated_at=? WHERE id=?",
                ("2000-01-01T00:00:00+00:00", self.file_id),
            )
        self.assertEqual(self.manager.requeue_stale_processing(), 1)
        current_claim = self.manager.claim_next_file()
        self.assertIsNotNone(current_claim)
        self.assertNotEqual(stale_claim["lease_token"], current_claim["lease_token"])

        self.assertFalse(self.manager.touch_processing_file(self.file_id, stale_claim["lease_token"]))
        self.assertFalse(self.manager.finish_file(
            self.file_id,
            lease_token=stale_claim["lease_token"],
            status="failed",
            error_code="stale_worker",
        ))
        self.assertEqual(self.manager.get_file(self.file_id)["status"], "processing")
        self.assertTrue(self.manager.finish_file(
            self.file_id,
            lease_token=current_claim["lease_token"],
            status="failed",
            error_code="current_worker",
        ))
        self.assertEqual(self.manager.get_file(self.file_id)["error_code"], "current_worker")

    def test_repeated_worker_crash_fails_job_after_retry_limit(self) -> None:
        self.assertEqual(self.manager.claim_next_file()["attempt_count"], 1)
        with self.manager._connection() as connection:
            connection.execute(
                "UPDATE batch_files SET updated_at=? WHERE id=?",
                ("2000-01-01T00:00:00+00:00", self.file_id),
            )
        self.manager.requeue_stale_processing()

        self.assertEqual(self.manager.claim_next_file()["attempt_count"], 2)
        with self.manager._connection() as connection:
            connection.execute(
                "UPDATE batch_files SET updated_at=? WHERE id=?",
                ("2000-01-01T00:00:00+00:00", self.file_id),
            )
        self.manager.requeue_stale_processing()

        failed = self.manager.get_file(self.file_id)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_code"], "worker_crashed")

    def test_queue_capacity_is_reserved_atomically(self) -> None:
        capacity_root = os.path.join(self.temp_dir.name, "capacity")
        with patch.dict(
            os.environ,
            {"AADHAAR_BATCH_STORAGE_ROOT": capacity_root, "AADHAAR_BATCH_MAX_PENDING_FILES": "1"},
        ):
            managers = (BatchManager(), BatchManager())
            batch_ids = (managers[0].new_id(), managers[1].new_id())
            file_ids = (managers[0].new_id(), managers[1].new_id())
            for manager, batch_id in zip(managers, batch_ids):
                manager.prepare_batch_dir(batch_id)
            barrier = threading.Barrier(2)

            def create(manager: BatchManager, batch_id: str, file_id: str) -> bool:
                barrier.wait()
                try:
                    manager.create_batch(
                        batch_id,
                        "upload",
                        [BatchSeed(file_id, "synthetic.jpg", "upload")],
                    )
                    return True
                except BatchQueueFull:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                accepted = list(pool.map(create, managers, batch_ids, file_ids))

        self.assertEqual(sum(accepted), 1)

    def test_legacy_score_columns_are_removed_on_migration(self) -> None:
        legacy_root = os.path.join(self.temp_dir.name, "legacy-schema")
        with patch.dict(os.environ, {"AADHAAR_BATCH_STORAGE_ROOT": legacy_root}):
            initial = BatchManager()
            with initial._connection() as connection:
                connection.execute("ALTER TABLE batch_files ADD COLUMN masking_confidence INTEGER")
                connection.execute("ALTER TABLE batch_files ADD COLUMN masking_confidence_source TEXT")
            initial.close()

            migrated = BatchManager()
            try:
                with migrated._connection() as connection:
                    columns = {
                        row["name"]
                        for row in connection.execute("PRAGMA table_info(batch_files)").fetchall()
                    }
                self.assertNotIn("masking_confidence", columns)
                self.assertNotIn("masking_confidence_source", columns)
            finally:
                migrated.close()


if __name__ == "__main__":
    unittest.main()
