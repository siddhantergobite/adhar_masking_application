from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from backend.benchmark import collect_input_paths, latency_summary, run_benchmark


class BatchBenchmarkTests(unittest.TestCase):
    def test_latency_summary_reports_mean_and_tail(self) -> None:
        self.assertEqual(
            latency_summary([1.0, 2.0, 3.0, 4.0]),
            {"count": 4, "mean": 2.5, "p50": 2.5, "p95": 3.85},
        )

    def test_positive_label_selection_skips_unlabeled_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (images / "negative.jpg").write_bytes(b"synthetic negative")
            (images / "positive.jpg").write_bytes(b"synthetic positive")
            (labels / "negative.txt").write_text("", encoding="utf-8")
            (labels / "positive.txt").write_text("0 0.5 0.5 0.4 0.4", encoding="utf-8")

            selected = collect_input_paths(images, labels)

        self.assertEqual([path.name for path in selected], ["positive.jpg"])

    def test_batch_benchmark_uploads_and_polls_without_exposing_names(self) -> None:
        received = {"files": 0, "api_key": "", "deleted": False}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                received["files"] = body.count(b'name="files"')
                received["api_key"] = self.headers.get("X-API-Key", "")
                payload = json.dumps({"batch_id": "abc123", "status": "queued"}).encode("utf-8")
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                payload = json.dumps({
                    "batch_id": "abc123",
                    "status": "completed_with_issues",
                    "files": [
                        {"status": "completed"},
                        {"status": "rejected"},
                    ],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_DELETE(self) -> None:
                received["deleted"] = True
                self.send_response(204)
                self.end_headers()

            def log_message(self, *_args) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                folder = Path(temp_dir)
                (folder / "private-one.jpg").write_bytes(b"synthetic-one")
                (folder / "private-two.jpg").write_bytes(b"synthetic-two")
                report = run_benchmark(
                    url=f"http://127.0.0.1:{server.server_port}/api/batches",
                    input_dir=folder,
                    mode="batch",
                    concurrency=1,
                    repeats=1,
                    warmups=0,
                    timeout=5,
                    poll_interval=0.01,
                    limit=2,
                )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

        self.assertEqual(received["files"], 2)
        self.assertEqual(report["successful_files"], 1)
        self.assertEqual(report["rejected_files"], 1)
        self.assertEqual(report["failed_files"], 0)
        self.assertEqual(report["cleanup_failures"], 0)
        self.assertTrue(received["deleted"])
        self.assertNotIn("private-one.jpg", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
