from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from . import cloudflare_kv


SUPPORTED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tif", ".tiff", ".gif", ".heic", ".heif",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _positive_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    value = int(raw) if raw else default
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _default_storage_root() -> Path:
    configured = os.getenv("AADHAAR_BATCH_STORAGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return (Path(local_app_data) / "AadhaarMaskingWorkbench" / "batches").resolve()
    return (Path.home() / ".aadhaar-masking-workbench" / "batches").resolve()


def safe_display_name(value: str | None) -> str:
    normalized = (value or "document").replace("\\", "/").split("/")[-1]
    normalized = "".join(character for character in normalized if character >= " " and character != "\x7f")
    return normalized[:180] or "document"


@dataclass(frozen=True)
class BatchSeed:
    file_id: str
    display_name: str
    source_type: str
    input_path: str | None = None
    source_bucket: str | None = None
    source_key: str | None = None
    source_size: int = 0
    initial_status: str = "queued"
    error_code: str | None = None
    error_message: str | None = None


Processor = Callable[[Path, str, Any, Any], Awaitable[tuple[bytes, dict[str, Any]]]]


class BatchManager:
    """Persistent, single-worker CPU batch queue and local result store."""

    def __init__(self) -> None:
        self.storage_root = _default_storage_root()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root = self.storage_root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        configured_input = os.getenv("AADHAAR_BATCH_INPUT_ROOT")
        self.input_root = (
            Path(configured_input).expanduser().resolve()
            if configured_input
            else (self.storage_root / "inbox").resolve()
        )
        self.input_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_root / "batches.sqlite3"
        self.max_files = _positive_env("AADHAAR_BATCH_MAX_FILES", 30)
        self.max_pending = _positive_env("AADHAAR_BATCH_MAX_PENDING_FILES", 90)
        self.max_total_bytes = _positive_env("AADHAAR_BATCH_MAX_BYTES", 750 * 1024 * 1024)
        self.retention_hours = _positive_env("AADHAAR_BATCH_RETENTION_HOURS", 24)
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._stop_requested = False
        self._init_db()
        self.purge_expired()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def batch_dir(self, batch_id: str) -> Path:
        if len(batch_id) != 32 or any(character not in "0123456789abcdef" for character in batch_id):
            raise ValueError("Invalid batch identifier")
        return self.jobs_root / batch_id

    def prepare_batch_dir(self, batch_id: str) -> Path:
        folder = self.batch_dir(batch_id)
        (folder / "input").mkdir(parents=True, exist_ok=False)
        (folder / "output").mkdir(parents=True, exist_ok=True)
        (folder / "work").mkdir(parents=True, exist_ok=True)
        return folder

    def staged_path(self, batch_id: str, file_id: str, suffix: str) -> Path:
        if len(file_id) != 32 or any(character not in "0123456789abcdef" for character in file_id):
            raise ValueError("Invalid file identifier")
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("Invalid file extension")
        return self.batch_dir(batch_id) / "input" / f"{file_id}{suffix.lower()}"

    def output_path(self, batch_id: str, file_id: str) -> Path:
        return self.batch_dir(batch_id) / "output" / f"masked-{file_id}.pdf"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_files INTEGER NOT NULL,
                    completed_files INTEGER NOT NULL DEFAULT 0,
                    rejected_files INTEGER NOT NULL DEFAULT 0,
                    failed_files INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS batch_files (
                    id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL REFERENCES batch_jobs(id) ON DELETE CASCADE,
                    display_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT,
                    source_bucket TEXT,
                    source_key TEXT,
                    source_size INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    output_path TEXT,
                    output_s3_uri TEXT,
                    document_count INTEGER NOT NULL DEFAULT 0,
                    masked_regions INTEGER NOT NULL DEFAULT 0,
                    masking_confidence INTEGER,
                    document_classes TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS batch_files_batch_idx
                    ON batch_files(batch_id, created_at);
                CREATE INDEX IF NOT EXISTS batch_files_status_idx
                    ON batch_files(status, created_at);
                """
            )

    def pending_count(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS amount FROM batch_files WHERE status IN ('queued', 'processing')"
            ).fetchone()
        return int(row["amount"])

    def create_batch(self, batch_id: str, source_type: str, seeds: list[BatchSeed]) -> None:
        if not seeds or len(seeds) > self.max_files:
            raise ValueError("Batch file count is outside the configured limit")
        created_at = _now()
        initial_status = "queued" if any(seed.initial_status == "queued" for seed in seeds) else "completed_with_issues"
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO batch_jobs(id, source_type, status, created_at, updated_at, total_files) VALUES (?, ?, ?, ?, ?, ?)",
                (batch_id, source_type, initial_status, created_at, created_at, len(seeds)),
            )
            for seed in seeds:
                connection.execute(
                    """INSERT INTO batch_files(
                        id, batch_id, display_name, source_type, source_ref, source_bucket, source_key,
                        source_size, status, error_code, error_message, created_at, updated_at,
                        completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        seed.file_id,
                        batch_id,
                        safe_display_name(seed.display_name),
                        seed.source_type,
                        seed.input_path,
                        seed.source_bucket,
                        seed.source_key,
                        seed.source_size,
                        seed.initial_status,
                        seed.error_code,
                        seed.error_message,
                        created_at,
                        created_at,
                        created_at if seed.initial_status != "queued" else None,
                    ),
                )
            self._refresh_batch(connection, batch_id)

    def discard_workspace(self, batch_id: str) -> None:
        folder = self.batch_dir(batch_id)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    def _refresh_batch(self, connection: sqlite3.Connection, batch_id: str) -> None:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS amount FROM batch_files WHERE batch_id=? GROUP BY status",
            (batch_id,),
        ).fetchall()
        counts = {row["status"]: int(row["amount"]) for row in rows}
        queued = counts.get("queued", 0)
        processing = counts.get("processing", 0)
        completed = counts.get("completed", 0)
        rejected = counts.get("rejected", 0)
        failed = counts.get("failed", 0)
        if queued or processing:
            status = "processing" if processing else "queued"
            completed_at = None
        elif rejected or failed:
            status = "completed_with_issues"
            completed_at = _now()
        else:
            status = "completed"
            completed_at = _now()
        connection.execute(
            """UPDATE batch_jobs SET status=?, updated_at=?, completed_at=?, completed_files=?,
               rejected_files=?, failed_files=? WHERE id=?""",
            (status, _now(), completed_at, completed, rejected, failed, batch_id),
        )

    def reset_interrupted(self) -> None:
        with self._connection() as connection:
            batch_ids = [row[0] for row in connection.execute(
                "SELECT DISTINCT batch_id FROM batch_files WHERE status='processing'"
            ).fetchall()]
            connection.execute(
                "UPDATE batch_files SET status='queued', updated_at=?, started_at=NULL WHERE status='processing'",
                (_now(),),
            )
            for batch_id in batch_ids:
                self._refresh_batch(connection, batch_id)

    def queued_file_ids(self) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id FROM batch_files WHERE status='queued' ORDER BY created_at, rowid"
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def claim_file(self, file_id: str) -> dict[str, Any] | None:
        now = _now()
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM batch_files WHERE id=?", (file_id,)).fetchone()
            if row is None or row["status"] != "queued":
                return None
            connection.execute(
                "UPDATE batch_files SET status='processing', started_at=?, updated_at=? WHERE id=? AND status='queued'",
                (now, now, file_id),
            )
            self._refresh_batch(connection, row["batch_id"])
            updated = connection.execute("SELECT * FROM batch_files WHERE id=?", (file_id,)).fetchone()
        return dict(updated)

    def finish_file(
        self,
        file_id: str,
        *,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
        output_path: str | None = None,
        output_s3_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        with self._connection() as connection:
            row = connection.execute("SELECT batch_id FROM batch_files WHERE id=?", (file_id,)).fetchone()
            if row is None:
                return
            connection.execute(
                """UPDATE batch_files SET status=?, error_code=?, error_message=?, output_path=?,
                   output_s3_uri=?, document_count=?, masked_regions=?, masking_confidence=?,
                   document_classes=?, updated_at=?, completed_at=? WHERE id=?""",
                (
                    status,
                    error_code,
                    error_message,
                    output_path,
                    output_s3_uri,
                    int(metadata.get("document_count", 0)),
                    int(metadata.get("masked_regions", 0)),
                    metadata.get("masking_confidence"),
                    json.dumps(metadata.get("document_classes", [])),
                    _now(),
                    _now(),
                    file_id,
                ),
            )
            self._refresh_batch(connection, row["batch_id"])

    def get_file(self, file_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM batch_files WHERE id=?", (file_id,)).fetchone()
        return dict(row) if row else None

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            batch = connection.execute("SELECT * FROM batch_jobs WHERE id=?", (batch_id,)).fetchone()
            if batch is None:
                return None
            files = connection.execute(
                "SELECT * FROM batch_files WHERE batch_id=? ORDER BY created_at, rowid",
                (batch_id,),
            ).fetchall()
        return {"batch": dict(batch), "files": [dict(row) for row in files]}

    def delete_batch(self, batch_id: str) -> bool:
        self.batch_dir(batch_id)
        with self._connection() as connection:
            row = connection.execute("SELECT status FROM batch_jobs WHERE id=?", (batch_id,)).fetchone()
            if row is None:
                return False
            if row["status"] in {"queued", "processing"}:
                raise ValueError("Wait until the batch finishes before deleting it")
            output_uris = [
                item[0] for item in connection.execute(
                    "SELECT output_s3_uri FROM batch_files WHERE batch_id=? AND output_s3_uri IS NOT NULL",
                    (batch_id,),
                ).fetchall()
            ]
        self.delete_s3_outputs(output_uris)
        with self._connection() as connection:
            connection.execute("DELETE FROM batch_jobs WHERE id=?", (batch_id,))
        self.discard_workspace(batch_id)
        return True

    def purge_expired(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)).isoformat(timespec="seconds")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id FROM batch_jobs WHERE completed_at IS NOT NULL AND completed_at < ?",
                (cutoff,),
            ).fetchall()
            ids = [str(row["id"]) for row in rows]
        deleted_ids = []
        for batch_id in ids:
            with self._connection() as connection:
                output_uris = [
                    item[0] for item in connection.execute(
                        "SELECT output_s3_uri FROM batch_files WHERE batch_id=? AND output_s3_uri IS NOT NULL",
                        (batch_id,),
                    ).fetchall()
                ]
            try:
                self.delete_s3_outputs(output_uris)
            except Exception:
                # Keep both the result record and local copy; retry on the next cleanup pass.
                continue
            with self._connection() as connection:
                connection.execute("DELETE FROM batch_jobs WHERE id=?", (batch_id,))
            self.discard_workspace(batch_id)
            deleted_ids.append(batch_id)
        return len(deleted_ids)

    def create_archive(self, batch_id: str) -> Path:
        info = self.get_batch(batch_id)
        if info is None:
            raise FileNotFoundError("Batch not found")
        if info["batch"]["status"] in {"queued", "processing"}:
            raise RuntimeError("Batch is still processing")
        archive = self.batch_dir(batch_id) / "output" / f"batch-{batch_id}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            manifest = []
            for item in info["files"]:
                output = item.get("output_path")
                if output and Path(output).is_file():
                    arcname = f"masked-{item['id']}.pdf"
                    bundle.write(output, arcname=arcname)
                    manifest.append({
                        "file_id": item["id"],
                        "source_name": item["display_name"],
                        "status": item["status"],
                        "output": arcname,
                        "masked_regions": item["masked_regions"],
                        "masking_confidence": item["masking_confidence"],
                    })
                else:
                    manifest.append({
                        "file_id": item["id"],
                        "source_name": item["display_name"],
                        "status": item["status"],
                        "error_code": item["error_code"],
                        "error_message": item["error_message"],
                    })
            bundle.writestr("manifest.json", json.dumps({"batch_id": batch_id, "files": manifest}, indent=2))
        return archive

    async def submit(self, batch_id: str) -> None:
        info = self.get_batch(batch_id)
        if info is None:
            return
        for item in info["files"]:
            if item["status"] == "queued":
                await self.queue.put(item["id"])

    async def run(self, app: Any, processor: Processor) -> None:
        self.reset_interrupted()
        for file_id in self.queued_file_ids():
            await self.queue.put(file_id)
        last_purge = time.monotonic()
        while not self._stop_requested:
            if app.state.pipeline is None:
                await asyncio.sleep(0.5)
                continue
            try:
                file_id = await asyncio.wait_for(self.queue.get(), timeout=1)
            except asyncio.TimeoutError:
                if time.monotonic() - last_purge >= 3600:
                    await asyncio.to_thread(self.purge_expired)
                    last_purge = time.monotonic()
                continue
            try:
                await self._process_one(app, processor, file_id)
            finally:
                self.queue.task_done()

    def request_stop(self) -> None:
        self._stop_requested = True

    async def _process_one(self, app: Any, processor: Processor, file_id: str) -> None:
        item = self.claim_file(file_id)
        if item is None:
            return
        batch_id = item["batch_id"]
        staged_path: Path | None = None
        temporary_s3_file = item["source_type"] in {"s3", "r2"}
        try:
            if temporary_s3_file:
                suffix = Path(item["source_key"]).suffix.lower()
                staged_path = self.staged_path(batch_id, file_id, suffix)
                try:
                    await asyncio.to_thread(
                        self.download_s3_source,
                        item["source_bucket"],
                        item["source_key"],
                        staged_path,
                        int(item["source_size"]),
                    )
                except Exception as exc:
                    raise S3SourceUnavailable from exc
            elif item["source_type"] == "cloudflare_kv":
                source_ref = item.get("source_ref")
                staged_path = Path(source_ref) if source_ref else self.staged_path(
                    batch_id,
                    file_id,
                    Path(item["source_key"]).suffix.lower(),
                )
                if not staged_path.is_file():
                    try:
                        await asyncio.to_thread(
                            cloudflare_kv.download_value,
                            item["source_key"],
                            staged_path,
                            app.state.settings.max_upload_bytes,
                        )
                    except Exception as exc:
                        raise CloudflareKVSourceUnavailable from exc
            else:
                staged_path = Path(item["source_ref"])
            if staged_path is None or not staged_path.is_file():
                raise FileNotFoundError("The staged input file is missing")

            settings = app.state.settings
            if staged_path.stat().st_size > settings.max_upload_bytes:
                self.finish_file(
                    file_id,
                    status="rejected",
                    error_code="file_too_large",
                    error_message="This file exceeds the per-file size limit.",
                )
                return

            pdf_bytes, metadata = await processor(
                staged_path,
                item["display_name"],
                app.state.pipeline,
                settings,
            )
            result_path = self.output_path(batch_id, file_id)
            await asyncio.to_thread(result_path.write_bytes, pdf_bytes)
            output_s3_uri = None
            if item["source_type"] == "cloudflare_kv":
                output_prefix = cloudflare_kv.output_prefix()
                output_key = f"{output_prefix}{batch_id}/masked-{file_id}.pdf"
                try:
                    await asyncio.to_thread(cloudflare_kv.put_file, output_key, result_path)
                    output_s3_uri = cloudflare_kv.make_uri(output_key)
                except Exception:
                    self.finish_file(
                        file_id,
                        status="failed",
                        error_code="output_upload_failed",
                        error_message="The masked PDF was created locally, but could not be saved to Workers KV.",
                        output_path=str(result_path),
                        metadata=metadata,
                    )
                    return

            output_bucket = (
                os.getenv("AADHAAR_S3_OUTPUT_BUCKET", "").strip()
                or os.getenv("CLOUDFLARE_R2_BUCKET", "").strip()
            )
            if output_bucket and item["source_type"] != "cloudflare_kv":
                default_output_prefix = (
                    os.getenv("CLOUDFLARE_R2_OUTPUT_PREFIX", "aadhaar/masked")
                    if os.getenv("CLOUDFLARE_R2_BUCKET", "").strip()
                    else "masked"
                )
                output_prefix = os.getenv("AADHAAR_S3_OUTPUT_PREFIX", default_output_prefix).strip("/")
                output_key = "/".join(part for part in (output_prefix, batch_id, f"masked-{file_id}.pdf") if part)
                try:
                    await asyncio.to_thread(self.upload_s3_result, output_bucket, output_key, result_path)
                    output_s3_uri = f"s3://{output_bucket}/{output_key}"
                except Exception:
                    self.finish_file(
                        file_id,
                        status="failed",
                        error_code="output_upload_failed",
                        error_message="The masked PDF was created locally, but could not be saved to the configured cloud storage location.",
                        output_path=str(result_path),
                        metadata=metadata,
                    )
                    return
            self.finish_file(
                file_id,
                status="completed",
                output_path=str(result_path),
                output_s3_uri=output_s3_uri,
                metadata=metadata,
            )
        except Exception as exc:
            from fastapi import HTTPException

            from .errors import PipelineError

            if isinstance(exc, PipelineError):
                detail = exc.detail()
                self.finish_file(
                    file_id,
                    status="rejected" if exc.status_code < 500 else "failed",
                    error_code=detail["code"],
                    error_message=detail["message"],
                )
            elif isinstance(exc, HTTPException):
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                self.finish_file(
                    file_id,
                    status="rejected" if exc.status_code < 500 else "failed",
                    error_code=str(detail.get("code", "unsupported_file" if exc.status_code == 415 else "processing_failed")),
                    error_message=str(detail.get("message", "The file could not be processed.")),
                )
            elif isinstance(exc, S3SourceUnavailable):
                self.finish_file(
                    file_id,
                    status="failed",
                    error_code="source_unavailable",
                    error_message="The S3 source could not be read. Check the configured bucket access and object key.",
                )
            elif isinstance(exc, CloudflareKVSourceUnavailable):
                self.finish_file(
                    file_id,
                    status="failed",
                    error_code="source_unavailable",
                    error_message="The Workers KV source could not be read. Check the configured account, namespace, and token permissions.",
                )
            else:
                self.finish_file(
                    file_id,
                    status="failed",
                    error_code="processing_failed",
                    error_message="Processing failed unexpectedly. No masked PDF was returned for this file.",
                )
        finally:
            if staged_path is not None and staged_path.exists():
                try:
                    staged_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _s3_client():
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("S3 support is optional; install backend/requirements-s3.txt first") from exc
        options: dict[str, Any] = {}
        r2_account_id = os.getenv("CLOUDFLARE_R2_ACCOUNT_ID", "").strip()
        r2_access_key = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID", "").strip()
        r2_secret_key = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", "").strip()
        if r2_account_id or r2_access_key or r2_secret_key:
            if not all((r2_account_id, r2_access_key, r2_secret_key)):
                raise RuntimeError(
                    "Configure CLOUDFLARE_R2_ACCOUNT_ID, CLOUDFLARE_R2_ACCESS_KEY_ID, and CLOUDFLARE_R2_SECRET_ACCESS_KEY together."
                )
            options["aws_access_key_id"] = r2_access_key
            options["aws_secret_access_key"] = r2_secret_key
            options["region_name"] = "auto"
            options["endpoint_url"] = (
                os.getenv("CLOUDFLARE_R2_ENDPOINT_URL", "").strip()
                or f"https://{r2_account_id}.r2.cloudflarestorage.com"
            )
        else:
            region = os.getenv("AADHAAR_S3_REGION", "").strip()
            endpoint = os.getenv("AADHAAR_S3_ENDPOINT_URL", "").strip()
            if region:
                options["region_name"] = region
            if endpoint:
                options["endpoint_url"] = endpoint
        return boto3.client("s3", **options)

    @staticmethod
    def cloudflare_r2_configured() -> bool:
        required = (
            "CLOUDFLARE_R2_ACCOUNT_ID",
            "CLOUDFLARE_R2_BUCKET",
            "CLOUDFLARE_R2_ACCESS_KEY_ID",
            "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
        )
        if not all(os.getenv(name, "").strip() for name in required):
            return False
        try:
            import boto3  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def list_s3_objects(cls, bucket: str, prefix: str, max_files: int) -> list[dict[str, Any]]:
        client = cls._s3_client()
        paginator = client.get_paginator("list_objects_v2")
        found: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                if Path(key).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                found.append({"key": key, "size": int(item.get("Size", 0))})
                if len(found) > max_files:
                    return found
        return found

    @classmethod
    def download_s3_source(cls, bucket: str, key: str, destination: Path, expected_size: int) -> None:
        if expected_size < 0:
            raise ValueError("Invalid S3 object size")
        destination.parent.mkdir(parents=True, exist_ok=True)
        cls._s3_client().download_file(bucket, key, str(destination))
        if destination.stat().st_size != expected_size:
            destination.unlink(missing_ok=True)
            raise IOError("Downloaded S3 object size did not match the listed object size")

    @classmethod
    def upload_s3_result(cls, bucket: str, key: str, source: Path) -> None:
        cls._s3_client().upload_file(
            str(source),
            bucket,
            key,
            ExtraArgs={"ContentType": "application/pdf"},
        )

    @classmethod
    def upload_s3_object(cls, bucket: str, key: str, source: Path, content_type: str) -> None:
        cls._s3_client().upload_file(
            str(source),
            bucket,
            key,
            ExtraArgs={"ContentType": content_type or "application/octet-stream"},
        )

    @classmethod
    def delete_s3_outputs(cls, uris: list[str]) -> None:
        if not uris:
            return
        s3_uris: list[str] = []
        for uri in uris:
            parsed = urlsplit(uri)
            if parsed.scheme == "kv":
                cloudflare_kv.delete_uri(uri)
                continue
            if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
                raise ValueError("Stored S3 output location is invalid")
            s3_uris.append(uri)
        if not s3_uris:
            return
        client = cls._s3_client()
        for uri in s3_uris:
            parsed = urlsplit(uri)
            client.delete_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))


def resolve_input_folder(root: Path, relative_folder: str) -> Path:
    relative = relative_folder.replace("\\", "/")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part == ".." for part in parsed.parts):
        raise ValueError("Folder must be a relative path inside the configured input root")
    target = (root / Path(*parsed.parts)).resolve()
    allowed_root = root.resolve()
    if target != allowed_root and allowed_root not in target.parents:
        raise ValueError("Folder must stay inside the configured input root")
    if not target.is_dir():
        raise FileNotFoundError("The configured input folder was not found")
    return target


def scan_input_folder(folder: Path, *, recursive: bool) -> list[Path]:
    allowed_root = folder.resolve()
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    found: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if (
            resolved.is_file()
            and (resolved == allowed_root or allowed_root in resolved.parents)
            and resolved.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            found.append(resolved)
    return sorted(found, key=lambda value: value.name.casefold())


class S3SourceUnavailable(RuntimeError):
    pass


class CloudflareKVSourceUnavailable(RuntimeError):
    pass

