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
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable
from urllib.parse import unquote, urlsplit

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


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    value = float(raw) if raw else default
    if value <= 0:
        raise ValueError(f"{name} must be positive")
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


class BatchQueueFull(RuntimeError):
    """The configured pending/processing capacity has been reached."""


class _PostgresConnectionAdapter:
    """Translate the small SQLite-style query surface to psycopg placeholders."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def execute(self, query: str, parameters: tuple[Any, ...] = ()) -> Any:
        return self.connection.execute(query.replace("?", "%s"), parameters)


class BatchManager:
    """Persistent batch queue with SQLite local or PostgreSQL shared state."""

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
        self.database_url = os.getenv("AADHAAR_DATABASE_URL", "").strip()
        self.use_postgres = bool(self.database_url)
        self._postgres_pool: Any | None = None
        self.shared_input_bucket = ""
        self.shared_input_prefix = ""
        self.shared_temp_prefix = ""
        self.shared_output_bucket = ""
        self.shared_output_prefix = ""
        self.max_files = _positive_env("AADHAAR_BATCH_MAX_FILES", 30)
        self.max_pending = _positive_env("AADHAAR_BATCH_MAX_PENDING_FILES", 90)
        self.max_total_bytes = _positive_env("AADHAAR_BATCH_MAX_BYTES", 750 * 1024 * 1024)
        self.retention_hours = _positive_env("AADHAAR_BATCH_RETENTION_HOURS", 24)
        self.poll_seconds = min(5.0, _positive_float_env("AADHAAR_BATCH_POLL_SECONDS", 0.25))
        self.lease_seconds = max(30, _positive_env("AADHAAR_BATCH_LEASE_SECONDS", 300))
        self.max_attempts = _positive_env("AADHAAR_BATCH_MAX_ATTEMPTS", 3)
        self._stop_requested = False
        if self.use_postgres:
            self._configure_shared_storage()
            try:
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL queue mode requires psycopg[binary,pool]; install backend/requirements-postgres.txt"
                ) from exc
            pool_size = _positive_env("AADHAAR_DB_POOL_SIZE", 4)
            self._postgres_pool = ConnectionPool(
                conninfo=self.database_url,
                min_size=1,
                max_size=pool_size,
                kwargs={"row_factory": dict_row},
                open=False,
            )
            try:
                self._postgres_pool.open(wait=True)
            except Exception:
                self._postgres_pool.close()
                self._postgres_pool = None
                raise
        self._init_db()
        self.purge_expired()

    def _configure_shared_storage(self) -> None:
        r2_bucket = os.getenv("CLOUDFLARE_R2_BUCKET", "").strip()
        self.shared_input_bucket = os.getenv("AADHAAR_S3_INPUT_BUCKET", "").strip() or r2_bucket
        self.shared_output_bucket = os.getenv("AADHAAR_S3_OUTPUT_BUCKET", "").strip() or r2_bucket
        self.shared_input_prefix = os.getenv(
            "AADHAAR_S3_INPUT_PREFIX",
            os.getenv("CLOUDFLARE_R2_INPUT_PREFIX", "aadhaar/inbox") if r2_bucket else "",
        ).strip("/")
        self.shared_temp_prefix = os.getenv("AADHAAR_S3_TEMP_PREFIX", "aadhaar/queued-input").strip("/")
        default_output = os.getenv("CLOUDFLARE_R2_OUTPUT_PREFIX", "aadhaar/masked") if r2_bucket else ""
        self.shared_output_prefix = os.getenv("AADHAAR_S3_OUTPUT_PREFIX", default_output).strip("/")
        if not all((self.shared_input_bucket, self.shared_input_prefix, self.shared_temp_prefix,
                    self.shared_output_bucket, self.shared_output_prefix)):
            raise RuntimeError(
                "PostgreSQL mode requires shared S3/R2 input, temporary-upload, and output buckets with non-empty prefixes"
            )
        prefixes_by_bucket = {
            self.shared_input_bucket: [self.shared_input_prefix, self.shared_temp_prefix],
        }
        prefixes_by_bucket.setdefault(self.shared_output_bucket, []).append(self.shared_output_prefix)
        for prefixes in prefixes_by_bucket.values():
            for index, first in enumerate(prefixes):
                for second in prefixes[index + 1:]:
                    if first == second or first.startswith(second + "/") or second.startswith(first + "/"):
                        raise RuntimeError("Shared S3/R2 input, temporary-upload, and output prefixes must not overlap")
        if r2_bucket and (
            self.shared_input_bucket != r2_bucket or self.shared_output_bucket != r2_bucket
        ):
            raise RuntimeError("Use one S3-compatible storage service for both shared input and output in PostgreSQL mode")
        try:
            import boto3  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL mode also requires the S3 client; install backend/requirements-s3.txt"
            ) from exc
        if r2_bucket and not self.cloudflare_r2_configured():
            raise RuntimeError("Configure complete Cloudflare R2 credentials before enabling PostgreSQL mode")

    def close(self) -> None:
        if self._postgres_pool is not None:
            self._postgres_pool.close()
            self._postgres_pool = None

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

    def attempt_path(self, batch_id: str, file_id: str, lease_token: str, suffix: str) -> Path:
        if not lease_token or any(character not in "0123456789abcdef" for character in lease_token):
            raise ValueError("Invalid worker lease token")
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("Invalid file extension")
        path = self.batch_dir(batch_id) / "work" / f"{file_id}-{lease_token}{suffix.lower()}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def store_shared_input(
        self,
        batch_id: str,
        file_id: str,
        suffix: str,
        source: Path,
        content_type: str,
    ) -> dict[str, Any] | None:
        if not self.use_postgres:
            return None
        bucket = self.shared_input_bucket
        key = f"{self.shared_temp_prefix}/{batch_id}/{file_id}{suffix.lower()}"
        self.upload_s3_object(bucket, key, source, content_type)
        r2_bucket = os.getenv("CLOUDFLARE_R2_BUCKET", "").strip()
        return {
            "source_type": "r2_temporary" if r2_bucket and bucket == r2_bucket else "s3_temporary",
            "source_bucket": bucket,
            "source_key": key,
            "source_size": source.stat().st_size,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self):
        if self._postgres_pool is not None:
            with self._postgres_pool.connection() as connection:
                yield _PostgresConnectionAdapter(connection)
            return
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
        if self.use_postgres:
            self._init_postgres_db()
            return
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
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    lease_token TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    output_path TEXT,
                    output_s3_uri TEXT,
                    document_count INTEGER NOT NULL DEFAULT 0,
                    masked_regions INTEGER NOT NULL DEFAULT 0,
                    document_classes TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS batch_queue_guard (
                    id INTEGER PRIMARY KEY
                );
                INSERT OR IGNORE INTO batch_queue_guard(id) VALUES (1);
                CREATE INDEX IF NOT EXISTS batch_files_batch_idx
                    ON batch_files(batch_id, created_at);
                CREATE INDEX IF NOT EXISTS batch_files_status_idx
                    ON batch_files(status, created_at);
                """
            )
            # Serialize SQLite schema migrations across Uvicorn worker startup.
            connection.execute("BEGIN IMMEDIATE")
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(batch_files)").fetchall()
            }
            if "attempt_count" not in columns:
                connection.execute(
                    "ALTER TABLE batch_files ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"
                )
            if "lease_token" not in columns:
                connection.execute("ALTER TABLE batch_files ADD COLUMN lease_token TEXT")
            for obsolete_column in ("masking_confidence", "masking_confidence_source"):
                if obsolete_column in columns:
                    connection.execute(f"ALTER TABLE batch_files DROP COLUMN {obsolete_column}")

    def _init_postgres_db(self) -> None:
        statements = (
            """CREATE TABLE IF NOT EXISTS batch_jobs (
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
            )""",
            """CREATE TABLE IF NOT EXISTS batch_files (
                id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES batch_jobs(id) ON DELETE CASCADE,
                display_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT,
                source_bucket TEXT,
                source_key TEXT,
                source_size BIGINT NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_token TEXT,
                error_code TEXT,
                error_message TEXT,
                output_path TEXT,
                output_s3_uri TEXT,
                document_count INTEGER NOT NULL DEFAULT 0,
                masked_regions INTEGER NOT NULL DEFAULT 0,
                document_classes TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )""",
            "CREATE TABLE IF NOT EXISTS batch_queue_guard (id INTEGER PRIMARY KEY)",
            "INSERT INTO batch_queue_guard(id) VALUES (1) ON CONFLICT (id) DO NOTHING",
            "CREATE INDEX IF NOT EXISTS batch_files_batch_idx ON batch_files(batch_id, created_at, id)",
            "CREATE INDEX IF NOT EXISTS batch_files_status_idx ON batch_files(status, created_at, id)",
            "ALTER TABLE batch_files ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE batch_files ADD COLUMN IF NOT EXISTS lease_token TEXT",
            "ALTER TABLE batch_files DROP COLUMN IF EXISTS masking_confidence",
            "ALTER TABLE batch_files DROP COLUMN IF EXISTS masking_confidence_source",
        )
        with self._connection() as connection:
            for statement in statements:
                connection.execute(statement)

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
            connection.execute("BEGIN" if self.use_postgres else "BEGIN IMMEDIATE")
            if self.use_postgres:
                connection.execute("SELECT id FROM batch_queue_guard WHERE id=1 FOR UPDATE").fetchone()
            pending = connection.execute(
                "SELECT COUNT(*) AS amount FROM batch_files WHERE status IN ('queued', 'processing')"
            ).fetchone()
            queued_seeds = sum(seed.initial_status == "queued" for seed in seeds)
            if int(pending["amount"]) + queued_seeds > self.max_pending:
                raise BatchQueueFull("The batch queue is full")
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

    def _refresh_batch(self, connection: Any, batch_id: str) -> None:
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

    def requeue_stale_processing(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=self.lease_seconds)).isoformat(timespec="seconds")
        now = _now()
        terminal_remote_sources: list[str] = []
        changed_count = 0
        with self._connection() as connection:
            connection.execute("BEGIN" if self.use_postgres else "BEGIN IMMEDIATE")
            stale_rows = connection.execute(
                "SELECT id, batch_id, attempt_count, source_type, source_bucket, source_key FROM batch_files "
                "WHERE status='processing' AND updated_at < ?",
                (cutoff,),
            ).fetchall()
            batch_ids = {row["batch_id"] for row in stale_rows}
            for row in stale_rows:
                if int(row["attempt_count"]) >= self.max_attempts:
                    cursor = connection.execute(
                        "UPDATE batch_files SET status='failed', error_code='worker_crashed', "
                        "error_message='The worker stopped repeatedly while processing this file.', "
                        "lease_token=NULL, updated_at=?, completed_at=? "
                        "WHERE id=? AND status='processing' AND updated_at < ?",
                        (now, now, row["id"], cutoff),
                    )
                    changed_count += int(cursor.rowcount == 1)
                    if cursor.rowcount == 1 and row["source_type"] in {"s3_temporary", "r2_temporary"} and row["source_bucket"] and row["source_key"]:
                        terminal_remote_sources.append(
                            f"s3://{row['source_bucket']}/{row['source_key']}"
                        )
                else:
                    cursor = connection.execute(
                        "UPDATE batch_files SET status='queued', lease_token=NULL, updated_at=?, started_at=NULL "
                        "WHERE id=? AND status='processing' AND updated_at < ?",
                        (now, row["id"], cutoff),
                    )
                    changed_count += int(cursor.rowcount == 1)
            for batch_id in batch_ids:
                self._refresh_batch(connection, batch_id)
        if terminal_remote_sources:
            try:
                self.delete_s3_outputs(terminal_remote_sources)
            except Exception:
                # Failed cleanup can be handled by the object-store lifecycle policy.
                pass
        return changed_count

    def claim_next_file(self) -> dict[str, Any] | None:
        now = _now()
        lease_token = uuid.uuid4().hex
        with self._connection() as connection:
            # Serialize the short claim transaction across API worker processes.
            connection.execute("BEGIN" if self.use_postgres else "BEGIN IMMEDIATE")
            claim_query = (
                "SELECT * FROM batch_files WHERE status='queued' ORDER BY created_at, id LIMIT 1"
            )
            if self.use_postgres:
                claim_query += " FOR UPDATE SKIP LOCKED"
            row = connection.execute(claim_query).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                "UPDATE batch_files SET status='processing', started_at=?, updated_at=?, lease_token=?, "
                "attempt_count=attempt_count+1 "
                "WHERE id=? AND status='queued'",
                (now, now, lease_token, row["id"]),
            )
            if cursor.rowcount != 1:
                return None
            self._refresh_batch(connection, row["batch_id"])
            updated = connection.execute("SELECT * FROM batch_files WHERE id=?", (row["id"],)).fetchone()
        return dict(updated)

    def touch_processing_file(self, file_id: str, lease_token: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE batch_files SET updated_at=? WHERE id=? AND status='processing' AND lease_token=?",
                (_now(), file_id, lease_token),
            )
        return cursor.rowcount == 1

    def finish_file(
        self,
        file_id: str,
        *,
        lease_token: str | None = None,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
        output_path: str | None = None,
        output_s3_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        metadata = metadata or {}
        with self._connection() as connection:
            row = connection.execute("SELECT batch_id FROM batch_files WHERE id=?", (file_id,)).fetchone()
            if row is None:
                return False
            lease_filter = " AND status='processing' AND lease_token=?" if lease_token is not None else ""
            parameters: tuple[Any, ...] = (
                status,
                error_code,
                error_message,
                output_path,
                output_s3_uri,
                int(metadata.get("document_count", 0)),
                int(metadata.get("masked_regions", 0)),
                json.dumps(metadata.get("document_classes", [])),
                _now(),
                _now(),
                file_id,
            )
            if lease_token is not None:
                parameters += (lease_token,)
            cursor = connection.execute(
                """UPDATE batch_files SET status=?, error_code=?, error_message=?, output_path=?,
                   output_s3_uri=?, document_count=?, masked_regions=?, document_classes=?,
                   updated_at=?, completed_at=? WHERE id=?"""
                + lease_filter,
                parameters,
            )
            if cursor.rowcount != 1:
                return False
            self._refresh_batch(connection, row["batch_id"])
        return True

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
                "SELECT * FROM batch_files WHERE batch_id=? ORDER BY created_at, id",
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
                item["output_s3_uri"] for item in connection.execute(
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
                    item["output_s3_uri"] for item in connection.execute(
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
        archive = self.batch_dir(batch_id) / "output" / f"batch-{batch_id}-{self.new_id()}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            manifest = []
            for item in info["files"]:
                output = item.get("output_path")
                output_path = Path(output) if output and Path(output).is_file() else None
                downloaded_output: Path | None = None
                if output_path is None and item.get("output_s3_uri"):
                    downloaded_output = archive.parent / f".download-{item['id']}-{self.new_id()}.pdf"
                    try:
                        self.materialize_output(item["output_s3_uri"], downloaded_output)
                        output_path = downloaded_output
                    except Exception:
                        downloaded_output.unlink(missing_ok=True)
                        raise
                if output_path is not None:
                    arcname = f"masked-{item['id']}.pdf"
                    try:
                        bundle.write(output_path, arcname=arcname)
                        manifest.append({
                            "file_id": item["id"],
                            "source_name": item["display_name"],
                            "status": item["status"],
                            "output": arcname,
                            "masked_regions": item["masked_regions"],
                        })
                    finally:
                        if downloaded_output is not None:
                            downloaded_output.unlink(missing_ok=True)
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

    async def run(self, app: Any, processor: Processor) -> None:
        last_purge = last_recovery = time.monotonic()
        recovery_interval = max(10.0, self.lease_seconds / 3)
        while not self._stop_requested:
            if app.state.pipeline is None:
                await asyncio.sleep(0.5)
                continue
            now = time.monotonic()
            if now - last_recovery >= recovery_interval:
                await asyncio.to_thread(self.requeue_stale_processing)
                last_recovery = now
            item = await asyncio.to_thread(self.claim_next_file)
            if item is None:
                if now - last_purge >= 3600:
                    await asyncio.to_thread(self.purge_expired)
                    last_purge = now
                await asyncio.sleep(self.poll_seconds)
                continue

            heartbeat = asyncio.create_task(self._heartbeat(item["id"], item["lease_token"]))
            try:
                await self._process_one(app, processor, item)
            finally:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat

    async def _heartbeat(self, file_id: str, lease_token: str) -> None:
        interval = max(5.0, self.lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            if not await asyncio.to_thread(self.touch_processing_file, file_id, lease_token):
                return

    def request_stop(self) -> None:
        self._stop_requested = True

    async def _process_one(self, app: Any, processor: Processor, item: dict[str, Any]) -> None:
        file_id = item["id"]
        batch_id = item["batch_id"]
        lease_token = item["lease_token"]
        staged_path: Path | None = None
        temporary_s3_file = item["source_type"] in {"s3", "r2", "s3_temporary", "r2_temporary"}
        temporary_remote_source = item["source_type"] in {"s3_temporary", "r2_temporary"}

        async def complete(**values: Any) -> bool:
            accepted = await asyncio.to_thread(
                self.finish_file,
                file_id,
                lease_token=lease_token,
                **values,
            )
            if not accepted:
                output_path = values.get("output_path")
                if output_path:
                    try:
                        await asyncio.to_thread(Path(output_path).unlink, missing_ok=True)
                    except OSError:
                        pass
                output_uri = values.get("output_s3_uri")
                if output_uri:
                    try:
                        await asyncio.to_thread(self.delete_s3_outputs, [output_uri])
                    except Exception:
                        pass
            return accepted

        try:
            if temporary_s3_file:
                suffix = Path(item["source_key"]).suffix.lower()
                staged_path = self.attempt_path(batch_id, file_id, lease_token, suffix)
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
                if source_ref and not self.use_postgres:
                    staged_path = Path(source_ref)
                else:
                    staged_path = self.attempt_path(
                        batch_id,
                        file_id,
                        lease_token,
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
                await complete(
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
            result_path = self.batch_dir(batch_id) / "work" / f"masked-{file_id}-{lease_token}.pdf"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(result_path.write_bytes, pdf_bytes)
            output_s3_uri = None
            if item["source_type"] == "cloudflare_kv":
                output_prefix = cloudflare_kv.output_prefix()
                output_key = f"{output_prefix}{batch_id}/masked-{file_id}-{lease_token}.pdf"
                try:
                    await asyncio.to_thread(cloudflare_kv.put_file, output_key, result_path)
                    output_s3_uri = cloudflare_kv.make_uri(output_key)
                except Exception:
                    await complete(
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
                output_key = "/".join(
                    part for part in (output_prefix, batch_id, f"masked-{file_id}-{lease_token}.pdf") if part
                )
                try:
                    await asyncio.to_thread(self.upload_s3_result, output_bucket, output_key, result_path)
                    output_s3_uri = f"s3://{output_bucket}/{output_key}"
                except Exception:
                    await complete(
                        status="failed",
                        error_code="output_upload_failed",
                        error_message="The masked PDF was created locally, but could not be saved to the configured cloud storage location.",
                        output_path=str(result_path),
                        metadata=metadata,
                    )
                    return
            await complete(
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
                await complete(
                    status="rejected" if exc.status_code < 500 else "failed",
                    error_code=detail["code"],
                    error_message=detail["message"],
                )
            elif isinstance(exc, HTTPException):
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                await complete(
                    status="rejected" if exc.status_code < 500 else "failed",
                    error_code=str(detail.get("code", "unsupported_file" if exc.status_code == 415 else "processing_failed")),
                    error_message=str(detail.get("message", "The file could not be processed.")),
                )
            elif isinstance(exc, S3SourceUnavailable):
                await complete(
                    status="failed",
                    error_code="source_unavailable",
                    error_message="The S3 source could not be read. Check the configured bucket access and object key.",
                )
            elif isinstance(exc, CloudflareKVSourceUnavailable):
                await complete(
                    status="failed",
                    error_code="source_unavailable",
                    error_message="The Workers KV source could not be read. Check the configured account, namespace, and token permissions.",
                )
            else:
                await complete(
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
            if temporary_remote_source and item.get("source_bucket") and item.get("source_key"):
                current = await asyncio.to_thread(self.get_file, file_id)
                if current and current["status"] not in {"queued", "processing"}:
                    try:
                        await asyncio.to_thread(
                            self.delete_s3_outputs,
                            [f"s3://{item['source_bucket']}/{item['source_key']}"],
                        )
                    except Exception:
                        # Failed cleanup can be handled by the object-store lifecycle policy.
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
    def materialize_output(cls, uri: str, destination: Path) -> None:
        parsed = urlsplit(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if parsed.scheme == "s3" and parsed.netloc and parsed.path.strip("/"):
            cls._s3_client().download_file(parsed.netloc, parsed.path.lstrip("/"), str(destination))
            return
        if parsed.scheme == "kv" and parsed.netloc and parsed.path.strip("/"):
            if parsed.netloc != cloudflare_kv.namespace_id():
                raise ValueError("Stored Workers KV output refers to a different namespace")
            cloudflare_kv.download_value(
                unquote(parsed.path.lstrip("/")),
                destination,
                cloudflare_kv.KV_MAX_VALUE_BYTES,
            )
            return
        raise ValueError("Stored output location is invalid")

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

