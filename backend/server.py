from __future__ import annotations

import asyncio
import hmac
import json
import mimetypes
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .configuration import load_project_environment

load_project_environment()

from . import cloudflare_kv
from .ai_summary import (
    CardSummaryRequest,
    confidence_evidence_for_result,
    confidence_score,
    pipeline_confidence_score,
    summarize_card_result,
)
from .batches import (
    SUPPORTED_EXTENSIONS,
    BatchManager,
    BatchSeed,
    resolve_input_folder,
    safe_display_name,
    scan_input_folder,
)
from .candidates import OpenCvDocumentProposer
from .detectors import (
    DetectorConfigurationError,
    UltralyticsDocumentSegmenter,
    UltralyticsNumberDetector,
)
from .documents import input_pages, make_flattened_pdf
from .errors import ModelNotReady, PipelineError
from .ocr import configure_cpu_threads, load_ocr_engine
from .pipeline import AadhaarPipeline
from .settings import Settings


# PaddlePaddle 3.3.x had a Windows CPU oneDNN/PIR regression. The project pins
# the tested 3.2.2 runtime and explicitly selects its stable executor.
os.environ.setdefault("FLAGS_enable_pir_api", "0")

_pipeline_lock = threading.Lock()


class FolderBatchRequest(BaseModel):
    relative_folder: str = Field(default=".", max_length=512)
    recursive: bool = False


class S3BatchRequest(BaseModel):
    relative_prefix: str = Field(default="", max_length=512)


def require_batch_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("AADHAAR_BATCH_API_KEY", "").strip()
    if expected and (not x_api_key or not hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")


def serialize_batch(manager: BatchManager, batch_id: str) -> dict[str, Any]:
    result = manager.get_batch(batch_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Batch not found.")
    batch = result["batch"]
    files = []
    for item in result["files"]:
        output_path = item.get("output_path")
        has_local_output = bool(output_path and Path(output_path).is_file())
        files.append({
            "file_id": item["id"],
            "file_name": item["display_name"],
            "status": item["status"],
            "error_code": item["error_code"],
            "error_message": item["error_message"],
            "document_count": item["document_count"],
            "document_classes": json.loads(item["document_classes"] or "[]"),
            "masked_regions": item["masked_regions"],
            "masking_confidence": item["masking_confidence"],
            "masking_confidence_source": item["masking_confidence_source"],
            "output_s3_uri": item["output_s3_uri"],
            "output_storage_uri": item["output_s3_uri"],
            "download_url": (
                f"/api/batches/{batch_id}/files/{item['id']}/download"
                if has_local_output
                else None
            ),
        })
    finished = batch["status"] not in {"queued", "processing"}
    return {
        "batch_id": batch_id,
        "source": batch["source_type"],
        "status": batch["status"],
        "created_at": batch["created_at"],
        "updated_at": batch["updated_at"],
        "total_files": batch["total_files"],
        "completed_files": batch["completed_files"],
        "rejected_files": batch["rejected_files"],
        "failed_files": batch["failed_files"],
        "files": files,
        "download_zip_url": f"/api/batches/{batch_id}/download.zip" if finished else None,
    }


async def _process_batch_input(
    path: Path,
    display_name: str,
    pipeline: AadhaarPipeline,
    settings: Settings,
) -> tuple[bytes, dict[str, Any]]:
    content = await run_in_threadpool(path.read_bytes)
    try:
        pages = await run_in_threadpool(input_pages, content, display_name, settings)
    finally:
        content = b""
    if not pages:
        raise HTTPException(status_code=415, detail="The document contains no readable pages.")
    result = await run_in_threadpool(_process_serial, pipeline, pages)
    flattened_pdf = await run_in_threadpool(make_flattened_pdf, result.documents)
    classes = sorted({document.document_class for document in result.documents})
    minimum_document_confidence = min(document.document_confidence for document in result.documents)
    minimum_verification_confidence = min(document.verification_confidence for document in result.documents)
    document_evidence = confidence_evidence_for_result(result, settings.min_blur_score)
    score_request = CardSummaryRequest(
        outcome="success",
        document_count=len(result.documents),
        masked_regions=result.masked_regions,
        document_classes=classes,
        document_confidence=minimum_document_confidence,
        verification_confidence=minimum_verification_confidence,
        pipeline_score=pipeline_confidence_score(result),
        document_evidence=document_evidence,
    )
    confidence_assessment = await run_in_threadpool(summarize_card_result, score_request)
    return flattened_pdf, {
        "document_count": len(result.documents),
        "document_classes": classes,
        "masked_regions": result.masked_regions,
        "masking_confidence": confidence_assessment["confidence_score"],
        "masking_confidence_source": confidence_assessment["confidence_source"],
    }


def _load_pipeline(settings: Settings) -> AadhaarPipeline:
    # Configure OpenMP/MKL before Ultralytics imports PyTorch. The same
    # bounded setting is used by PaddleOCR later in pipeline initialization.
    configure_cpu_threads()
    document_segmenter = UltralyticsDocumentSegmenter(
        settings.document_model_path,
        device=settings.yolo_device,
        confidence=settings.document_confidence,
        iou=settings.document_iou,
        image_size=settings.inference_size,
        fallback_confidence=settings.document_fallback_confidence,
    )
    number_detector = UltralyticsNumberDetector(
        settings.number_model_path,
        device=settings.yolo_device,
        confidence=settings.number_confidence,
        iou=settings.document_iou,
        image_size=settings.inference_size,
        fallback_confidence=settings.number_fallback_confidence,
    )
    ocr_engine = load_ocr_engine()
    return AadhaarPipeline(
        document_segmenter,
        number_detector,
        ocr_engine,
        settings,
        document_proposer=OpenCvDocumentProposer(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_environment()
    app.state.settings = settings
    app.state.pipeline = None
    app.state.engine_status = "loading"
    app.state.engine_error = None
    batch_manager = BatchManager()
    app.state.batch_manager = batch_manager

    async def initialize() -> None:
        missing = [
            path.name
            for path in (settings.document_model_path, settings.number_model_path)
            if not path.is_file()
        ]
        if missing:
            app.state.engine_status = "models_missing"
            app.state.engine_error = (
                "Missing trained model weights: " + ", ".join(missing) + ". See training/README.md."
            )
            return
        try:
            app.state.pipeline = await run_in_threadpool(_load_pipeline, settings)
            app.state.engine_status = "ready"
        except DetectorConfigurationError as exc:
            app.state.engine_status = "model_error"
            app.state.engine_error = str(exc)
        except Exception as exc:
            app.state.engine_status = "error"
            app.state.engine_error = f"{type(exc).__name__}: {exc}"

    task = asyncio.create_task(initialize())
    batch_worker = asyncio.create_task(batch_manager.run(app, _process_batch_input))
    try:
        yield
    finally:
        batch_manager.request_stop()
        await batch_worker
        if not task.done():
            task.cancel()


app = FastAPI(
    title="Private Aadhaar masking processor",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    settings: Settings = app.state.settings
    return {
        "ready": app.state.pipeline is not None,
        "status": app.state.engine_status,
        "engine": "YOLO segmentation + verified OpenCV fallback + PaddleOCR",
        "detail": app.state.engine_error,
        "components": {
            "document_detector": {
                "ready": settings.document_model_path.is_file(),
                "type": "YOLO instance segmentation",
                "weight": settings.document_model_path.name,
            },
            "number_detector": {
                "ready": settings.number_model_path.is_file(),
                "type": "YOLO object detection",
                "weight": settings.number_model_path.name,
            },
            "ocr": {
                "ready": app.state.pipeline is not None,
                "type": "PaddleOCR word-position OCR",
            },
            "geometric_fallback": {
                "ready": app.state.pipeline is not None,
                "type": "OpenCV quadrilateral proposals with strict Aadhaar verification",
            },
        },
        "cloud_storage": {
            "cloudflare_r2_configured": BatchManager.cloudflare_r2_configured(),
            "cloudflare_workers_kv_configured": cloudflare_kv.configured()
            and cloudflare_kv.prefixes_are_separate(),
        },
    }


@app.post("/api/card-summary")
async def card_summary(request: CardSummaryRequest) -> dict[str, Any]:
    return await run_in_threadpool(summarize_card_result, request)


def _process_serial(pipeline: AadhaarPipeline, pages: list[Any]):
    with _pipeline_lock:
        return pipeline.process(pages)


@app.post("/api/process")
async def process_document(file: UploadFile = File(...)) -> Response:
    pipeline: AadhaarPipeline | None = app.state.pipeline
    if pipeline is None:
        error = ModelNotReady(app.state.engine_error or None)
        raise HTTPException(error.status_code, error.detail())

    settings: Settings = app.state.settings
    filename = Path(file.filename or "upload").name
    content = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"Please upload a file smaller than {settings.max_upload_bytes // (1024 * 1024)} MB.")

    try:
        pages = await run_in_threadpool(input_pages, content, filename, settings)
        if not pages:
            raise HTTPException(415, "The document contains no readable pages.")
        result = await run_in_threadpool(_process_serial, pipeline, pages)
        flattened_pdf = await run_in_threadpool(make_flattened_pdf, result.documents)
    except PipelineError as exc:
        raise HTTPException(exc.status_code, exc.detail()) from exc
    finally:
        content = b""

    classes = sorted({document.document_class for document in result.documents})
    minimum_document_confidence = min(document.document_confidence for document in result.documents)
    minimum_verification_confidence = min(document.verification_confidence for document in result.documents)
    minimum_boundary_confidence = min(document.boundary_confidence for document in result.documents)
    minimum_scan_quality = min(document.scan_quality_score for document in result.documents)
    minimum_ocr_confidence = min(document.ocr_confidence for document in result.documents)
    number_detector_scores = [
        document.number_detector_confidence
        for document in result.documents
        if document.number_detector_confidence is not None
    ]
    minimum_number_detector_confidence = min(number_detector_scores) if number_detector_scores else None
    pipeline_score = pipeline_confidence_score(result)
    document_evidence = confidence_evidence_for_result(result, settings.min_blur_score)
    score_request = CardSummaryRequest(
        outcome="success",
        document_count=len(result.documents),
        masked_regions=result.masked_regions,
        document_classes=classes,
        document_confidence=minimum_document_confidence,
        verification_confidence=minimum_verification_confidence,
        pipeline_score=pipeline_score,
        document_evidence=document_evidence,
    )
    return Response(
        content=flattened_pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="masked-aadhaar.pdf"',
            "X-Aadhaar-Documents": str(len(result.documents)),
            "X-Redacted-Regions": str(result.masked_regions),
            "X-Document-Classes": ",".join(classes),
            "X-Document-Confidence": f"{minimum_document_confidence:.3f}",
            "X-Boundary-Confidence": f"{minimum_boundary_confidence:.3f}",
            "X-Scan-Quality-Score": f"{minimum_scan_quality:.3f}",
            "X-OCR-Confidence": f"{minimum_ocr_confidence:.3f}",
            "X-Verification-Confidence": f"{minimum_verification_confidence:.3f}",
            "X-Number-Localization-Confidence": (
                f"{minimum_number_detector_confidence:.3f}"
                if minimum_number_detector_confidence is not None
                else "not_matched"
            ),
            "X-Masking-Confidence": str(confidence_score(score_request)),
            "X-Confidence-Evidence": json.dumps(document_evidence, separators=(",", ":")),
            "Cache-Control": "no-store, private",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _require_pipeline() -> None:
    if app.state.pipeline is None:
        error = ModelNotReady(app.state.engine_error or None)
        raise HTTPException(error.status_code, error.detail())


def _check_batch_capacity(manager: BatchManager, queued_files: int) -> None:
    if manager.pending_count() + queued_files > manager.max_pending:
        raise HTTPException(
            status_code=429,
            detail="The CPU batch queue is full. Wait for current files to finish before submitting another batch.",
        )


def _prefixes_overlap(first: str, second: str) -> bool:
    first = first.strip("/")
    second = second.strip("/")
    return bool(first and second) and (
        first == second or first.startswith(second + "/") or second.startswith(first + "/")
    )


def _file_result_response(manager: BatchManager, batch_id: str, file_id: str) -> FileResponse:
    item = manager.get_file(file_id)
    if item is None or item["batch_id"] != batch_id:
        raise HTTPException(status_code=404, detail="Batch file not found.")
    output_path = item.get("output_path")
    if not output_path or not Path(output_path).is_file():
        if item["status"] in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail="This file is still processing.")
        raise HTTPException(status_code=404, detail="No masked PDF was created for this file.")
    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"masked-{file_id}.pdf",
        headers={"Cache-Control": "no-store, private", "X-Content-Type-Options": "nosniff"},
    )


@app.post("/api/batches", status_code=202, dependencies=[Depends(require_batch_api_key)])
async def create_upload_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Queue a batch of uploaded PDFs/images; each input is processed on CPU, one at a time."""
    try:
        _require_pipeline()
        manager: BatchManager = app.state.batch_manager
        settings: Settings = app.state.settings
        if not files:
            raise HTTPException(status_code=400, detail="Choose at least one supported file.")
        if len(files) > manager.max_files:
            raise HTTPException(status_code=413, detail=f"A batch may contain at most {manager.max_files} files.")
        _check_batch_capacity(manager, len(files))

        prepared: list[tuple[UploadFile, str, str]] = []
        for uploaded in files:
            display_name = safe_display_name(uploaded.filename)
            suffix = Path(display_name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=415, detail="Every batch file must be a supported image, PDF, or Word document.")
            prepared.append((uploaded, display_name, suffix))

        batch_id = manager.new_id()
        manager.prepare_batch_dir(batch_id)
        seeds: list[BatchSeed] = []
        batch_bytes = 0
        try:
            for uploaded, display_name, suffix in prepared:
                file_id = manager.new_id()
                destination = manager.staged_path(batch_id, file_id, suffix)
                file_bytes = 0
                with destination.open("xb") as output:
                    while chunk := await uploaded.read(1024 * 1024):
                        file_bytes += len(chunk)
                        batch_bytes += len(chunk)
                        if file_bytes > settings.max_upload_bytes:
                            raise HTTPException(status_code=413, detail="A batch file exceeds the per-file size limit.")
                        if batch_bytes > manager.max_total_bytes:
                            raise HTTPException(status_code=413, detail="The batch exceeds the configured total upload size limit.")
                        output.write(chunk)
                if file_bytes == 0:
                    raise HTTPException(status_code=400, detail="Empty files cannot be processed.")
                seeds.append(BatchSeed(
                    file_id=file_id,
                    display_name=display_name,
                    source_type="upload",
                    input_path=str(destination),
                    source_size=file_bytes,
                ))
            manager.create_batch(batch_id, "upload", seeds)
        except Exception:
            manager.discard_workspace(batch_id)
            raise
        await manager.submit(batch_id)
        return serialize_batch(manager, batch_id)
    finally:
        for uploaded in files:
            await uploaded.close()


@app.post("/api/batches/folder", status_code=202, dependencies=[Depends(require_batch_api_key)])
async def create_folder_batch(request: FolderBatchRequest) -> dict[str, Any]:
    """Copy supported files from a configured input folder into a CPU batch."""
    _require_pipeline()
    manager: BatchManager = app.state.batch_manager
    settings: Settings = app.state.settings
    try:
        folder = resolve_input_folder(manager.input_root, request.relative_folder)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sources = await run_in_threadpool(scan_input_folder, folder, recursive=request.recursive)
    if not sources:
        raise HTTPException(status_code=400, detail="No supported files were found in that configured folder.")
    if len(sources) > manager.max_files:
        raise HTTPException(status_code=413, detail=f"A batch may contain at most {manager.max_files} files.")

    accepted_size = sum(path.stat().st_size for path in sources if path.stat().st_size <= settings.max_upload_bytes)
    if accepted_size > manager.max_total_bytes:
        raise HTTPException(status_code=413, detail="The folder batch exceeds the configured total upload size limit.")
    queued_count = sum(path.stat().st_size <= settings.max_upload_bytes for path in sources)
    _check_batch_capacity(manager, queued_count)

    batch_id = manager.new_id()
    manager.prepare_batch_dir(batch_id)
    seeds: list[BatchSeed] = []
    try:
        for source in sources:
            file_id = manager.new_id()
            display_name = safe_display_name(source.name)
            size = source.stat().st_size
            if size > settings.max_upload_bytes:
                seeds.append(BatchSeed(
                    file_id=file_id,
                    display_name=display_name,
                    source_type="folder",
                    source_size=size,
                    initial_status="rejected",
                    error_code="file_too_large",
                    error_message="This file exceeds the per-file size limit.",
                ))
                continue
            staged = manager.staged_path(batch_id, file_id, source.suffix.lower())
            await run_in_threadpool(shutil.copyfile, source, staged)
            seeds.append(BatchSeed(
                file_id=file_id,
                display_name=display_name,
                source_type="folder",
                input_path=str(staged),
                source_size=size,
            ))
        manager.create_batch(batch_id, "folder", seeds)
    except Exception:
        manager.discard_workspace(batch_id)
        raise
    await manager.submit(batch_id)
    return serialize_batch(manager, batch_id)


@app.post("/api/batches/r2", status_code=202, dependencies=[Depends(require_batch_api_key)])
@app.post("/api/batches/s3", status_code=202, dependencies=[Depends(require_batch_api_key)])
async def create_s3_batch(request: S3BatchRequest) -> dict[str, Any]:
    """Queue files under a configured S3-compatible input prefix."""
    _require_pipeline()
    manager: BatchManager = app.state.batch_manager
    settings: Settings = app.state.settings
    r2_bucket = os.getenv("CLOUDFLARE_R2_BUCKET", "").strip()
    bucket = os.getenv("AADHAAR_S3_INPUT_BUCKET", "").strip() or r2_bucket
    default_input_prefix = os.getenv("CLOUDFLARE_R2_INPUT_PREFIX", "aadhaar/inbox") if r2_bucket else ""
    base_prefix = os.getenv("AADHAAR_S3_INPUT_PREFIX", default_input_prefix).strip("/")
    relative_prefix = request.relative_prefix.replace("\\", "/").strip("/")
    if not bucket or not base_prefix:
        raise HTTPException(
            status_code=503,
            detail="Configure a private S3/R2 bucket and a non-empty input prefix before importing cloud files.",
        )
    if any(part == ".." for part in relative_prefix.split("/")) or relative_prefix.startswith("/"):
        raise HTTPException(status_code=400, detail="The S3 prefix must stay inside the configured input prefix.")
    output_bucket = os.getenv("AADHAAR_S3_OUTPUT_BUCKET", "").strip() or r2_bucket
    default_output_prefix = os.getenv("CLOUDFLARE_R2_OUTPUT_PREFIX", "aadhaar/masked") if r2_bucket else "masked"
    output_prefix = os.getenv("AADHAAR_S3_OUTPUT_PREFIX", "").strip("/") or default_output_prefix.strip("/")
    if (
        output_bucket == bucket
        and output_prefix
        and (output_prefix == base_prefix or output_prefix.startswith(base_prefix + "/"))
    ):
        raise HTTPException(
            status_code=400,
            detail="The S3 output prefix must be outside the configured S3 input prefix.",
        )
    prefix = "/".join(part for part in (base_prefix, relative_prefix) if part).rstrip("/") + "/"
    try:
        objects = await run_in_threadpool(BatchManager.list_s3_objects, bucket, prefix, manager.max_files)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not list files from the configured S3 prefix.") from exc
    if not objects:
        raise HTTPException(status_code=404, detail="No supported files were found under that S3 prefix.")
    if len(objects) > manager.max_files:
        raise HTTPException(status_code=413, detail=f"A batch may contain at most {manager.max_files} files.")
    total_bytes = sum(item["size"] for item in objects if item["size"] <= settings.max_upload_bytes)
    if total_bytes > manager.max_total_bytes:
        raise HTTPException(status_code=413, detail="The S3 batch exceeds the configured total upload size limit.")
    queued_count = sum(item["size"] <= settings.max_upload_bytes for item in objects)
    _check_batch_capacity(manager, queued_count)

    source_type = "r2" if bucket == r2_bucket and r2_bucket else "s3"
    batch_id = manager.new_id()
    manager.prepare_batch_dir(batch_id)
    seeds = []
    for item in objects:
        file_id = manager.new_id()
        display_name = safe_display_name(Path(item["key"]).name)
        if item["size"] > settings.max_upload_bytes:
            seeds.append(BatchSeed(
                file_id=file_id,
                display_name=display_name,
                source_type=source_type,
                source_bucket=bucket,
                source_key=item["key"],
                source_size=item["size"],
                initial_status="rejected",
                error_code="file_too_large",
                error_message="This file exceeds the per-file size limit.",
            ))
        else:
            seeds.append(BatchSeed(
                file_id=file_id,
                display_name=display_name,
                source_type=source_type,
                source_bucket=bucket,
                source_key=item["key"],
                source_size=item["size"],
            ))
    try:
        manager.create_batch(batch_id, source_type, seeds)
    except Exception:
        manager.discard_workspace(batch_id)
        raise
    await manager.submit(batch_id)
    return serialize_batch(manager, batch_id)


@app.post("/api/batches/r2/upload-folder", status_code=202, dependencies=[Depends(require_batch_api_key)])
async def upload_local_folder_to_r2(request: FolderBatchRequest) -> dict[str, Any]:
    """Copy a configured local inbox to private Cloudflare R2, then process it."""
    _require_pipeline()
    manager: BatchManager = app.state.batch_manager
    settings: Settings = app.state.settings
    if not BatchManager.cloudflare_r2_configured():
        raise HTTPException(
            status_code=503,
            detail="Cloudflare R2 is not ready. Configure a private bucket, account ID, scoped R2 keys, and install backend/requirements-s3.txt.",
        )

    bucket = os.getenv("CLOUDFLARE_R2_BUCKET", "").strip()
    inbox_prefix = os.getenv("CLOUDFLARE_R2_INPUT_PREFIX", "aadhaar/inbox").strip("/")
    upload_prefix = os.getenv("CLOUDFLARE_R2_UPLOAD_PREFIX", "aadhaar/uploads").strip("/")
    default_output_prefix = os.getenv("CLOUDFLARE_R2_OUTPUT_PREFIX", "aadhaar/masked").strip("/")
    output_bucket = os.getenv("AADHAAR_S3_OUTPUT_BUCKET", "").strip() or bucket
    output_prefix = os.getenv("AADHAAR_S3_OUTPUT_PREFIX", "").strip("/") or default_output_prefix
    if not all((inbox_prefix, upload_prefix, output_prefix)):
        raise HTTPException(status_code=503, detail="R2 input, upload, and output prefixes must be non-empty.")
    if (
        _prefixes_overlap(inbox_prefix, upload_prefix)
        or (output_bucket == bucket and _prefixes_overlap(inbox_prefix, output_prefix))
        or (output_bucket == bucket and _prefixes_overlap(upload_prefix, output_prefix))
    ):
        raise HTTPException(
            status_code=400,
            detail="Use separate, non-overlapping R2 prefixes for inbox imports, folder uploads, and masked outputs.",
        )

    try:
        folder = resolve_input_folder(manager.input_root, request.relative_folder)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sources = await run_in_threadpool(scan_input_folder, folder, recursive=request.recursive)
    if not sources:
        raise HTTPException(status_code=400, detail="No supported files were found in the configured local inbox.")
    if len(sources) > manager.max_files:
        raise HTTPException(status_code=413, detail=f"A batch may contain at most {manager.max_files} files.")

    valid_sources = [path for path in sources if path.stat().st_size <= settings.max_upload_bytes]
    if sum(path.stat().st_size for path in valid_sources) > manager.max_total_bytes:
        raise HTTPException(status_code=413, detail="The folder exceeds the configured total upload size limit.")
    _check_batch_capacity(manager, len(valid_sources))

    batch_id = manager.new_id()
    manager.prepare_batch_dir(batch_id)
    seeds: list[BatchSeed] = []
    uploaded_keys: list[str] = []
    total_bytes = 0
    try:
        for source in sources:
            file_id = manager.new_id()
            display_name = safe_display_name(source.name)
            size = source.stat().st_size
            if size > settings.max_upload_bytes:
                seeds.append(BatchSeed(
                    file_id=file_id,
                    display_name=display_name,
                    source_type="r2",
                    source_bucket=bucket,
                    source_key=None,
                    source_size=size,
                    initial_status="rejected",
                    error_code="file_too_large",
                    error_message="This file exceeds the per-file size limit.",
                ))
                continue

            staged = manager.staged_path(batch_id, file_id, source.suffix.lower())
            await run_in_threadpool(shutil.copyfile, source, staged)
            actual_size = staged.stat().st_size
            total_bytes += actual_size
            if actual_size > settings.max_upload_bytes or total_bytes > manager.max_total_bytes:
                raise HTTPException(status_code=413, detail="The folder changed during upload and exceeded the configured size limit.")

            object_key = f"{upload_prefix}/{batch_id}/{file_id}{source.suffix.lower()}"
            content_type = mimetypes.guess_type(display_name)[0] or "application/octet-stream"
            try:
                await run_in_threadpool(
                    BatchManager.upload_s3_object,
                    bucket,
                    object_key,
                    staged,
                    content_type,
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail="Could not store a source file in private Cloudflare R2.") from exc
            uploaded_keys.append(object_key)
            staged.unlink(missing_ok=True)
            seeds.append(BatchSeed(
                file_id=file_id,
                display_name=display_name,
                source_type="r2",
                source_bucket=bucket,
                source_key=object_key,
                source_size=actual_size,
            ))
        manager.create_batch(batch_id, "r2_upload", seeds)
    except Exception:
        manager.discard_workspace(batch_id)
        if uploaded_keys:
            try:
                await run_in_threadpool(
                    BatchManager.delete_s3_outputs,
                    [f"s3://{bucket}/{key}" for key in uploaded_keys],
                )
            except Exception:
                pass
        raise

    await manager.submit(batch_id)
    return serialize_batch(manager, batch_id)


def _require_workers_kv() -> None:
    if not cloudflare_kv.configured():
        raise HTTPException(
            status_code=503,
            detail="Workers KV is not configured. Add the Cloudflare account ID, namespace ID, and scoped API token to the backend .env file.",
        )
    if not cloudflare_kv.prefixes_are_separate():
        raise HTTPException(
            status_code=503,
            detail="Workers KV input and masked-output prefixes must be separate and non-overlapping.",
        )


@app.post("/api/batches/kv/upload-folder", status_code=202, dependencies=[Depends(require_batch_api_key)])
async def upload_local_folder_to_workers_kv(request: FolderBatchRequest) -> dict[str, Any]:
    """Store configured local inbox originals in Workers KV, then process them on this computer."""
    _require_pipeline()
    _require_workers_kv()
    manager: BatchManager = app.state.batch_manager
    settings: Settings = app.state.settings
    try:
        folder = resolve_input_folder(manager.input_root, request.relative_folder)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sources = await run_in_threadpool(scan_input_folder, folder, recursive=request.recursive)
    if not sources:
        raise HTTPException(status_code=400, detail="No supported files were found in the configured local inbox.")
    if len(sources) > manager.max_files:
        raise HTTPException(status_code=413, detail=f"A batch may contain at most {manager.max_files} files.")

    accepted = [path for path in sources if path.stat().st_size <= settings.max_upload_bytes]
    if sum(path.stat().st_size for path in accepted) > manager.max_total_bytes:
        raise HTTPException(status_code=413, detail="The folder exceeds the configured total upload size limit.")
    _check_batch_capacity(manager, len(accepted))

    batch_id = manager.new_id()
    manager.prepare_batch_dir(batch_id)
    seeds: list[BatchSeed] = []
    uploaded_keys: list[str] = []
    total_bytes = 0
    input_prefix = cloudflare_kv.input_prefix()
    try:
        for source in sources:
            file_id = manager.new_id()
            display_name = safe_display_name(source.name)
            size = source.stat().st_size
            if size > settings.max_upload_bytes or size > cloudflare_kv.KV_MAX_VALUE_BYTES:
                seeds.append(BatchSeed(
                    file_id=file_id,
                    display_name=display_name,
                    source_type="cloudflare_kv",
                    source_size=size,
                    initial_status="rejected",
                    error_code="file_too_large",
                    error_message="This file exceeds the Workers KV value or per-file size limit.",
                ))
                continue

            staged = manager.staged_path(batch_id, file_id, source.suffix.lower())
            await run_in_threadpool(shutil.copyfile, source, staged)
            actual_size = staged.stat().st_size
            total_bytes += actual_size
            if actual_size > settings.max_upload_bytes or actual_size > cloudflare_kv.KV_MAX_VALUE_BYTES:
                raise HTTPException(status_code=413, detail="A file changed during upload and exceeded the configured size limit.")
            if total_bytes > manager.max_total_bytes:
                raise HTTPException(status_code=413, detail="The folder exceeds the configured total upload size limit.")

            key = f"{input_prefix}{batch_id}/{file_id}{source.suffix.lower()}"
            try:
                await run_in_threadpool(cloudflare_kv.put_file, key, staged)
            except Exception as exc:
                raise HTTPException(status_code=502, detail="Could not store a source file in Workers KV. Check the namespace and token permissions.") from exc
            uploaded_keys.append(key)
            seeds.append(BatchSeed(
                file_id=file_id,
                display_name=display_name,
                source_type="cloudflare_kv",
                input_path=str(staged),
                source_key=key,
                source_size=actual_size,
            ))

        manager.create_batch(batch_id, "workers_kv_upload", seeds)
    except Exception:
        manager.discard_workspace(batch_id)
        for key in uploaded_keys:
            try:
                await run_in_threadpool(cloudflare_kv.delete_value, key)
            except Exception:
                pass
        raise

    await manager.submit(batch_id)
    return serialize_batch(manager, batch_id)


@app.post("/api/batches/kv/store-folder", status_code=201, dependencies=[Depends(require_batch_api_key)])
async def store_local_folder_in_workers_kv(request: FolderBatchRequest) -> dict[str, Any]:
    """Store supported local inbox files in Workers KV without starting processing."""
    _require_workers_kv()
    manager: BatchManager = app.state.batch_manager
    settings: Settings = app.state.settings
    try:
        folder = resolve_input_folder(manager.input_root, request.relative_folder)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sources = await run_in_threadpool(scan_input_folder, folder, recursive=request.recursive)
    if not sources:
        raise HTTPException(status_code=400, detail="No supported files were found in the configured local inbox.")
    if len(sources) > manager.max_files:
        raise HTTPException(status_code=413, detail=f"A stored upload may contain at most {manager.max_files} files.")
    max_file_bytes = min(settings.max_upload_bytes, cloudflare_kv.KV_MAX_VALUE_BYTES)
    if any(source.stat().st_size > max_file_bytes for source in sources):
        raise HTTPException(status_code=413, detail="A file exceeds the Workers KV 25 MiB value limit or the configured per-file limit.")
    if sum(source.stat().st_size for source in sources) > manager.max_total_bytes:
        raise HTTPException(status_code=413, detail="The folder exceeds the configured total upload size limit.")

    upload_id = manager.new_id()
    relative_prefix = f"manual/{upload_id}"
    prefix = f"{cloudflare_kv.input_prefix()}{relative_prefix}/"
    stored_keys: list[str] = []
    stored_names: list[str] = []
    try:
        for index, source in enumerate(sources, start=1):
            suffix = source.suffix.lower()
            stored_name = f"document-{index:02d}{suffix}"
            key = f"{prefix}{stored_name}"
            size = source.stat().st_size
            try:
                await run_in_threadpool(
                    cloudflare_kv.put_file,
                    key,
                    source,
                    {"display_name": stored_name, "size": size},
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail="Could not store a file in Workers KV. Check the namespace and token permissions.") from exc
            stored_keys.append(key)
            stored_names.append(stored_name)

        return {
            "status": "stored",
            "upload_id": upload_id,
            "relative_prefix": relative_prefix,
            "storage_prefix": prefix,
            "uploaded_count": len(stored_keys),
            "files": stored_names,
        }
    except Exception:
        for key in stored_keys:
            try:
                await run_in_threadpool(cloudflare_kv.delete_value, key)
            except Exception:
                pass
        raise


@app.post("/api/batches/kv", status_code=202, dependencies=[Depends(require_batch_api_key)])
async def create_workers_kv_batch(request: S3BatchRequest) -> dict[str, Any]:
    """Import supported files stored under the configured Workers KV inbox prefix."""
    _require_pipeline()
    _require_workers_kv()
    manager: BatchManager = app.state.batch_manager
    settings: Settings = app.state.settings
    input_prefix = cloudflare_kv.input_prefix()
    relative_prefix = request.relative_prefix.replace("\\", "/").strip("/")
    if any(part == ".." for part in relative_prefix.split("/")) or request.relative_prefix.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="The Workers KV prefix must stay inside the configured inbox prefix.")
    prefix = input_prefix + (relative_prefix + "/" if relative_prefix else "")
    try:
        key_records = await run_in_threadpool(cloudflare_kv.list_keys, prefix, manager.max_files * 10)
    except cloudflare_kv.CloudflareKVError as exc:
        raise HTTPException(status_code=502, detail="Could not list files from the configured Workers KV inbox. Check the API token permissions.") from exc
    key_records = [
        record for record in key_records
        if Path(str(record.get("name", ""))).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not key_records:
        raise HTTPException(status_code=404, detail="No supported files were found under the Workers KV inbox prefix.")
    if len(key_records) > manager.max_files:
        raise HTTPException(status_code=413, detail=f"A batch may contain at most {manager.max_files} files.")
    _check_batch_capacity(manager, len(key_records))

    batch_id = manager.new_id()
    manager.prepare_batch_dir(batch_id)
    seeds: list[BatchSeed] = []
    total_bytes = 0
    try:
        for record in key_records:
            key = str(record["name"])
            file_id = manager.new_id()
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            display_name = safe_display_name(str(metadata.get("display_name") or Path(key).name))
            suffix = Path(key).suffix.lower()
            staged = manager.staged_path(batch_id, file_id, suffix)
            try:
                size = await run_in_threadpool(
                    cloudflare_kv.download_value,
                    key,
                    staged,
                    min(settings.max_upload_bytes, cloudflare_kv.KV_MAX_VALUE_BYTES),
                )
            except cloudflare_kv.CloudflareKVValueTooLarge:
                seeds.append(BatchSeed(
                    file_id=file_id,
                    display_name=display_name,
                    source_type="cloudflare_kv",
                    source_key=key,
                    source_size=cloudflare_kv.KV_MAX_VALUE_BYTES + 1,
                    initial_status="rejected",
                    error_code="file_too_large",
                    error_message="This file exceeds the Workers KV value or per-file size limit.",
                ))
                continue
            except cloudflare_kv.CloudflareKVError as exc:
                raise HTTPException(status_code=502, detail="Could not read a file from the Workers KV inbox. Check the API token permissions.") from exc
            if total_bytes + size > manager.max_total_bytes:
                raise HTTPException(status_code=413, detail="The Workers KV inbox exceeds the configured total batch size limit.")
            total_bytes += size
            seeds.append(BatchSeed(
                file_id=file_id,
                display_name=display_name,
                source_type="cloudflare_kv",
                input_path=str(staged),
                source_key=key,
                source_size=size,
            ))
        manager.create_batch(batch_id, "workers_kv", seeds)
    except Exception:
        manager.discard_workspace(batch_id)
        raise

    await manager.submit(batch_id)
    return serialize_batch(manager, batch_id)


@app.get("/api/batches/{batch_id}", dependencies=[Depends(require_batch_api_key)])
async def get_batch_status(batch_id: str) -> dict[str, Any]:
    return serialize_batch(app.state.batch_manager, batch_id)


@app.get("/api/batches/{batch_id}/files/{file_id}/download", dependencies=[Depends(require_batch_api_key)])
async def download_batch_file(batch_id: str, file_id: str) -> FileResponse:
    manager: BatchManager = app.state.batch_manager
    try:
        return _file_result_response(manager, batch_id, file_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid batch or file identifier.") from exc


@app.get("/api/batches/{batch_id}/download.zip", dependencies=[Depends(require_batch_api_key)])
async def download_batch_archive(batch_id: str) -> FileResponse:
    manager: BatchManager = app.state.batch_manager
    try:
        manager.batch_dir(batch_id)
        archive = await run_in_threadpool(manager.create_archive, batch_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid batch identifier.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(
        archive,
        media_type="application/zip",
        filename=f"masked-batch-{batch_id}.zip",
        headers={"Cache-Control": "no-store, private", "X-Content-Type-Options": "nosniff"},
    )


@app.delete("/api/batches/{batch_id}", status_code=204, dependencies=[Depends(require_batch_api_key)])
async def delete_batch(batch_id: str) -> Response:
    manager: BatchManager = app.state.batch_manager
    try:
        manager.batch_dir(batch_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid batch identifier.") from exc
    try:
        deleted = manager.delete_batch(batch_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not delete the configured S3 output. The local batch was kept; retry after checking S3 access.",
        ) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Batch not found.")
    return Response(status_code=204)

