from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from .candidates import OpenCvDocumentProposer
from .detectors import (
    DetectorConfigurationError,
    UltralyticsDocumentSegmenter,
    UltralyticsNumberDetector,
)
from .documents import input_pages, make_flattened_pdf
from .errors import ModelNotReady, PipelineError
from .ocr import load_ocr_engine
from .pipeline import AadhaarPipeline
from .settings import Settings


# PaddlePaddle 3.3.x had a Windows CPU oneDNN/PIR regression. The project pins
# the tested 3.2.2 runtime and explicitly selects its stable executor.
os.environ.setdefault("FLAGS_enable_pir_api", "0")

_pipeline_lock = threading.Lock()


def _load_pipeline(settings: Settings) -> AadhaarPipeline:
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
    try:
        yield
    finally:
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
    }


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
    return Response(
        content=flattened_pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="masked-aadhaar.pdf"',
            "X-Aadhaar-Documents": str(len(result.documents)),
            "X-Redacted-Regions": str(result.masked_regions),
            "X-Document-Classes": ",".join(classes),
            "X-Document-Confidence": f"{minimum_document_confidence:.3f}",
            "X-Verification-Confidence": f"{minimum_verification_confidence:.3f}",
            "Cache-Control": "no-store, private",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )

