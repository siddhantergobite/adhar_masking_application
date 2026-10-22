from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .domain import ProcessingResult

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # The API can still start and use its local summary fallback.
    pass


DocumentClass = Literal["aadhaar_front", "aadhaar_back", "aadhaar_letter"]
ReasonCode = Literal[
    "no_aadhaar_detected",
    "review_required",
    "no_aadhaar_number",
    "models_not_ready",
    "ocr_failed",
    "processing_failed",
    "processing_timeout",
    "unsupported_file",
    "file_too_large",
]


class CardSummaryRequest(BaseModel):
    """Allow only non-identifying processing metadata into the summary prompt."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "rejected", "unavailable"]
    document_count: int = Field(default=0, ge=0, le=1000)
    masked_regions: int = Field(default=0, ge=0, le=1000)
    document_classes: list[DocumentClass] = Field(default_factory=list, max_length=8)
    document_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    verification_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    pipeline_score: Optional[int] = Field(default=None, ge=0, le=100)
    reason_code: Optional[ReasonCode] = None


def confidence_score(request: CardSummaryRequest) -> int | None:
    """Return the backend-computed score, never an accuracy probability."""
    if request.outcome != "success":
        # A rejected or unfinished request has no masked result to score.
        return None
    return request.pipeline_score


def pipeline_confidence_score(result: ProcessingResult) -> int | None:
    """Combine evidence per document after all mandatory safety gates pass.

    The number detector is corroborating evidence; OCR word boxes supply the
    actual mask coordinates. If the detector did not match, score that missing
    corroboration neutrally rather than treating it as proof of failure. The
    overall file uses its weakest document so a weaker page is not averaged
    away. This is an evidence score, not a calibrated probability.
    """
    weights = (
        ("boundary_confidence", 0.30),
        ("scan_quality_score", 0.10),
        ("ocr_confidence", 0.25),
        ("verification_confidence", 0.25),
    )
    no_localization_match = 0.50
    document_scores: list[float] = []
    for document in result.documents:
        score = sum(
            _bounded_score(getattr(document, name)) * weight
            for name, weight in weights
        )
        localization = (
            no_localization_match
            if document.number_detector_confidence is None
            else _bounded_score(document.number_detector_confidence)
        )
        score += localization * 0.10
        document_scores.append(score)

    if not document_scores:
        return None
    return round(100 * min(document_scores))


def _bounded_score(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _local_summary(request: CardSummaryRequest, score: int | None) -> str:
    if request.outcome == "success":
        docs = f"{request.document_count} Aadhaar document{'s' if request.document_count != 1 else ''}"
        regions = f"{request.masked_regions} number line{'s' if request.masked_regions != 1 else ''}"
        if score is None:
            return f"The local checks created a masked copy for {docs} and redacted {regions}. A confidence score could not be calculated."
        return f"The local checks accepted a masked copy for {docs} and redacted {regions}. The evidence-weighted masking confidence score is {score}/100."
    if request.outcome == "rejected":
        return "No safe masked copy was produced, so this file needs review or a clearer upload. No confidence score is shown because there is no masked result to assess."
    return "Processing did not finish, so confidence could not be measured. Review the processing message before trying again."


def _two_sentence_summary(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip().strip('"')
    if not compact:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    return " ".join(sentences[:2])[:360].strip()


def _azure_base_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/openai/v1"):
        return base + "/"
    return base + "/openai/v1/"


def summarize_card_result(request: CardSummaryRequest) -> dict[str, int | str | None]:
    score = confidence_score(request)
    fallback = _local_summary(request, score)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if not endpoint or not api_key:
        return {"summary": fallback, "source": "local", "confidence_score": score}

    metadata = {
        "outcome": request.outcome,
        "document_count": request.document_count,
        "masked_regions": request.masked_regions,
        "document_classes": request.document_classes,
        "document_detection_confidence": request.document_confidence,
        "mask_verification_confidence": request.verification_confidence,
        "safe_reason_code": request.reason_code,
    }
    prompt = (
        "Write a concise, honest 1-2 sentence summary for an operator reviewing an Aadhaar masking result. "
        "Use only the supplied processing metadata. Never claim perfect masking, infer document contents, "
        "or output a confidence score; the application displays its own evidence-weighted score. "
        "For rejected or unavailable results, clearly say that no score or safe output is available as appropriate.\n\n"
        f"Processing metadata: {json.dumps(metadata, separators=(',', ':'))}"
    )

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=_azure_base_url(endpoint),
            timeout=8.0,
            max_retries=0,
        )
        response = client.responses.create(
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-5-nano").strip() or "gpt-5-nano",
            input=prompt,
            max_output_tokens=120,
        )
        summary = _two_sentence_summary(response.output_text or "")
        if summary:
            return {"summary": summary, "source": "gpt-5-nano", "confidence_score": score}
    except Exception:
        # Summary generation is optional; the masking decision and score remain local.
        pass

    return {"summary": fallback, "source": "local", "confidence_score": score}
