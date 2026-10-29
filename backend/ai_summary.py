from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .domain import ProcessedDocument, ProcessingResult

logger = logging.getLogger(__name__)

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


class DocumentConfidenceEvidence(BaseModel):
    """Non-identifying, per-document signals used by the Azure assessment."""

    model_config = ConfigDict(extra="forbid")

    document_index: int = Field(ge=1, le=1000)
    document_class: DocumentClass
    boundary_confidence: float = Field(ge=0, le=1)
    scan_quality_score: float = Field(ge=0, le=1)
    raw_blur_score: float = Field(ge=0)
    minimum_blur_score: float = Field(ge=0)
    ocr_confidence: float = Field(ge=0, le=1)
    verification_confidence: float = Field(ge=0, le=1)
    number_detector_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    masked_regions: int = Field(ge=1, le=1000)
    post_mask_ocr_audit_passed: Literal[True] = True
    local_evidence_score: int = Field(ge=0, le=100)


class CardSummaryRequest(BaseModel):
    """Allow only non-identifying processing metadata into the assessment."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "rejected", "unavailable"]
    document_count: int = Field(default=0, ge=0, le=1000)
    masked_regions: int = Field(default=0, ge=0, le=1000)
    document_classes: list[DocumentClass] = Field(default_factory=list, max_length=8)
    document_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    verification_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    pipeline_score: Optional[int] = Field(default=None, ge=0, le=100)
    document_evidence: list[DocumentConfidenceEvidence] = Field(default_factory=list, max_length=100)
    reason_code: Optional[ReasonCode] = None


class AzureDocumentAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_index: int = Field(ge=1, le=1000)
    confidence_score: int = Field(ge=0, le=100)


class AzureMaskingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(max_length=360)
    document_scores: list[AzureDocumentAssessment]


_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "document_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_index": {"type": "integer"},
                    "confidence_score": {"type": "integer"},
                },
                "required": ["document_index", "confidence_score"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "document_scores"],
    "additionalProperties": False,
}


def _bounded_score(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _blur_quality_ceiling(evidence: DocumentConfidenceEvidence) -> int:
    """Keep Azure scores monotonic with the measured sharpness signal."""
    quality = _bounded_score(evidence.scan_quality_score)
    return round(50 + 50 * quality)


def _confidence_reason(
    evidence: DocumentConfidenceEvidence,
    azure_score: int,
    final_score: int,
    blur_ceiling: int,
) -> str:
    localization = (
        "not matched (neutral)"
        if evidence.number_detector_confidence is None
        else f"{evidence.number_detector_confidence:.2f}"
    )
    reason = (
        f"Azure {azure_score}/100; blur quality {evidence.scan_quality_score:.2f} of minimum "
        f"(variance {evidence.raw_blur_score:.1f}/{evidence.minimum_blur_score:.1f}); "
        f"boundary {evidence.boundary_confidence:.2f}, OCR {evidence.ocr_confidence:.2f}, "
        f"verification {evidence.verification_confidence:.2f}, number detector {localization}."
    )
    if final_score < azure_score:
        reason += f" Blur ceiling {blur_ceiling}/100 applied."
    reason += " Post-mask OCR audit passed."
    return reason[:360]


def _document_pipeline_confidence(document: ProcessedDocument) -> float:
    weights = (
        ("boundary_confidence", 0.30),
        ("scan_quality_score", 0.10),
        ("ocr_confidence", 0.25),
        ("verification_confidence", 0.25),
    )
    score = sum(
        _bounded_score(getattr(document, name)) * weight
        for name, weight in weights
    )
    localization = (
        0.50
        if document.number_detector_confidence is None
        else _bounded_score(document.number_detector_confidence)
    )
    return score + localization * 0.10


def confidence_evidence_for_result(
    result: ProcessingResult,
    minimum_blur_score: float,
) -> list[dict[str, object]]:
    """Build per-document quality evidence without sending document pixels or text."""
    evidence = []
    for index, document in enumerate(result.documents, start=1):
        item = DocumentConfidenceEvidence(
            document_index=index,
            document_class=document.document_class,
            boundary_confidence=document.boundary_confidence,
            scan_quality_score=document.scan_quality_score,
            raw_blur_score=document.raw_blur_score,
            minimum_blur_score=minimum_blur_score,
            ocr_confidence=document.ocr_confidence,
            verification_confidence=document.verification_confidence,
            number_detector_confidence=document.number_detector_confidence,
            masked_regions=document.masked_regions,
            post_mask_ocr_audit_passed=True,
            local_evidence_score=round(100 * _document_pipeline_confidence(document)),
        )
        evidence.append(item.model_dump(mode="json"))
    return evidence


def confidence_score(request: CardSummaryRequest) -> int | None:
    """Return the local fallback score, never a calibrated probability."""
    if request.outcome != "success":
        return None
    return request.pipeline_score


def pipeline_confidence_score(result: ProcessingResult) -> int | None:
    """Return the weakest document's local evidence score after safety gates pass."""
    if not result.documents:
        return None
    return round(100 * min(_document_pipeline_confidence(document) for document in result.documents))


def _local_summary(request: CardSummaryRequest, score: int | None) -> str:
    if request.outcome == "success":
        docs = f"{request.document_count} Aadhaar document{'s' if request.document_count != 1 else ''}"
        regions = f"{request.masked_regions} number line{'s' if request.masked_regions != 1 else ''}"
        if score is None:
            return f"The local checks created a masked copy for {docs} and redacted {regions}. A confidence score could not be calculated."
        return f"The local checks accepted a masked copy for {docs} and redacted {regions}. The evidence-weighted pipeline score is {score}/100."
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


def _azure_model_name() -> str:
    return os.getenv("AZURE_OPENAI_MODEL", "gpt-5-nano").strip() or "gpt-5-nano"


def _azure_assessment_prompt(request: CardSummaryRequest) -> list[dict[str, str]]:
    documents = [item.model_dump(mode="json") for item in request.document_evidence]
    system = (
        "You assess the quality of Aadhaar masking results from non-identifying numeric pipeline evidence. "
        "Do not ask for or infer personal details. The input contains no document image or OCR text. "
        "A successful result means deterministic checks and the post-mask OCR audit already passed; "
        "your score does not authorize or reject release. Give each listed document an integer score from 0 to 100. "
        "This is an operational evidence score, never a probability or guarantee. "
        "Base the score on every signal: boundary confidence, sharpness, OCR confidence, identity verification, "
        "optional number-detector corroboration, and the number of masked regions. "
        "Treat scan_quality_score as measured sharpness relative to the configured minimum: values below 1 are blurrier than the minimum. "
        "A higher raw_blur_score means sharper detail; a lower raw_blur_score means more blur. "
        "For example, raw_blur_score 10.5 with minimum_blur_score 14.0 is below the sharpness threshold, not crisper than it. "
        "Compare raw_blur_score with minimum_blur_score as well; lower ratios must reduce confidence compared with crisp results. "
        "Lower scores when blur or any required signal is weak, even though safety gates passed. "
        "A null number_detector_confidence is neutral because OCR word coordinates locate the mask and the audit passed. "
        "Use local_evidence_score as an anchor, but adjust it according to the evidence rather than copying it blindly. "
        "Score every document_index exactly once. "
        "Return a concise 1-2 sentence operator summary. Do not claim visual inspection, perfect masking, or an accuracy probability."
    )
    user = {
        "outcome": request.outcome,
        "document_count": request.document_count,
        "masked_regions": request.masked_regions,
        "overall_file_rule": "The file score is the lowest per-document score.",
        "documents": documents,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
    ]


def _azure_assessment(request: CardSummaryRequest, endpoint: str, api_key: str) -> dict[str, int | str | None]:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=_azure_base_url(endpoint),
        timeout=12.0,
        max_retries=0,
    )
    response = client.responses.create(
        model=_azure_model_name(),
        input=_azure_assessment_prompt(request),
        reasoning={"effort": "low"},
        text={
            "format": {
                "type": "json_schema",
                "name": "aadhaar_masking_assessment",
                "strict": True,
                "schema": _ASSESSMENT_SCHEMA,
            }
        },
        max_output_tokens=1200,
    )
    output_text = response.output_text or ""
    if not output_text.strip():
        incomplete_details = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete_details, "reason", None)
        output_types = [getattr(item, "type", "unknown") for item in (response.output or [])]
        raise ValueError(
            f"Azure returned no assessment text (status={response.status}, "
            f"reason={incomplete_reason}, output_types={output_types})."
        )
    assessment = AzureMaskingAssessment.model_validate_json(output_text)
    expected_indexes = {item.document_index for item in request.document_evidence}
    actual_indexes = [item.document_index for item in assessment.document_scores]
    if len(actual_indexes) != len(expected_indexes) or set(actual_indexes) != expected_indexes:
        raise ValueError("Azure assessment did not return exactly one score for each document.")

    evidence_by_index = {item.document_index: item for item in request.document_evidence}
    final_scores = []
    score_ceiling_by_index = {}
    for item in assessment.document_scores:
        evidence = evidence_by_index[item.document_index]
        ceiling = _blur_quality_ceiling(evidence)
        final_scores.append(
            AzureDocumentAssessment(
                document_index=item.document_index,
                confidence_score=min(item.confidence_score, ceiling),
            )
        )
        score_ceiling_by_index[item.document_index] = ceiling
    weakest = min(final_scores, key=lambda item: item.confidence_score)
    weakest_evidence = evidence_by_index[weakest.document_index]
    azure_score = next(
        item.confidence_score
        for item in assessment.document_scores
        if item.document_index == weakest.document_index
    )
    summary = _two_sentence_summary(assessment.summary)
    if not summary:
        summary = _local_summary(request, confidence_score(request))
    return {
        "summary": summary,
        "source": "azure_openai",
        "confidence_score": weakest.confidence_score,
        "confidence_source": "azure_openai",
        "confidence_reason": f"Document {weakest.document_index}: " + _confidence_reason(
            weakest_evidence,
            azure_score,
            weakest.confidence_score,
            score_ceiling_by_index[weakest.document_index],
        ),
    }


def _azure_summary_only(
    request: CardSummaryRequest,
    score: int | None,
    endpoint: str,
    api_key: str,
) -> dict[str, int | str | None]:
    from openai import OpenAI

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
        "Use only the supplied non-identifying processing metadata. Never claim perfect masking, infer document contents, "
        "or output a confidence score. For rejected or unavailable results, clearly say that no safe output is available.\n\n"
        f"Processing metadata: {json.dumps(metadata, separators=(',', ':'))}"
    )
    client = OpenAI(
        api_key=api_key,
        base_url=_azure_base_url(endpoint),
        timeout=8.0,
        max_retries=0,
    )
    response = client.responses.create(
        model=_azure_model_name(),
        input=prompt,
        max_output_tokens=120,
    )
    summary = _two_sentence_summary(response.output_text or "")
    if not summary:
        raise ValueError("Azure returned an empty summary.")
    return {
        "summary": summary,
        "source": "azure_openai",
        "confidence_score": score,
        "confidence_source": "pipeline" if score is not None else "none",
        "confidence_reason": None,
    }


def summarize_card_result(request: CardSummaryRequest) -> dict[str, int | str | None]:
    pipeline_score = confidence_score(request)
    fallback = {
        "summary": _local_summary(request, pipeline_score),
        "source": "local",
        "confidence_score": pipeline_score,
        "confidence_source": "pipeline" if pipeline_score is not None else "none",
        "confidence_reason": None,
    }
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if not endpoint or not api_key:
        return fallback

    try:
        if request.outcome == "success" and request.document_evidence:
            return _azure_assessment(request, endpoint, api_key)
        return _azure_summary_only(request, pipeline_score, endpoint, api_key)
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError):
            failure = exc.name
        elif isinstance(exc, ValidationError):
            failure = "; ".join(error["msg"] for error in exc.errors())
        elif isinstance(exc, ValueError):
            failure = str(exc)[:240]
        else:
            failure = type(exc).__name__
        logger.warning(
            "Azure OpenAI assessment unavailable; using the local evidence score (%s).",
            failure,
        )
        return fallback
