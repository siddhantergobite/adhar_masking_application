from __future__ import annotations


class PipelineError(Exception):
    code = "processing_failed"
    status_code = 422
    public_message = "The document could not be processed safely."
    action = "Try a sharper, well-lit image with the complete Aadhaar visible."

    def __init__(self, message: str | None = None, *, action: str | None = None):
        super().__init__(message or self.public_message)
        self.message = message or self.public_message
        self.user_action = action or self.action

    def detail(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "action": self.user_action,
        }


class ModelNotReady(PipelineError):
    code = "models_not_ready"
    status_code = 503
    public_message = "The trained Aadhaar detection models are not installed."
    action = "Train or copy both approved model weights into backend/models, then restart the API."


class NoAadhaarDetected(PipelineError):
    code = "no_aadhaar_detected"
    public_message = "No Aadhaar document was detected. Nothing was changed."
    action = "Upload an Aadhaar card or Aadhaar letter with its full boundary visible."


class UnsafeDocument(PipelineError):
    code = "review_required"
    public_message = "The Aadhaar could not be masked with enough confidence. No output was created."


class NoAadhaarNumber(PipelineError):
    code = "no_aadhaar_number"
    public_message = "An Aadhaar document was found, but no unmasked Aadhaar number could be verified."
    action = "Use the front or full letter and make sure the complete number is sharp and unobstructed."


class OcrFailure(PipelineError):
    code = "ocr_failed"
    status_code = 500
    public_message = "The private OCR engine failed while reading the document."
    action = "Restart the API and try once more."

