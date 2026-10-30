from __future__ import annotations

import hmac
import os
from typing import Any

from starlette.responses import JSONResponse


def api_key_required() -> bool:
    return os.getenv("AADHAAR_REQUIRE_API_KEY", "false").strip().lower() in {"1", "true", "yes", "on"}


def configured_api_key() -> str:
    return os.getenv("AADHAAR_API_KEY", "").strip()


def validate_api_key_configuration() -> None:
    key = configured_api_key()
    batch_key = os.getenv("AADHAAR_BATCH_API_KEY", "").strip()
    if api_key_required() and not key:
        raise RuntimeError("AADHAAR_REQUIRE_API_KEY is enabled but AADHAAR_API_KEY is empty")
    if key and len(key) < 32:
        raise RuntimeError("AADHAAR_API_KEY must contain at least 32 characters when configured")
    if batch_key and len(batch_key) < 32:
        raise RuntimeError("AADHAAR_BATCH_API_KEY must contain at least 32 characters")


class ApiKeyMiddleware:
    """Protect API routes with a secret injected by a trusted reverse proxy."""

    def __init__(self, app: Any, expected_key: str) -> None:
        self.app = app
        self.expected_key = expected_key.encode("utf-8")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope.get("path", "").startswith("/api/") and self.expected_key:
            headers = dict(scope.get("headers", []))
            supplied_key = headers.get(b"x-api-key", b"")
            if not hmac.compare_digest(supplied_key, self.expected_key):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "A valid API key is required."},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
