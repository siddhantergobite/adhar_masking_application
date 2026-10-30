from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from backend.server import require_batch_api_key
from backend.security import ApiKeyMiddleware, validate_api_key_configuration


class ApiKeyMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_request_without_key_is_rejected(self) -> None:
        downstream_called = False

        async def app(scope, receive, send):
            nonlocal downstream_called
            downstream_called = True

        middleware = ApiKeyMiddleware(app, "private-key")
        sent = []

        async def send(message):
            sent.append(message)

        await middleware(
            {"type": "http", "path": "/api/process", "headers": []},
            lambda: None,
            send,
        )

        self.assertFalse(downstream_called)
        self.assertEqual(sent[0]["status"], 401)

    async def test_api_request_with_key_reaches_application(self) -> None:
        downstream_called = False

        async def app(scope, receive, send):
            nonlocal downstream_called
            downstream_called = True

        middleware = ApiKeyMiddleware(app, "private-key")
        await middleware(
            {"type": "http", "path": "/api/process", "headers": [(b"x-api-key", b"private-key")]},
            lambda: None,
            lambda message: None,
        )

        self.assertTrue(downstream_called)

    async def test_non_api_route_is_not_guarded(self) -> None:
        downstream_called = False

        async def app(scope, receive, send):
            nonlocal downstream_called
            downstream_called = True

        middleware = ApiKeyMiddleware(app, "private-key")
        await middleware(
            {"type": "http", "path": "/", "headers": []},
            lambda: None,
            lambda message: None,
        )

        self.assertTrue(downstream_called)


class ApiKeyConfigurationTests(unittest.TestCase):
    def test_required_key_must_be_configured(self) -> None:
        with patch.dict(os.environ, {"AADHAAR_REQUIRE_API_KEY": "true", "AADHAAR_API_KEY": "", "AADHAAR_BATCH_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                validate_api_key_configuration()

    def test_optional_key_does_not_block_local_startup(self) -> None:
        with patch.dict(os.environ, {"AADHAAR_REQUIRE_API_KEY": "false", "AADHAAR_API_KEY": "", "AADHAAR_BATCH_API_KEY": ""}):
            validate_api_key_configuration()

    def test_required_key_must_have_sufficient_length(self) -> None:
        with patch.dict(os.environ, {"AADHAAR_REQUIRE_API_KEY": "true", "AADHAAR_API_KEY": "short", "AADHAAR_BATCH_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                validate_api_key_configuration()

    def test_configured_batch_key_must_have_sufficient_length(self) -> None:
        with patch.dict(os.environ, {"AADHAAR_REQUIRE_API_KEY": "false", "AADHAAR_API_KEY": "", "AADHAAR_BATCH_API_KEY": "short"}):
            with self.assertRaises(RuntimeError):
                validate_api_key_configuration()


class BatchApiKeyTests(unittest.TestCase):
    def test_separate_batch_key_is_checked_on_its_own_header(self) -> None:
        with patch.dict(os.environ, {"AADHAAR_API_KEY": "global-key", "AADHAAR_BATCH_API_KEY": "batch-key"}):
            require_batch_api_key(x_api_key="global-key", x_batch_api_key="batch-key")
            with self.assertRaises(HTTPException):
                require_batch_api_key(x_api_key="global-key", x_batch_api_key="wrong-key")

    def test_matching_global_and_batch_key_keeps_single_header_compatibility(self) -> None:
        with patch.dict(os.environ, {"AADHAAR_BATCH_API_KEY": "same-key"}):
            require_batch_api_key(x_api_key="same-key", x_batch_api_key=None)


if __name__ == "__main__":
    unittest.main()
