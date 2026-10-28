from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


API_ROOT = "https://api.cloudflare.com/client/v4"
KV_MAX_VALUE_BYTES = 25 * 1024 * 1024


class CloudflareKVError(RuntimeError):
    pass


class CloudflareKVValueTooLarge(CloudflareKVError):
    pass


def configured() -> bool:
    return all(os.getenv(name, "").strip() for name in (
        "CLOUDFLARE_KV_ACCOUNT_ID",
        "CLOUDFLARE_KV_NAMESPACE_ID",
        "CLOUDFLARE_KV_API_TOKEN",
    ))


def namespace_id() -> str:
    return os.getenv("CLOUDFLARE_KV_NAMESPACE_ID", "").strip()


def input_prefix() -> str:
    return _prefix("CLOUDFLARE_KV_INPUT_PREFIX", "aadhaar/inbox")


def output_prefix() -> str:
    return _prefix("CLOUDFLARE_KV_OUTPUT_PREFIX", "aadhaar/masked")


def _prefix(env_name: str, default: str) -> str:
    value = os.getenv(env_name, default).strip().strip("/")
    if not value:
        raise CloudflareKVError(f"{env_name} must not be empty")
    return value + "/"


def prefixes_are_separate() -> bool:
    try:
        inbox = input_prefix()
        masked = output_prefix()
    except CloudflareKVError:
        return False
    return not (inbox.startswith(masked) or masked.startswith(inbox))


def _base_path() -> str:
    account = os.getenv("CLOUDFLARE_KV_ACCOUNT_ID", "").strip()
    namespace = namespace_id()
    token = os.getenv("CLOUDFLARE_KV_API_TOKEN", "").strip()
    if not all((account, namespace, token)):
        raise CloudflareKVError("Workers KV account, namespace, and API token are not configured")
    if len(account) > 32 or len(namespace) > 32 or not account.isalnum() or not namespace.isalnum():
        raise CloudflareKVError("Workers KV account or namespace identifier is invalid")
    return f"/accounts/{account}/storage/kv/namespaces/{namespace}"


def _request(method: str, path: str, *, data: bytes | None = None, content_type: str | None = None,
             query: dict[str, str | int] | None = None, max_response_bytes: int = 2 * 1024 * 1024) -> bytes:
    token = os.getenv("CLOUDFLARE_KV_API_TOKEN", "").strip()
    url = API_ROOT + path
    if query:
        url += "?" + urlencode(query)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read(max_response_bytes + 1)
            if len(payload) > max_response_bytes:
                raise CloudflareKVError("Workers KV returned an unexpectedly large response")
            return payload
    except HTTPError as exc:
        # Do not forward the response body: it can contain account details and
        # should not be shown to users of the local masking interface.
        raise CloudflareKVError(f"Workers KV request failed with HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise CloudflareKVError("Could not reach the Cloudflare Workers KV API") from exc


def _api_json(method: str, path: str, *, query: dict[str, str | int] | None = None) -> dict:
    raw = _request(method, path, query=query)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudflareKVError("Cloudflare returned an invalid Workers KV API response") from exc
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise CloudflareKVError("Cloudflare rejected the Workers KV API request")
    return payload


def list_keys(prefix: str, max_keys: int) -> list[dict[str, object]]:
    base = _base_path()
    keys: list[dict[str, object]] = []
    cursor = ""
    while True:
        query: dict[str, str | int] = {"prefix": prefix, "limit": 1000}
        if cursor:
            query["cursor"] = cursor
        payload = _api_json("GET", f"{base}/keys", query=query)
        result = payload.get("result")
        if not isinstance(result, list):
            raise CloudflareKVError("Cloudflare returned an invalid key list")
        for row in result:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                keys.append({
                    "name": row["name"],
                    "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                })
                if len(keys) > max_keys:
                    return keys
        result_info = payload.get("result_info") or {}
        cursor = str(result_info.get("cursor") or "") if isinstance(result_info, dict) else ""
        if not cursor:
            return keys


def put_file(key: str, source: Path, metadata: dict[str, object] | None = None) -> None:
    size = source.stat().st_size
    if size > KV_MAX_VALUE_BYTES:
        raise CloudflareKVValueTooLarge("Workers KV values cannot exceed 25 MiB")
    value = source.read_bytes()
    if not metadata:
        put_bytes(key, value)
        return
    boundary = "----AadhaarMaskKV" + uuid.uuid4().hex
    metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(metadata_json) > 1024:
        raise ValueError("Workers KV metadata cannot exceed 1024 bytes")
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="value"; filename="value"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("ascii")
    metadata_part = (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="metadata"\r\n'
        "Content-Type: application/json\r\n\r\n"
    ).encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    _request(
        "PUT",
        f"{_base_path()}/values/{quote_key(key)}",
        data=prefix + value + metadata_part + metadata_json + suffix,
        content_type=f"multipart/form-data; boundary={boundary}",
        max_response_bytes=1024 * 1024,
    )


def put_bytes(key: str, value: bytes) -> None:
    if len(value) > KV_MAX_VALUE_BYTES:
        raise CloudflareKVValueTooLarge("Workers KV values cannot exceed 25 MiB")
    _request(
        "PUT",
        f"{_base_path()}/values/{quote_key(key)}",
        data=value,
        content_type="application/octet-stream",
        max_response_bytes=1024 * 1024,
    )


def download_value(key: str, destination: Path, max_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = os.getenv("CLOUDFLARE_KV_API_TOKEN", "").strip()
    request = Request(
        API_ROOT + f"{_base_path()}/values/{quote_key(key)}",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    size = 0
    try:
        with urlopen(request, timeout=30) as response, destination.open("wb") as target:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > min(max_bytes, KV_MAX_VALUE_BYTES):
                    raise CloudflareKVValueTooLarge("Workers KV value exceeds the configured file limit")
                target.write(chunk)
        return size
    except HTTPError as exc:
        destination.unlink(missing_ok=True)
        raise CloudflareKVError(f"Workers KV read failed with HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise CloudflareKVError("Could not read a file from Workers KV") from exc
    except CloudflareKVValueTooLarge:
        destination.unlink(missing_ok=True)
        raise


def delete_value(key: str) -> None:
    _request(
        "DELETE",
        f"{_base_path()}/values/{quote_key(key)}",
        max_response_bytes=1024 * 1024,
    )


def make_uri(key: str) -> str:
    return f"kv://{namespace_id()}/{quote_key(key)}"


def delete_uri(uri: str) -> None:
    from urllib.parse import unquote, urlsplit

    parsed = urlsplit(uri)
    if parsed.scheme != "kv" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Stored Workers KV output location is invalid")
    if parsed.netloc != namespace_id():
        raise CloudflareKVError("Stored Workers KV output refers to a different namespace")
    delete_value(unquote(parsed.path.lstrip("/")))


def quote_key(key: str) -> str:
    if not key or len(key.encode("utf-8")) > 512 or any(ord(char) < 33 for char in key):
        raise ValueError("Workers KV key is invalid")
    return quote(key, safe="")
