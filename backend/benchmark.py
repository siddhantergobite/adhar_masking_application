from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


SUPPORTED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif", ".doc", ".docx",
}
TERMINAL_BATCH_STATES = {"completed", "completed_with_issues"}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3) if values else 0.0,
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
    }


def collect_input_paths(input_dir: Path, positive_label_dir: Path | None = None) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if positive_label_dir is not None:
        if not positive_label_dir.is_dir():
            raise ValueError(f"Positive-label directory does not exist: {positive_label_dir}")
        paths = [
            path for path in paths
            if (label := positive_label_dir / f"{path.stem}.txt").is_file()
            and label.read_text(encoding="utf-8").strip()
        ]
    return paths


def collect_labeled_input_paths(input_dir: Path, labels_dir: Path) -> tuple[list[Path], dict[Path, str]]:
    paths = collect_input_paths(input_dir)
    if not labels_dir.is_dir():
        raise ValueError("Label directory does not exist.")
    categories: dict[Path, str] = {}
    for path in paths:
        label = labels_dir / f"{path.stem}.txt"
        if not label.is_file():
            raise ValueError("Every input needs a matching label file; missing labels are not treated as negatives.")
        categories[path] = "aadhaar" if label.read_text(encoding="utf-8-sig").strip() else "non_aadhaar"
    return paths, categories


def select_labeled_samples(
    categories: dict[Path, str],
    *,
    aadhaar_count: int,
    non_aadhaar_count: int,
    seed: int,
) -> list[Path]:
    if min(aadhaar_count, non_aadhaar_count) < 0:
        raise ValueError("Sample counts cannot be negative.")
    generator = random.Random(seed)
    selected: list[Path] = []
    for category, requested in (("aadhaar", aadhaar_count), ("non_aadhaar", non_aadhaar_count)):
        available = [path for path, label in categories.items() if label == category]
        count = requested or len(available)
        if count > len(available):
            raise ValueError(f"Requested {count} {category} samples, but only {len(available)} are available.")
        selected.extend(generator.sample(available, count))
    if not any(categories[path] == "aadhaar" for path in selected):
        raise ValueError("Select at least one Aadhaar-present sample.")
    if not any(categories[path] == "non_aadhaar" for path in selected):
        raise ValueError("Select at least one non-Aadhaar sample.")
    generator.shuffle(selected)
    return selected


def _request_headers(api_key: str, content_type: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _send_json_request(
    url: str,
    *,
    method: str,
    timeout: float,
    api_key: str,
    batch_api_key: str = "",
    body: bytes | None = None,
) -> tuple[int, dict[str, object]]:
    headers = _request_headers(api_key or batch_api_key)
    if batch_api_key:
        headers["X-Batch-API-Key"] = batch_api_key
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
            status = response.status
    except HTTPError as exc:
        return exc.code, {}
    if not payload:
        return status, {}
    try:
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, {}
    return status, data if isinstance(data, dict) else {}


def wait_for_backend(url: str, api_key: str, timeout: float) -> None:
    parsed = urlsplit(url)
    health_url = urlunsplit((parsed.scheme, parsed.netloc, f"{parsed.path.rsplit('/', 1)[0]}/health", "", ""))
    deadline = time.perf_counter() + timeout
    last_status = "unreachable"
    while time.perf_counter() < deadline:
        try:
            status, health = _send_json_request(
                health_url,
                method="GET",
                timeout=min(10.0, max(0.1, deadline - time.perf_counter())),
                api_key=api_key,
            )
        except (URLError, TimeoutError, OSError):
            status, health = 0, {}
        if status in {401, 403}:
            raise RuntimeError("Health endpoint rejected authentication; check AADHAAR_API_KEY.")
        if 200 <= status < 300 and health.get("ready") is True and health.get("status") == "ready":
            return
        last_status = str(health.get("status") or (f"HTTP {status}" if status else "unreachable"))
        time.sleep(min(2.0, max(0.0, deadline - time.perf_counter())))
    raise TimeoutError(f"Backend did not become ready within {timeout:g}s (last state: {last_status}).")


def submit_file(url: str, path: Path, timeout: float, api_key: str) -> dict[str, float | int | str]:
    boundary = f"----AadhaarMaskBench{uuid.uuid4().hex}"
    file_data = path.read_bytes()
    suffix = path.suffix.lower() or ".bin"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="benchmark-input{suffix}"\r\n'
        f"Content-Type: {mimetypes.guess_type(path.name)[0] or 'application/octet-stream'}\r\n\r\n"
    ).encode("ascii") + file_data + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = Request(
        url,
        data=body,
        headers={**_request_headers(api_key, f"multipart/form-data; boundary={boundary}"), "Accept": "application/pdf"},
        method="POST",
    )
    started = time.perf_counter()
    error_type = ""
    is_pdf = 0
    try:
        with urlopen(request, timeout=timeout) as response:
            # Consume the output so timing includes PDF creation and transfer.
            first_chunk = response.read(1024 * 1024)
            is_pdf = int(response.headers.get_content_type() == "application/pdf" and first_chunk.startswith(b"%PDF-"))
            while response.read(1024 * 1024):
                pass
            status = response.status
    except HTTPError as exc:
        status = exc.code
        exc.read()
    except (URLError, TimeoutError, OSError) as exc:
        status = 0
        error_type = type(exc).__name__
    elapsed = time.perf_counter() - started
    return {"status": status, "seconds": elapsed, "is_pdf": is_pdf, "error": error_type if status == 0 else ""}


def submit_batch(
    url: str,
    paths: list[Path],
    timeout: float,
    poll_interval: float,
    api_key: str,
    batch_api_key: str = "",
) -> dict[str, float | int | str]:
    started = time.perf_counter()
    boundary = f"----AadhaarMaskBench{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for path in paths:
        suffix = path.suffix.lower() or ".bin"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend([
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="benchmark-input{suffix}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            path.read_bytes(),
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    body = b"".join(chunks)
    headers = _request_headers(api_key or batch_api_key, f"multipart/form-data; boundary={boundary}")
    if batch_api_key:
        headers["X-Batch-API-Key"] = batch_api_key
    request = Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            payload = json.loads(response.read() or b"{}")
    except HTTPError as exc:
        return {"status": exc.code, "seconds": time.perf_counter() - started, "successful_files": 0,
                "rejected_files": 0, "failed_files": len(paths), "error": "http_error"}
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"status": 0, "seconds": time.perf_counter() - started, "successful_files": 0,
                "rejected_files": 0, "failed_files": len(paths), "error": type(exc).__name__}

    batch_id = payload.get("batch_id") if isinstance(payload, dict) else None
    if not isinstance(batch_id, str) or not batch_id:
        return {"status": status, "seconds": time.perf_counter() - started, "successful_files": 0,
                "rejected_files": 0, "failed_files": len(paths), "error": "missing_batch_id"}

    parsed_url = urlsplit(url)
    status_url = urlunsplit((parsed_url.scheme, parsed_url.netloc, f"{parsed_url.path.rstrip('/')}/{quote(batch_id)}", "", ""))
    deadline = started + timeout
    while time.perf_counter() < deadline:
        try:
            poll_status, batch = _send_json_request(
                status_url,
                method="GET",
                timeout=min(10.0, max(0.1, deadline - time.perf_counter())),
                api_key=api_key,
                batch_api_key=batch_api_key,
            )
        except (URLError, TimeoutError, OSError) as exc:
            return {"status": 0, "seconds": time.perf_counter() - started, "successful_files": 0,
                    "rejected_files": 0, "failed_files": len(paths), "error": type(exc).__name__}
        if not 200 <= poll_status < 300:
            return {"status": poll_status, "seconds": time.perf_counter() - started, "successful_files": 0,
                    "rejected_files": 0, "failed_files": len(paths), "error": "status_poll_failed"}
        if batch.get("status") in TERMINAL_BATCH_STATES:
            files = batch.get("files", [])
            statuses = [item.get("status") for item in files if isinstance(item, dict)] if isinstance(files, list) else []
            try:
                delete_status, _ = _send_json_request(
                    status_url,
                    method="DELETE",
                    timeout=min(10.0, max(0.1, deadline - time.perf_counter())),
                    api_key=api_key,
                    batch_api_key=batch_api_key,
                )
            except (URLError, TimeoutError, OSError):
                delete_status = 0
            return {
                "status": status,
                "seconds": time.perf_counter() - started,
                "successful_files": sum(value == "completed" for value in statuses),
                "rejected_files": sum(value == "rejected" for value in statuses),
                "failed_files": sum(value == "failed" for value in statuses),
                "cleanup_completed": int(200 <= delete_status < 300),
                "error": "",
            }
        time.sleep(min(poll_interval, max(0.0, deadline - time.perf_counter())))

    return {"status": 0, "seconds": time.perf_counter() - started, "successful_files": 0,
            "rejected_files": 0, "failed_files": len(paths), "error": "batch_timeout"}


def run_benchmark(
    *,
    url: str,
    input_dir: Path,
    mode: str,
    concurrency: int,
    repeats: int,
    warmups: int,
    timeout: float,
    poll_interval: float,
    limit: int = 0,
    positive_label_dir: Path | None = None,
    labels_dir: Path | None = None,
    aadhaar_count: int = 0,
    non_aadhaar_count: int = 0,
    seed: int = 20261030,
    health_timeout: float = 180,
) -> dict[str, object]:
    if positive_label_dir is not None and labels_dir is not None:
        raise ValueError("Use either --positive-label-dir or --labels-dir, not both.")
    categories: dict[Path, str] = {}
    if labels_dir is not None:
        if limit:
            raise ValueError("Use --aadhaar-count and --non-aadhaar-count instead of --limit with --labels-dir.")
        paths, categories = collect_labeled_input_paths(input_dir, labels_dir)
        paths = select_labeled_samples(
            categories,
            aadhaar_count=aadhaar_count,
            non_aadhaar_count=non_aadhaar_count,
            seed=seed,
        )
    else:
        paths = collect_input_paths(input_dir, positive_label_dir)
    if not paths:
        raise ValueError("No supported input files found for the requested selection.")
    if mode not in {"single", "batch"}:
        raise ValueError("Mode must be single or batch")
    if labels_dir is not None and (mode != "single" or concurrency != 1):
        raise ValueError("Ground-truth class timing requires --mode single --concurrency 1.")
    if concurrency < 1 or repeats < 1 or warmups < 0 or limit < 0 or poll_interval <= 0 or timeout <= 0 or health_timeout <= 0:
        raise ValueError("Concurrency and repeats must be positive; warmups and limit cannot be negative")
    if limit:
        paths = paths[:limit]

    api_key = os.getenv("AADHAAR_API_KEY", "").strip()
    if mode == "batch":
        batch_api_key = os.getenv("AADHAAR_BATCH_API_KEY", "").strip()
        jobs = [paths for _ in range(repeats)]
        task = lambda files: submit_batch(url, files, timeout, poll_interval, api_key, batch_api_key)
        measured_file_count = len(paths) * repeats
        request_count = len(jobs)
    else:
        jobs = [paths[index % len(paths)] for index in range(len(paths) * repeats)]
        task = lambda path: submit_file(url, path, timeout, api_key)
        measured_file_count = len(jobs)
        request_count = len(jobs)

    wait_for_backend(url, api_key or (batch_api_key if mode == "batch" else ""), health_timeout)
    for index in range(warmups):
        if mode == "batch":
            warmup = submit_batch(url, [paths[index % len(paths)]], timeout, poll_interval, api_key, batch_api_key)
        else:
            warmup = submit_file(url, paths[index % len(paths)], timeout, api_key)
        if int(warmup.get("status", 0)) in {0, 401, 403}:
            raise RuntimeError("A warm-up request could not reach the backend or was not authorized.")

    started = time.perf_counter()
    results: list[dict[str, float | int | str]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(task, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            job = futures[future]
            if mode == "single" and categories:
                result["input_category"] = categories[job]
            results.append(result)
    wall_seconds = time.perf_counter() - started

    latencies = [float(result["seconds"]) for result in results]
    statuses = [int(result["status"]) for result in results]
    successful_responses = (
        sum(status == 200 and int(result.get("is_pdf", 0)) == 1 for status, result in zip(statuses, results))
        if mode == "single"
        else sum(200 <= status < 300 for status in statuses)
    )
    rejected_responses = sum(400 <= status < 500 for status in statuses)
    failed_responses = len(statuses) - successful_responses - rejected_responses
    successful_files = (
        sum(int(result["successful_files"]) for result in results)
        if mode == "batch" else successful_responses
    )
    rejected_files = (
        sum(int(result["rejected_files"]) for result in results)
        if mode == "batch" else rejected_responses
    )
    failed_files = (
        sum(int(result["failed_files"]) for result in results)
        if mode == "batch" else failed_responses
    )
    report: dict[str, object] = {
        "endpoint": urlsplit(url).path,
        "mode": mode,
        "selection": (
            "ground_truth_aadhaar_vs_non_aadhaar_labels"
            if labels_dir is not None
            else "positive_yolo_segmentation_labels"
            if positive_label_dir is not None
            else "all_supported_files"
        ),
        "files": measured_file_count,
        "requests": request_count,
        "files_per_batch": len(paths) if mode == "batch" else 1,
        "concurrency": concurrency,
        "warmup_requests": warmups,
        "successful_files": successful_files,
        "rejected_files": rejected_files,
        "failed_files": failed_files,
        "cleanup_failures": (
            sum(not bool(result.get("cleanup_completed", 0)) for result in results)
            if mode == "batch" else 0
        ),
        "successful_responses": successful_responses,
        "rejected_responses": rejected_responses,
        "failed_responses": failed_responses,
        "wall_seconds": round(wall_seconds, 3),
        "submitted_files_per_minute": round(measured_file_count * 60 / wall_seconds, 2) if wall_seconds else 0,
        "terminal_outcomes_per_minute": round((successful_files + rejected_files) * 60 / wall_seconds, 2) if wall_seconds else 0,
        "throughput_tps": round(measured_file_count / wall_seconds, 4) if wall_seconds else 0,
        "average_response_seconds": round(statistics.fmean(latencies), 3) if latencies else 0,
        "latency_seconds_per_request": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0,
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
        },
        "latency_seconds_by_outcome": (
            {
                "successful_masked_pdf": latency_summary([
                    float(result["seconds"])
                    for result in results
                    if int(result["status"]) == 200 and int(result.get("is_pdf", 0)) == 1
                ]),
                "rejected_input": latency_summary([
                    float(result["seconds"])
                    for result in results
                    if int(result["status"]) == 422
                ]),
                "other_error": latency_summary([
                    float(result["seconds"])
                    for result in results
                    if not (
                        int(result["status"]) == 200 and int(result.get("is_pdf", 0)) == 1
                    ) and int(result["status"]) != 422
                ]),
            }
            if mode == "single"
            else None
        ),
        "http_status_counts": {str(status): statuses.count(status) for status in sorted(set(statuses))},
    }
    if categories and mode == "single":
        class_results: dict[str, object] = {}
        for category in ("aadhaar", "non_aadhaar"):
            class_records = [result for result in results if result.get("input_category") == category]
            class_statuses = [int(result["status"]) for result in class_records]
            class_latencies = [float(result["seconds"]) for result in class_records]
            masked_pdf_latencies = [
                float(result["seconds"])
                for result in class_records
                if int(result["status"]) == 200 and int(result.get("is_pdf", 0)) == 1
            ]
            masked_count = sum(
                status == 200 and int(result.get("is_pdf", 0)) == 1
                for status, result in zip(class_statuses, class_records)
            )
            class_results[category] = {
                "inputs": len(class_records),
                "masked_pdfs": masked_count,
                "rejected_422": sum(status == 422 for status in class_statuses),
                "errors": len(class_records) - masked_count - sum(status == 422 for status in class_statuses),
                "latency_seconds": latency_summary(class_latencies),
                "masked_pdf_latency_seconds": latency_summary(masked_pdf_latencies),
            }
        report["input_mix"] = {
            "aadhaar": int(class_results["aadhaar"]["inputs"]),
            "non_aadhaar": int(class_results["non_aadhaar"]["inputs"]),
            "total": len(results),
        }
        report["outcomes_by_input_category"] = class_results
        report["average_aadhaar_document_seconds"] = class_results["aadhaar"]["latency_seconds"]["mean"]
        report["average_non_aadhaar_document_seconds"] = class_results["non_aadhaar"]["latency_seconds"]["mean"]
    return report


def main() -> None:
    from .configuration import load_project_environment

    load_project_environment()
    parser = argparse.ArgumentParser(
        description="Measure local Aadhaar masking latency and throughput without printing filenames or document contents."
    )
    parser.add_argument("input_dir", type=Path, help="Use synthetic or approved de-identified test files only")
    parser.add_argument("--mode", choices=("single", "batch"), default="single")
    parser.add_argument("--url", default=None, help="API URL; defaults to /api/process or /api/batches for the selected mode")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent API requests or batch submissions")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N inputs; zero uses all files")
    parser.add_argument(
        "--positive-label-dir",
        type=Path,
        default=None,
        help="Only use images with non-empty YOLO segmentation labels in this directory",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Use all labeled inputs and report separate Aadhaar/non-Aadhaar latency; empty labels mean non-Aadhaar",
    )
    parser.add_argument("--aadhaar-count", type=int, default=0, help="Labeled mode: sample this many Aadhaar inputs; zero selects all")
    parser.add_argument("--non-aadhaar-count", type=int, default=0, help="Labeled mode: sample this many negative inputs; zero selects all")
    parser.add_argument("--seed", type=int, default=20261030, help="Deterministic labeled sample and ordering seed")
    parser.add_argument("--timeout", type=float, default=300, help="End-to-end request/job timeout in seconds")
    parser.add_argument("--health-timeout", type=float, default=180, help="Wait for backend readiness before timing")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Batch status polling interval in seconds")
    args = parser.parse_args()
    url = args.url or (
        "http://127.0.0.1:8002/api/batches" if args.mode == "batch"
        else "http://127.0.0.1:8002/api/process"
    )
    try:
        report = run_benchmark(
            url=url,
            input_dir=args.input_dir.resolve(),
            mode=args.mode,
            concurrency=args.concurrency,
            repeats=args.repeats,
            warmups=args.warmups,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            limit=args.limit,
            positive_label_dir=args.positive_label_dir.resolve() if args.positive_label_dir else None,
            labels_dir=args.labels_dir.resolve() if args.labels_dir else None,
            aadhaar_count=args.aadhaar_count,
            non_aadhaar_count=args.non_aadhaar_count,
            seed=args.seed,
            health_timeout=args.health_timeout,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
