# Local benchmark results

**Run date:** 2026-10-29  
**Status:** exploratory workstation measurements; not a production capacity claim or SLA.

The current local serial profile uses one Uvicorn worker, four OCR CPU threads, 512-pixel detector inference, 960-pixel first-pass OCR, a 1200-pixel post-mask audit, and one browser request at a time. Historical two-worker runs below are retained for context and do not predict current serial throughput.

## Fresh serial Aadhaar-only retest

The API was allowed to become ready before timing. One warm-up was excluded. Then all 21 synthetic held-out images with non-empty Aadhaar segmentation labels were submitted one at a time to `POST /api/process`. Each request was timed through the complete masked-PDF response body.

| Metric | Result |
| --- | ---: |
| Labeled inputs | 21 |
| Successful masked PDFs | 15 |
| Verification rejections (HTTP 422) | 6 |
| Server failures | 0 |
| Successful PDF mean / median | 19.31 s / 17.40 s per document |
| Successful PDF p95 / range | 31.02 s / 8.23–35.84 s |
| Mean across all labeled inputs, including rejections | 17.11 s/request |
| Input throughput | 0.0584 requests/s (3.51 inputs/min) |
| Successful masked-PDF throughput | 2.51 PDFs/min |

These are synthetic CPU-only workstation results, not a production SLA. A positive training label does not guarantee that the runtime identity, number, and quality checks should accept a document. Rejections are excluded from the successful-PDF latency mean.

On the same single synthetic Aadhaar image, a fresh three-request repeat at the current 512/960 settings returned 16.21 s, 16.85 s, and 16.60 s (mean 16.55 s; warm-up 19.91 s, excluded). This single-image repeat is not a substitute for the 21-image average above.

Two temporary speed trials on that same image were slower and were not kept: changing first-pass OCR from 960 to 640 produced three successful responses averaging 27.0 s; changing detector inference from 512 to 384 produced three successful responses averaging 28.92 s. The saved profile remains 512/960.

## Same-document serial CPU retest

The same successful synthetic Aadhaar image was submitted three times per profile to `POST /api/process`, with one Uvicorn worker and one request at a time. Each response body was consumed so timing includes the masked PDF transfer.

| OCR CPU threads | Successful response times | Mean |
| ---: | --- | ---: |
| 1 | 37.90 s, 27.99 s, 28.14 s | 31.34 s/document |
| 4, earlier repeat | 19.93 s, 15.60 s, 17.47 s | 17.67 s/document |
| 4, fresh repeat at 512/960 | 16.21 s, 16.85 s, 16.60 s | 16.55 s/document |

Four OCR threads reduced the mean by about 44% on this sample. The result still misses a 7–8 second target; this local CPU test does not establish the latency available on a 16-vCPU server or GPU.

## Environment

- Windows 10, 8 logical CPUs
- CPU inference: YOLO and PaddleOCR both configured for CPU; CUDA unavailable
- One Uvicorn worker for the fresh serial retest; historical runs also used two
- `AADHAAR_OCR_CPU_THREADS=4` for the current serial profile
- `AADHAAR_INFERENCE_SIZE=512`, first-pass OCR long edge 960, audit long edge 1200
- Health endpoint became consistently ready about 30 seconds after the server process started

## Test data and outcomes

The benchmark used nine images from the synthetic held-out test split, selected by non-empty YOLO segmentation labels. The dataset manifest says the generated set contains no real personal data. The same nine images were used for direct serial requests and queued batches.

Each nine-file run returned six masked PDFs, rejected three files with HTTP 422, and had zero server errors. A non-empty training label does not guarantee that the stricter runtime identity, number, and quality checks should accept a file; review the rejected cases before drawing accuracy conclusions.

## Timings

| Run | Workload | Wall time | Throughput | Outcomes |
| --- | --- | ---: | ---: | --- |
| Direct serial, exploratory | 9 files, one warm-up | 99.3 s | 5.44 files/min | 6 masked, 3 rejected |
| Direct serial, final code | 9 files, one warm-up | 202.8 s | 2.66 files/min | 6 masked, 3 rejected |
| Two-worker batch, final code | 9 files | 105.0 s | 5.14 files/min | 6 masked, 3 rejected; cleanup succeeded |
| Two-worker batch, repeated | 2 batches of 9, one warm-up | 184.5 s total; 92.3 s mean per batch | 5.85 files/min | 12 masked, 6 rejected; 0 failed; cleanup succeeded |

The same direct serial workload varied from 99 to 203 seconds. The earlier 99-second serial result was 5.44 files/min; the repeated batch result was 5.85 files/min, only about 7.5% higher. A later serial run measured 2.66 files/min, which shows substantial run-to-run variability. These runs do not establish a stable speedup. Repeat on a quiet machine with the expected document mix before making a capacity claim.

The older throughput measurements are historical two-worker/four-thread runs. The current profile uses one Uvicorn worker and four OCR CPU threads, with one browser request at a time. Multiple OCR CPU threads operate within the single document being processed; they do not process multiple documents concurrently.

An exploratory two-thread-per-worker run was slower on this workstation: 149.6 seconds for nine direct requests and 159.2 seconds for a nine-file batch. Four threads per worker was the better measured setting here; retune it for the deployment CPU and memory limits.

The first nine files in the sorted test directory are all negative examples. They produced nine 422 rejections, which is why the benchmark now supports `--positive-label-dir` to select labeled examples explicitly. Rejection-only timing is not representative of masking throughput.

## Reproduce

To reproduce these historical results, start the API with two workers and four OCR CPU threads, then run:

```powershell
$env:AADHAAR_OCR_CPU_THREADS = "4"
.\.venv\Scripts\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8002 --workers 2 --no-access-log
```

In another PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m backend.benchmark datasets\aadhaar-document-seg\images\test --positive-label-dir datasets\aadhaar-document-seg\labels\test --mode single --concurrency 1 --limit 9 --warmups 1
.\.venv\Scripts\python.exe -m backend.benchmark datasets\aadhaar-document-seg\images\test --positive-label-dir datasets\aadhaar-document-seg\labels\test --mode batch --concurrency 1 --limit 9 --repeats 2 --warmups 1
```

The benchmark output contains aggregate counts and timings only. It does not print file names, OCR text, or document contents. Batch benchmark records are deleted after each measured job completes.

## Limits before production

- These are synthetic CPU-only results. GPU and AWS performance were not measured.
- The synthetic checkpoints and nine-file sample do not establish real-world masking accuracy; three labeled-positive files were rejected.
- The PostgreSQL integration test was skipped because no test database is configured. Multi-host PostgreSQL/S3/R2 mode still needs a target-environment test.
- Production still needs a TLS authentication gateway, rate limits, isolated document conversion, monitoring, and a reviewed retention policy.
- No confidence scores are displayed, returned by the API, or written to new batch records or manifests. Detection and verification thresholds remain internal safety controls.

## Fresh crop-first CPU check

**Run date:** 2026-10-30. The backend used one Uvicorn worker, four OCR CPU threads, four ordered orientation attempts, 512-pixel detector inference, 960-pixel first-pass OCR, a 1200-pixel post-mask audit, and one request at a time. One warm-up was excluded. The same 21 synthetic labeled inputs were sent serially, with each complete PDF response consumed.

| Metric | Result |
| --- | ---: |
| Masked PDFs / verification rejections / server failures | 15 / 6 / 0 |
| Accepted PDF mean / median / p95 | 13.60 s / 12.58 s / 21.63 s |
| Mean over all requests | 11.76 s |
| Serial input throughput | 5.10 requests/min |
| Successful PDF throughput | 3.65 PDFs/min |
| Status and redacted-line counts matching four-orientation reference | 21 / 21 |

This run improved on the slower staged two-orientation recovery (20.76 seconds per accepted PDF). The mixed-document sample returned the Aadhaar crop in 10.61 seconds; the staged path took 28.81 seconds for the same crop. The earlier two-orientation-only result averaged 7.28 seconds but returned an overly broad image for that sample, so it is not a safe speed result. The crop-first run still missed the 7-second target and was slower than an earlier four-orientation capture (8.96 seconds); workstation timings varied substantially, so use target-server measurements for capacity claims.

## 2026-10-30 OCR preprocessing and serial retest

The OCR path now downsamples a rectified page to its existing configured long-edge limit before CLAHE and sharpening. OCR coordinates are still mapped back to the original crop, and the full-resolution post-mask audit remains enabled. Detector size (512), first-pass OCR limit (960), audit limit (1200), four OCR CPU threads, one Uvicorn worker, and one active document request were unchanged.

| Run | Inputs | Outcomes | Mean latency | p95 | Throughput |
| --- | --- | --- | ---: | ---: | ---: |
| Synthetic positive + payment-card negative; one warm-up | 2 measured requests | 1 masked PDF, 1 HTTP 422, 0 errors | 10.06 s/request | 10.24 s | 5.96 inputs/min |
| Same synthetic positive repeated serially; one warm-up | 3 measured requests | 3 masked PDFs, 0 errors | 17.75 s/PDF | 23.42 s | 3.38 PDFs/min |
| Held-out synthetic inputs with non-empty segmentation labels; one warm-up | 9 measured requests | 6 masked PDFs, 3 HTTP 422, 0 errors | 26.72 s/request | 43.56 s | 2.25 inputs/min |

In the two-input run, the single masked-PDF response took 9.87 seconds and the single payment-card rejection took 10.26 seconds. A separate one-request measurement of the same positive synthetic sample took 9.66 seconds, but the three-repeat run averaged 17.75 seconds. This spread means a fast one-off does not establish a stable under-10-second average. The larger held-out run also remains well above the target. These workstation results do not prove that the preprocessing change reduced end-to-end latency; the change reduces image area for enhancement, while request timings remain highly variable.

These were direct local API tests, not a completed browser upload test. Windows Computer Use was stopped with Escape before file selection, so no claim is made that the frontend upload flow was tested in this run.
