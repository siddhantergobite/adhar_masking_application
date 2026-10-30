# Backend performance test

This procedure measures the FastAPI backend directly, one document at a time. It does not measure the browser or Nginx. The benchmark waits for `/api/health` to report `ready`, sends and times each `POST /api/process`, consumes the full response, and excludes warm-up requests.

## Current available port

On 2026-10-30, ports `8000` through `8005` and Vite ports `5173` through `5175` were not listening on this workstation. Use backend port `8002`, which is also the project's usual local port. Check again before starting the service because port availability can change.

PowerShell port check:

```powershell
$ports = 8000..8005
$rows = foreach ($port in $ports) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -First 1
    [PSCustomObject]@{
        Port      = $port
        Available = -not [bool]$listener
        ProcessId = if ($listener) { $listener.OwningProcess } else { $null }
    }
}
$rows | Format-Table -AutoSize
```

## Correct sample set

The workspace's held-out document test split is under:

```text
datasets/aadhaar-document-seg/images/test
datasets/aadhaar-document-seg/labels/test
```

It currently contains 36 generated images: 21 with non-empty document-segmentation labels (Aadhaar is present) and 15 with empty labels (non-Aadhaar negatives). The labels are ground truth for the synthetic examples; they do not guarantee that the runtime should accept a sample. These samples are safe for an initial integration and timing run, but they are **synthetic bootstrap data**, not a production-representative accuracy set.

For a production report, use a locked, consented, de-identified test corpus with reviewed YOLO labels. The benchmark reports Aadhaar-present versus non-Aadhaar-only inputs; Aadhaar-positive scenes may include unrelated cards. It rejects missing labels instead of silently treating them as negatives.

Do not claim a 30,000-unique-document result by replaying the workspace's 36 synthetic images. `--repeats` replays the same selected files; it is useful for a repeated-load timing check, not for sample diversity or production accuracy.

## Start the backend

Run the same backend configuration planned for the server, with Uvicorn bound to loopback so Nginx and public network time are excluded. Keep the deployment's intended model, CPU-thread, and worker settings; do not tune them just for this report.

Windows PowerShell:

```powershell
Set-Location "<project directory>"
.\.venv\Scripts\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8002 --workers 1 --no-access-log
```

Linux server:

```bash
cd <project-directory>
source .venv/bin/activate
uvicorn backend.server:app --host 127.0.0.1 --port 8002 --workers 1 --no-access-log
```

The benchmark loads `AADHAAR_API_KEY` from the project `.env` or process environment and never prints it. Keep the API key out of command-line arguments and report files.

## Run the benchmark

Quick balanced smoke run: six Aadhaar-present and six non-Aadhaar samples, one warm-up, then twelve measured requests in randomized serial order.

```powershell
.\.venv\Scripts\python.exe -m backend.benchmark datasets\aadhaar-document-seg\images\test `
    --url http://127.0.0.1:8002/api/process `
    --mode single `
    --concurrency 1 `
    --labels-dir datasets\aadhaar-document-seg\labels\test `
    --aadhaar-count 6 `
    --non-aadhaar-count 6 `
    --warmups 1 `
    --repeats 1
```

Run the complete available synthetic split (21 Aadhaar-present and 15 negative samples):

```powershell
.\.venv\Scripts\python.exe -m backend.benchmark datasets\aadhaar-document-seg\images\test `
    --url http://127.0.0.1:8002/api/process `
    --mode single `
    --concurrency 1 `
    --labels-dir datasets\aadhaar-document-seg\labels\test `
    --aadhaar-count 21 `
    --non-aadhaar-count 15 `
    --warmups 1 `
    --repeats 1
```

For a reviewed real-world test corpus, use its image folder and matching YOLO label directory with the same command. Run from the repository root. The script prints aggregate JSON and does not save masked PDFs or print filenames, OCR text, Aadhaar numbers, API keys, or error response bodies. Save a report with `Tee-Object` if needed.

## Metrics and interpretation

| Requested metric | Script calculation |
| --- | --- |
| Average response time | Mean end-to-end `POST /api/process` request time across all measured inputs; expected HTTP 422 responses are included. |
| Throughput (TPS) | Measured requests divided by total measured wall time. The script is serial, so this is one-request-at-a-time TPS. It also reports documents/minute and masked PDFs/minute. |
| Average Aadhaar document time | Mean request time for Aadhaar-present samples, including safe rejections; the category report also gives latency for successfully returned masked PDFs only. |
| Average non-Aadhaar document time | Mean request time for non-Aadhaar samples, including expected HTTP 422 responses; rejection latency is also reported separately. |
| Non-Aadhaar document mix response | Input counts for Aadhaar-present and non-Aadhaar-only samples, with masked-PDF, HTTP 422, and error outcomes per category. |

Timing starts when the HTTP request is sent and ends after the full response body is read. It includes API processing, PDF creation, and response transfer over loopback. It excludes model startup and warm-ups. An HTTP 422 is a completed request and contributes to latency and TPS. A 200 response is counted as a masked PDF only when the response body is a PDF.

For a separate Nginx/network measurement, run a second report with `--url` set to the Nginx `/api/process` URL and label it as an end-to-end gateway measurement. Do not mix that result with the direct-backend metrics above.
