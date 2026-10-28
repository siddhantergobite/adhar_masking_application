# Aadhaar Masking Workbench

A local React/Vite web app and FastAPI service for isolating Aadhaar documents and creating masked PDF copies. The current interface supports a queue of multiple files: each file is processed separately and gets its own status, preview, and download.

This is a prototype. Its bundled YOLO checkpoints were fine-tuned on generated synthetic examples, not real Aadhaar records, and have not been certified for production use.

For the detailed system design and recorded regression cases, see [ARCHITECTURE.md](ARCHITECTURE.md). For model provenance and training records, see [CTO_APPLICATION_GUIDE.md](CTO_APPLICATION_GUIDE.md).

## What the app does

- Accepts JPG/JPEG, PNG, WebP, BMP, TIFF, GIF, HEIC/HEIF, PDF, DOC, and DOCX files.
- Accepts multiple files in one selection or by drag and drop. Files are queued and processed one at a time; they are not merged together.
- Supports multi-page images and documents, up to 10 pages per file. The upload limit is 25 MiB per file.
- Shows each file's original and masked preview, processing status, and a separate masked PDF download.
- Rejects a file if it cannot safely identify and mask an Aadhaar number. A debit-card-only file is expected to produce no output.

DOC and DOCX processing requires LibreOffice on the computer running the API. The browser can preview DOCX files, but the backend uses LibreOffice to convert Word documents before processing them.

## Processing flow

1. Decode the uploaded file into image pages.
2. Run the custom YOLO segmentation model for `aadhaar_front`, `aadhaar_back`, and `aadhaar_letter`. OpenCV independently proposes document-shaped quadrilaterals.
3. Reconcile Aadhaar detections with document geometry, crop the candidate, and correct perspective with OpenCV. OpenCV geometry alone does not establish Aadhaar identity; fallback candidates need stronger text, layout, number, and checksum evidence.
4. Read text and word positions with PaddleOCR. Check Aadhaar context and require a 12-digit number whose first digit is neither 0 nor 1, and whose Verhoeff checksum is valid.
5. Use a second YOLO model to find likely Aadhaar-number regions. OCR word positions supply the digit coordinates used for masking.
6. Cover the first eight digits of every verified number occurrence and retain the final four digits.
7. Run OCR again. If a complete valid number remains readable, reject the result.
8. Return a flattened, image-only PDF for the successfully processed file.

The pipeline is designed to fail closed on uncertain, cut-off, blurry, or partially processed documents. The checksum checks number format; the application does not authenticate an identity or verify a number against UIDAI.

Each result card also shows a conservative end-to-end pipeline score and a one- or two-sentence summary. For a successful file, the score is the weakest signal across document/boundary detection (including OpenCV agreement when available), scan quality relative to the configured blur threshold, OCR confidence for accepted number lines, Aadhaar verification, and number-region localization. Missing number-region support scores zero. The system returns a score only after digit-coordinate checks, masking, PDF creation, and the post-mask OCR audit pass; failures reject the file without a PDF. For multi-document files, the lowest document score is shown. This heuristic is not a calibrated probability of correctness and cannot prove OCR found every sensitive number. Calibrate it against human-reviewed, identity-disjoint examples before presenting it as a likelihood. A rejected file shows 0; an unavailable result shows “Not scored.”

When Azure OpenAI is configured, GPT-5-nano summarizes only validated result metadata (outcome, document class, counts, and confidence signals). Images, filenames, OCR text, and Aadhaar numbers are not sent for summaries. If the model call is unavailable, the card displays a local summary instead.

## Model files

The API loads these local checkpoints:

~~~text
backend/models/aadhaar-document-seg.pt
backend/models/aadhaar-number-det.pt
~~~

Both checkpoints are present in this working folder, but Git ignores `backend/models/*.pt`. A fresh checkout therefore needs the model files delivered separately or new weights trained before the API can become ready. The API reports `models_missing` if either file is absent and `model_error` if a loaded model has an incompatible class contract. The current weights are synthetic bootstrap models for local development and regression work, not production-approved models.

See [training/README.md](training/README.md) for dataset preparation, training, and release evaluation guidance. See [CTO_APPLICATION_GUIDE.md](CTO_APPLICATION_GUIDE.md) for model provenance, recorded training runs, checkpoint sizes, and limitations.

## Install on Windows

Run these commands from the project root:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
npm install
~~~

PyTorch, PaddlePaddle, and OCR model files take additional disk space. First-time installation and model initialization can take several minutes. PaddleOCR may need to fetch its model files the first time it is initialized.

## Start the application

Open two PowerShell terminals in the project root.

Terminal 1 - start the local API:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8002 --workers 1 --no-access-log
~~~

Terminal 2 - start the web app:

~~~powershell
npm run dev
~~~

Open the Vite URL printed in the second terminal. The current Vite configuration uses `http://127.0.0.1:5174/` and proxies `/api` requests to the API at `http://127.0.0.1:8002`.

The API provides `GET /api/health` for readiness and `POST /api/process` for one uploaded file. The browser's multi-file queue sends each file in its own request. A successful response is a flattened PDF with `Cache-Control: no-store, private`. The backend's inference lock serializes processing on CPU. For folder or multi-file jobs, use the batch routes below.

The browser also calls `POST /api/card-summary` with the validated, non-identifying result metadata described above. The Azure API key stays in the backend environment and is never sent to the browser.

If you do not already have a `.env`, copy `.env.example` to create one; otherwise, keep your existing credentials and add or update `AZURE_OPENAI_MODEL`. The setting must match the GPT-5-nano deployment name in your Azure resource. Keep `.env` local; it is ignored by Git.

## CPU-only batch API

The batch API accepts up to 30 supported files per batch and processes them **one at a time on CPU**. It reuses the same Aadhaar verification, masking, and post-mask OCR safety checks as the single-file route. A batch returns a job ID immediately; use the status route to see progress and download completed outputs. Batch state is stored in SQLite under the local batch storage folder. Uploaded copies are removed after processing; masked outputs and the manifest are retained for 24 hours by default.

The web app has a **Folder and Cloudflare Workers KV imports** panel. **Process local inbox** scans `AADHAAR_BATCH_INPUT_ROOT` directly, so a browser folder-upload control is not needed. Your configured inbox is `C:\Users\admin\Documents\AadhaarInbox`; the app scans its top level by default. You can store those local files in a private Workers KV namespace first, then separately import the stored files for local processing, or import values already stored under the KV inbox prefix.

Open the interactive API page at `http://127.0.0.1:8002/docs` while the local API is running. The main routes are:

| Method and route | Purpose |
| --- | --- |
| `POST /api/batches` | Upload several PDFs/images as repeated multipart `files` fields and queue them. |
| `POST /api/batches/folder` | Queue supported files from a subfolder of the configured local inbox. |
| `POST /api/batches/s3` | Queue supported files from a configured S3 input prefix. |
| `POST /api/batches/r2` | Queue supported files from the configured private Cloudflare R2 inbox prefix. |
| `POST /api/batches/r2/upload-folder` | Upload files from the configured local inbox to private R2 and process them. |
| `POST /api/batches/kv/store-folder` | Copy supported local inbox files to a private Workers KV prefix without starting masking. |
| `POST /api/batches/kv` | Queue supported files from the configured Workers KV inbox prefix, optionally limited to a relative key prefix. |
| `POST /api/batches/kv/upload-folder` | Store configured local inbox files in Workers KV and process them locally. |
| `GET /api/batches/{batch_id}` | Check batch and per-file status, rejection reason, score, and download links. |
| `GET /api/batches/{batch_id}/files/{file_id}/download` | Download one completed masked PDF. |
| `GET /api/batches/{batch_id}/download.zip` | Download successful PDFs plus a JSON manifest of all file outcomes. |
| `DELETE /api/batches/{batch_id}` | Delete a finished local batch and, when configured, its cloud output objects. |

The default local storage path on Windows is `%LOCALAPPDATA%\AadhaarMaskingWorkbench\batches`. Each job's masked PDFs and manifest are saved under `jobs\<batch_id>\output`. To keep results in a folder you can easily find, set `AADHAAR_BATCH_STORAGE_ROOT` in `.env` to a dedicated path under Documents, for example `C:\Users\admin\Documents\AadhaarMaskingResults`.

For a watched input folder, set `AADHAAR_BATCH_INPUT_ROOT` to a dedicated folder such as `C:\Users\admin\Documents\AadhaarInbox`, copy up to 30 source PDFs/images there, and restart the backend. The API only reads files inside this configured root. Then call `POST /api/batches/folder` with:

```json
{"relative_folder": ".", "recursive": false}
```

The route only reads inside that configured root; it does not accept arbitrary computer paths. It copies inputs into the private batch workspace and leaves the originals in the inbox.

Example from PowerShell, after the backend is running:

~~~powershell
$api = 'http://127.0.0.1:8002'
$headers = @{}
if ($env:AADHAAR_BATCH_API_KEY) { $headers['X-API-Key'] = $env:AADHAAR_BATCH_API_KEY }
$job = Invoke-RestMethod -Method Post -Uri "$api/api/batches/folder" `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body '{"relative_folder":".","recursive":false}'
$job

# Check progress later with the returned batch_id:
Invoke-RestMethod "$api/api/batches/$($job.batch_id)" -Headers $headers

# After it finishes, save the ZIP of successful PDFs and the result manifest:
Invoke-WebRequest "$api$($job.download_zip_url)" `
  -Headers $headers `
  -OutFile "$env:USERPROFILE\Documents\masked-batch.zip"
~~~

For direct API uploads, use `POST /api/batches` in the interactive page at `http://127.0.0.1:8002/docs` and add each PDF/image as a repeated `files` field. The status response includes a `batch_id`, per-file outcomes, and download links. A non-Aadhaar or uncertain file is reported as rejected and has no masked PDF.

Cloudflare Workers KV support is optional. Create a KV namespace, then set these values in the ignored backend `.env` file. Create a Cloudflare API token with **Workers KV Storage Edit** permission and keep it out of the browser and source control:

```text
CLOUDFLARE_KV_ACCOUNT_ID=your-account-id
CLOUDFLARE_KV_NAMESPACE_ID=your-kv-namespace-id
CLOUDFLARE_KV_API_TOKEN=your-scoped-api-token
CLOUDFLARE_KV_INPUT_PREFIX=aadhaar/inbox
CLOUDFLARE_KV_OUTPUT_PREFIX=aadhaar/masked
```

The input and masked-output prefixes must remain separate. Restart the backend. In the web app, click **Upload inbox to Workers KV** to store the supported top-level files from the configured local inbox without starting masking. The app shows the exact key prefix; you can inspect it under the namespace's **KV Pairs** tab. Then click **Process uploaded KV files** to fetch those values and run local CPU masking. Masked copies go under the separate output prefix and are also available through the local batch download route. Original KV inputs remain until removed from the namespace; masked KV copies follow the batch retention period. No public URL is created. KV operations require internet access; local processing remains available offline.

The Workers Free plan has limited quotas, including 1 GB of stored data, 25 MiB per value, and daily read/write limits. Keep the account on the Free plan and keep files within the published limits; over-limit operations fail on the Free plan. See Cloudflare's [Workers KV pricing and limits](https://developers.cloudflare.com/workers/platform/pricing/) before using client documents. Cloudflare KV is a small key-value service rather than an object-storage bucket; use it here only when the document set fits the per-value and namespace limits.

The backend calls Cloudflare's KV API directly; a separate deployed Worker is not required. Aadhaar detection, OCR, validation, and masking still run on the local CPU.

Cloudflare R2 support remains available as an optional connector. It uses a private R2 bucket, scoped R2 API token, and `backend/requirements-s3.txt`. Set these variables in the ignored `.env` file:

```text
CLOUDFLARE_R2_ACCOUNT_ID=your-account-id
CLOUDFLARE_R2_BUCKET=your-private-bucket
CLOUDFLARE_R2_ACCESS_KEY_ID=your-r2-access-key-id
CLOUDFLARE_R2_SECRET_ACCESS_KEY=your-r2-secret-access-key
CLOUDFLARE_R2_INPUT_PREFIX=aadhaar/inbox
CLOUDFLARE_R2_UPLOAD_PREFIX=aadhaar/uploads
CLOUDFLARE_R2_OUTPUT_PREFIX=aadhaar/masked
```

Keep the three R2 prefixes separate. Restart the backend. The R2 routes remain available to API clients; the web panel uses Workers KV. R2 input routes read from the configured private inbox prefix, and the R2 upload route copies local inbox files to the configured upload prefix before queueing them. Original files stay in the private bucket. Masked results are copied to the output prefix and are also available through the local batch download route. No public bucket or public link is created. R2 requires network access when used; local inbox processing works without it.

R2 keys must be kept private and should not be committed to source control. To give someone else temporary access to a result, use a controlled authenticated API or a short-lived presigned download URL; do not make the bucket public. The batch API currently returns private `s3://` object references for R2 outputs. Original objects remain in R2 until you delete them or apply a bucket lifecycle rule. Masked R2 outputs follow the batch retention setting and are removed after 24 hours by default; increase `AADHAAR_BATCH_RETENTION_HOURS` if the client needs a longer review window.

Generic S3 support remains available: install `backend/requirements-s3.txt`, configure `AADHAAR_S3_INPUT_BUCKET` plus a non-empty `AADHAAR_S3_INPUT_PREFIX`, and call `POST /api/batches/s3`. Set `AADHAAR_S3_OUTPUT_BUCKET` to copy successful masked PDFs back to S3. S3 ingestion/output needs network access when used; direct upload and local-folder processing do not.

The defaults are 30 files per batch, 25 MiB per file, 750 MiB total per batch, 90 pending/processing files across batches, and 24-hour result retention. Override them with the corresponding `AADHAAR_BATCH_*` settings in `.env`. `AADHAAR_BATCH_API_KEY` optionally protects the batch routes with an `X-API-Key` header; it should be set before allowing clients beyond the local machine. The documented local API still binds to `127.0.0.1`; it is not a public hosted service.

CPU inference is enforced by the API; YOLO uses `cpu`, PaddleOCR uses CPU, PyTorch/OpenCV/Paddle thread settings are capped at four, and one batch worker processes a single file at a time. This bounds inference concurrency, but does not promise a completion time. Benchmark a representative set before setting a client SLA. The current model weights remain synthetic bootstrap weights and need real-world, consented evaluation before production use.

## Runtime configuration

These environment variables can be used to configure the local service. Defaults below come from `backend/settings.py` and `backend/ocr.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AADHAAR_DOCUMENT_MODEL` | `backend/models/aadhaar-document-seg.pt` | Document segmentation checkpoint path |
| `AADHAAR_NUMBER_MODEL` | `backend/models/aadhaar-number-det.pt` | Number detector checkpoint path |
| `AADHAAR_YOLO_DEVICE` | CPU, forced | Runtime YOLO inference is CPU-only; GPU selection is not exposed |
| `AADHAAR_INFERENCE_SIZE` | `512` | YOLO inference image size |
| `AADHAAR_DOCUMENT_CONFIDENCE` | `0.72` | Primary document detection confidence |
| `AADHAAR_NUMBER_CONFIDENCE` | `0.62` | Primary number detection confidence |
| `AADHAAR_DOCUMENT_IOU` | `0.45` | YOLO overlap threshold |
| `AADHAAR_DOCUMENT_FALLBACK_CONFIDENCE` | `0.12` | Lower document threshold, used when the primary pass finds none |
| `AADHAAR_NUMBER_FALLBACK_CONFIDENCE` | `0.12` | Lower number threshold, used when the primary pass finds none |
| `AADHAAR_VERIFICATION_CONFIDENCE` | `0.68` | Minimum combined Aadhaar verification confidence |
| `AADHAAR_OCR_CONFIDENCE` | `0.60` | OCR confidence threshold |
| `AADHAAR_ORIENTATION_ATTEMPTS` | `4` | Maximum orientation attempts |
| `AADHAAR_OCR_MAX_LONG_EDGE` | `960` | First-pass OCR size cap; detected word coordinates are mapped back before masking |
| `AADHAAR_OCR_AUDIT_MAX_LONG_EDGE` | `1200` | Higher-resolution OCR pass used to audit the masked output |
| `AADHAAR_OCR_DETECTION_MODEL` | `PP-OCRv5_mobile_det` | PaddleOCR text detection model |
| `AADHAAR_OCR_RECOGNITION_MODEL` | `en_PP-OCRv5_mobile_rec` | PaddleOCR text recognition model |
| `AADHAAR_OCR_BATCH_SIZE` | `8` | OCR recognition batch size |
| `AADHAAR_OCR_CPU_THREADS` | Up to `4` | Shared CPU thread limit for PyTorch and PaddleOCR |
| `AADHAAR_MAX_UPLOAD_BYTES` | `26214400` | Maximum upload size in bytes (25 MiB) |
| `AADHAAR_MAX_PAGES` | `10` | Maximum pages in one file |
| `AADHAAR_MAX_PAGE_PIXELS` | `18000000` | Maximum decoded pixels per page |
| `AADHAAR_PDF_SCALE` | `2.5` | PDF page rendering scale |
| `AADHAAR_MIN_REGION_EDGE` | `220` | Minimum rectified document edge in pixels |
| `AADHAAR_MIN_BLUR_SCORE` | `14.0` | Minimum blur score for acceptance |
| `AZURE_OPENAI_ENDPOINT` | unset | Azure OpenAI resource endpoint used for short summaries only |
| `AZURE_OPENAI_API_KEY` | unset | Backend-only Azure OpenAI credential |
| `AZURE_OPENAI_MODEL` | `gpt-5-nano` | Azure deployment name for the summary model |

Confidence and quality thresholds are safety controls. Calibrate them against a reviewed, identity-disjoint evaluation set; do not lower them just to make difficult examples pass.

## Developer commands

Run the existing backend unit tests and build the web app with:

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
npm run build
~~~

## Privacy and deployment

- The documented local commands bind both services to `127.0.0.1`; image inference and masking use local YOLO checkpoints, OpenCV, and PaddleOCR. If configured, only result metadata is sent to Azure OpenAI to write the card summary.
- The single-file API does not persist source uploads. Batch processing keeps a temporary copy until that file finishes, then removes it; masked batch outputs and status are retained locally for 24 hours by default.
- The API command disables Uvicorn access logs so uploaded filenames are not recorded in routine request logs.
- Do not add uploads, OCR text, or complete Aadhaar numbers to application logs, analytics, or error monitoring.
- The optional `AADHAAR_BATCH_API_KEY` protects batch routes. Before exposing any part of the service on a network, add application-wide authentication, TLS, rate limits, request isolation, appropriate temporary storage controls, and a reviewed retention/deletion policy. Run LibreOffice conversion for untrusted files in an isolated worker or container. The current command binds to localhost and does not create a public hosted API.
- Review Ultralytics licensing before commercial deployment; see the [Ultralytics license page](https://www.ultralytics.com/license).

## Accuracy and release status

The current checkpoints were trained on generated synthetic examples. Synthetic validation or test scores and a small set of application regression examples do not establish real-world accuracy. Before production use, train or review weights using consented and de-identified representative data, evaluate them on a locked identity-disjoint test set, and document false detections, missed number occurrences, and unsafe-output rates.

No vision system can safely accept every unclear or damaged image. This application should reject cases it cannot verify and mask safely.

