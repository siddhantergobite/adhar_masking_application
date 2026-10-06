# Private Aadhaar Mask

A React single-page application and private FastAPI computer-vision service for isolating Aadhaar documents, correcting perspective, and irreversibly masking the first eight digits of verified Aadhaar numbers.

The application is intentionally fail-closed. It does not OCR the whole upload and mask arbitrary 12-digit strings.

For the complete system design, component responsibilities, model rationale, measured results, security controls, limitations, and CTO review questions, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Processing pipeline

1. Decode JPEG, PNG, WebP, BMP, TIFF, GIF, HEIC, PDF, DOC or DOCX input.
2. Detect only `aadhaar_front`, `aadhaar_back` or `aadhaar_letter` instances with a custom YOLO segmentation model, while OpenCV independently proposes physical document quadrilaterals.
3. Reconcile learned Aadhaar identity with the best supported physical boundary, convert it into four corners, and rectify it with an OpenCV homography. An OpenCV-only proposal remains untrusted until it passes stricter Aadhaar text, layout, number, and checksum verification. Surrounding objects—including debit cards—are removed from the result.
4. Verify Aadhaar identity using the trained class, OCR layout text, QR presence and confidence signals.
5. Localize printed Aadhaar-number regions with a second custom YOLO model.
6. Read the candidate with PaddleOCR word positioning; require 12 digits, a valid first digit and a Verhoeff checksum. The number detector is corroboration when available, while checksum-valid OCR word boxes remain the exact masking coordinates when a low-confidence detector pass misses.
7. Mask the first eight digit coordinates and retain the final four.
8. OCR the result again and reject the entire output if any complete valid Aadhaar number remains readable.
9. Return only a flattened PDF with no selectable source text.

Expected behavior:

- Debit-card-only photo: `No Aadhaar document was detected`; no output is created.
- Aadhaar plus debit card: only the Aadhaar polygon is cropped, rectified and processed.
- Full Aadhaar letter: the page is rectified and every detected printed Aadhaar-number occurrence must be masked.
- Uncertain, blurred, cut-off or partially processed input: rejected without an output.

## Required model weights

The API requires both files below and will report `models_missing` until they exist:

```text
backend/models/aadhaar-document-seg.pt
backend/models/aadhaar-number-det.pt
```

Weights are not fabricated or bundled from public identity documents. Follow [training/README.md](training/README.md) to annotate hard negatives, audit the datasets, train both models and run the release evaluation. The public five-class entity dataset discussed during development is not an outer-boundary segmentation dataset and is therefore not accepted as the document model.

## Install on Windows

From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
npm install
```

PyTorch, PaddlePaddle and the OCR weights are large. The first installation and first model initialization can take several minutes.

## Start the application

Use two PowerShell terminals.

Terminal 1 — FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8002 --no-access-log
```

Terminal 2 — React/Vite:

```powershell
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`. The interface displays whether both YOLO weights and PaddleOCR are ready. Vite proxies `/api` to the private FastAPI service on port 8002.

## Tests and production build

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
npm run build
```

The unit suite covers checksum validation, 16-digit payment-card rejection, polygon rectification, debit-only rejection and preservation of the final four digits. Model accuracy must additionally be measured on the locked real-world test set described in the training guide.

## Runtime configuration

Optional environment variables:

```text
AADHAAR_DOCUMENT_MODEL
AADHAAR_NUMBER_MODEL
AADHAAR_YOLO_DEVICE             # cpu, 0, 1, ...
AADHAAR_DOCUMENT_CONFIDENCE     # default 0.72
AADHAAR_NUMBER_CONFIDENCE       # default 0.62
AADHAAR_VERIFICATION_CONFIDENCE # default 0.68
AADHAAR_OCR_CONFIDENCE          # default 0.60
AADHAAR_INFERENCE_SIZE          # default 512 (CPU-safe for the bundled bootstrap weights)
AADHAAR_ORIENTATION_ATTEMPTS    # default 4; set lower only when latency is more important than rotation recovery
AADHAAR_OCR_DETECTION_MODEL     # default PP-OCRv5_mobile_det
AADHAAR_OCR_RECOGNITION_MODEL   # default en_PP-OCRv5_mobile_rec
AADHAAR_OCR_BATCH_SIZE           # default 8; CPU recognition batch size
AADHAAR_OCR_CPU_THREADS         # default up to 8; CPU OCR threads
AADHAAR_OCR_MAX_LONG_EDGE        # default 1200; OCR-only input cap with coordinate remapping
AADHAAR_DOCUMENT_FALLBACK_CONFIDENCE # default 0.12; second YOLO segmentation pass when the strict pass finds nothing
AADHAAR_NUMBER_FALLBACK_CONFIDENCE   # default 0.12; second YOLO number pass when the strict pass finds nothing
```

Thresholds are safety controls, not demo controls. Calibrate them against an identity-disjoint release set and do not lower them merely to make difficult samples pass.

## Privacy and deployment

- Processing is local by default and has no Roboflow or hosted inference call.
- API responses use `Cache-Control: no-store`; source bytes are kept only for the request.
- Access logs are disabled by the documented local command to avoid filenames appearing in routine logs.
- Do not store uploads, OCR text or full Aadhaar numbers in application logs, analytics or error monitoring.
- For a network deployment, add TLS, authentication, rate limits, request isolation, encrypted temporary storage and a retention/deletion policy at the infrastructure boundary.
- Run LibreOffice conversion for untrusted Word files in an isolated worker/container in a hosted deployment.

Ultralytics is distributed under AGPL-3.0 with a separate enterprise licensing option. Confirm licensing before a commercial release.

## Accuracy boundary

The code path is production-oriented, but the system is not production-certified merely because it runs. Release requires reviewed model weights and evidence from the locked test set. No computer-vision system can safely accept literally every blurred, cropped or obstructed image; a safe system rejects inputs it cannot mask with sufficient confidence.

