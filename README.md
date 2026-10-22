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
.\.venv\Scripts\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8002 --no-access-log
~~~

Terminal 2 - start the web app:

~~~powershell
npm run dev
~~~

Open the Vite URL printed in the second terminal. The current Vite configuration uses `http://127.0.0.1:5174/` and proxies `/api` requests to the API at `http://127.0.0.1:8002`.

The API provides `GET /api/health` for readiness and `POST /api/process` for one uploaded file. The browser's multi-file queue sends each file in its own request. A successful response is a flattened PDF with `Cache-Control: no-store, private`.

The browser also calls `POST /api/card-summary` with the validated, non-identifying result metadata described above. The Azure API key stays in the backend environment and is never sent to the browser.

If you do not already have a `.env`, copy `.env.example` to create one; otherwise, keep your existing credentials and add or update `AZURE_OPENAI_MODEL`. The setting must match the GPT-5-nano deployment name in your Azure resource. Keep `.env` local; it is ignored by Git.

## Runtime configuration

These environment variables can be used to configure the local service. Defaults below come from `backend/settings.py` and `backend/ocr.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AADHAAR_DOCUMENT_MODEL` | `backend/models/aadhaar-document-seg.pt` | Document segmentation checkpoint path |
| `AADHAAR_NUMBER_MODEL` | `backend/models/aadhaar-number-det.pt` | Number detector checkpoint path |
| `AADHAAR_YOLO_DEVICE` | `cpu` | YOLO device, such as `cpu` or `0` |
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
- The API does not persist source uploads as part of its normal processing flow and returns responses with `Cache-Control: no-store, private`.
- The API command disables Uvicorn access logs so uploaded filenames are not recorded in routine request logs.
- Do not add uploads, OCR text, or complete Aadhaar numbers to application logs, analytics, or error monitoring.
- Before exposing the service on a network, add authentication, TLS, rate limits, request isolation, appropriate temporary storage controls, and a retention/deletion policy. Run LibreOffice conversion for untrusted files in an isolated worker or container.
- Review Ultralytics licensing before commercial deployment; see the [Ultralytics license page](https://www.ultralytics.com/license).

## Accuracy and release status

The current checkpoints were trained on generated synthetic examples. Synthetic validation or test scores and a small set of application regression examples do not establish real-world accuracy. Before production use, train or review weights using consented and de-identified representative data, evaluate them on a locked identity-disjoint test set, and document false detections, missed number occurrences, and unsafe-output rates.

No vision system can safely accept every unclear or damaged image. This application should reject cases it cannot verify and mask safely.

