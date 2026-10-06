# Aadhaar Masking Application - CTO System Architecture Guide

## 1. One-line architecture

This is a private, fail-closed document-processing pipeline: React/Vite handles the user interface, FastAPI handles requests, and a combination of custom YOLO models, OpenCV, PaddleOCR, validation, redaction, and a final safety audit produces a flattened masked PDF.

## 2. End-to-end architecture

```text
Browser
  -> React/Vite frontend
  -> FastAPI API
  -> File and page decoding
       |-> Custom YOLO document segmentation
       |-> OpenCV quadrilateral proposals
       -> Boundary reconciliation
       -> Perspective correction with OpenCV
       -> PaddleOCR text and word positions
       -> Custom YOLO Aadhaar-number detection
       -> Aadhaar identity and Verhoeff validation
       -> Mask first eight digits using OCR coordinates
       -> Post-mask OCR safety audit
       -> ReportLab flattened PDF
  -> Browser preview and download
```

The important design principle is **defense in depth**. No single model is trusted to identify, locate, read, and redact the document by itself.

## 3. What each component is responsible for

| Component | Responsibility | Why it exists |
|---|---|---|
| React and Vite | Upload, preview, status, download | Simple browser workflow |
| FastAPI | API boundary and request controls | Clear backend contract and error handling |
| YOLO document segmentation | Identify Aadhaar objects and return polygons | Semantic document identity plus an object boundary |
| OpenCV | Find physical quadrilaterals and rectify perspective | Accurate corners and document geometry |
| YOLO number detector | Find likely Aadhaar-number regions | Independent spatial evidence |
| PaddleOCR | Read text and return word coordinates | Exact number content and masking coordinates |
| Validation layer | Apply Aadhaar rules and Verhoeff checksum | Reject random or invalid 12-digit strings |
| Redaction layer | Cover digits 1 to 8 | Deterministic, pixel-level masking |
| Post-mask audit | OCR the result again | Reject any output where a complete number remains readable |
| ReportLab | Create image-only PDF | Avoid selectable source text or removable PDF annotations |

## 4. Terms the CTO may ask about

### What is YOLO?

YOLO means **You Only Look Once**. It is a family of one-stage computer-vision models that process an image in one inference flow and predict objects with their classes, confidence scores, and locations.

In this application, “YOLO” is the model family. The deployed implementation is Ultralytics YOLO with our own task-specific trained weights.

### What is YOLO segmentation?

Object detection normally returns a rectangle. YOLO segmentation returns a rectangle plus a pixel-level mask or polygon for each detected object.

Our document model uses instance segmentation because the application needs to isolate the complete Aadhaar boundary from a photograph that may also contain a debit card, table, screen, or background. The polygon is then converted into four corners for perspective correction.

### What is Ultralytics YOLO?

Ultralytics is the framework and implementation used to train and run the YOLO models. It provides the model API, segmentation support, training loop, augmentation, validation, checkpoint loading, and inference.

It is not a second detection concept separate from YOLO. A simple explanation is:

```text
YOLO = model family and approach
Ultralytics = framework/implementation used by this project
Custom weights = weights trained for our Aadhaar task
```

### What does custom segmentation mean here?

It means the model was trained for our specific classes and labels, rather than using a generic object detector unchanged.

The document model has these classes:

- `aadhaar_front`
- `aadhaar_back`
- `aadhaar_letter`

The model is based on an Ultralytics segmentation model, but its final weights are trained on Aadhaar-specific polygons and hard negatives. This is custom training and a custom task contract; it is not a completely new neural-network architecture written from scratch.

## 5. Main architecture questions and answers

### Why did we use YOLO?

YOLO is a good fit for local document processing because it gives a practical balance of inference speed, accuracy, deployment simplicity, and CPU/GPU support. The application needs fast object-level decisions and must run privately without sending documents to a hosted vision API.

It is not selected because it is universally better than every other model. It is selected because the current problem is a narrow object-detection/segmentation problem with a local processing requirement.

### Why YOLO segmentation instead of normal YOLO object detection?

A normal detector gives an axis-aligned rectangle. That rectangle may include a debit card, background, or nearby text and is not precise enough for a tilted or perspective-distorted Aadhaar.

Segmentation gives the document polygon. The polygon lets OpenCV calculate the physical corners and crop only the Aadhaar before OCR. This reduces false OCR text from surrounding objects.

### Why not use only a bounding box?

A bounding box is simpler and faster, but it includes pixels outside the document and does not describe the real four-corner boundary. It is acceptable for many detection tasks, but less suitable when the output must exclude nearby cards and when perspective correction matters.

### Why use custom segmentation instead of a generic pretrained model?

A generic model may know common objects such as a person, car, or document, but it does not reliably know the difference between an Aadhaar card, a debit card, and an Aadhaar letter in our required operating conditions.

Our custom dataset defines the exact classes, the complete outer boundary, mixed scenes, and hard negatives such as debit cards, PAN cards, licences, screenshots, blur, glare, rotation, and partial occlusion. Generic pretrained weights are used only as a starting point for transfer learning; the deployed checkpoint is task-specific.

### Why use two YOLO models instead of one?

The two models solve different problems at different image scales:

- The document model works on the original scene and finds the Aadhaar object.
- The number model works after rectification and finds printed Aadhaar-number regions.

Keeping them separate makes the training labels, image sizes, augmentations, thresholds, and validation metrics appropriate for each task. A single multi-purpose model could reduce model count, but it would combine scene-level document recognition with small-text localization and would be harder to tune safely.

The trade-off is an extra model and extra inference work. For a privacy-sensitive masking system, the independent evidence is more valuable than minimizing the number of models.

### Why is the document model segmentation, but the number model only detection?

For the document, the exact outer shape is important, so a mask is needed.

For the number, the model only needs to identify the likely number region. PaddleOCR supplies the actual word positions, and the redaction code divides those word boxes into digit positions. Training a number segmentation model would add annotation and complexity without being necessary for exact digit masking.

### Why use OpenCV if YOLO already detects the document?

YOLO provides semantic identity: "this is likely an Aadhaar." OpenCV provides geometric evidence: "these are likely the physical four corners." These are different responsibilities.

OpenCV can refine a YOLO boundary when the segmentation mask clips part of a physical page. An OpenCV-only rectangle is never trusted as Aadhaar by itself; it must pass stricter OCR, layout, checksum, and post-mask checks.

### Why use OCR if YOLO can detect the number region?

YOLO detection gives an approximate region. It does not reliably provide the exact characters or the exact boundary between digit 8 and digit 9.

PaddleOCR gives recognized text plus word-level coordinates. Those coordinates allow the redactor to cover digits 1 to 8 while preserving digits 9 to 12. OCR is therefore required for exact content validation and exact masking.

### Why not use OCR only?

OCR-only processing could find a 12-digit-looking string in a debit card, a screenshot, or unrelated text. It also does not reliably identify the complete physical Aadhaar boundary or remove surrounding objects.

The architecture uses vision for document identity and geometry, then OCR for text and coordinates. Each component covers a weakness of the other.

### Why not use regular expressions only?

A regular expression can check whether text looks like twelve digits, but it cannot establish that the text belongs to an Aadhaar document. It also cannot provide reliable pixel coordinates for the first eight digits.

The application combines structure, context, model confidence, OCR confidence, and the Verhoeff checksum instead of treating a digit pattern as proof.

### Why use the Verhoeff checksum?

The checksum is a mathematical validity check for the Aadhaar number format. It helps reject random OCR errors and unrelated 12-digit values.

It is not sufficient by itself. A number can pass a checksum and still not be Aadhaar, so the system also requires Aadhaar identity and layout evidence.

### Why not use Mask R-CNN, Faster R-CNN, or DETR?

Those models can be valid alternatives. The current choice prioritizes a practical local deployment, simpler training and inference integration, and reasonable latency for the narrow task.

Heavier two-stage or transformer-based models may be considered if the locked evaluation set shows that YOLO misses important document conditions and the additional compute and latency are acceptable. The architecture is replaceable at the detector boundary; the validation and redaction contracts can remain the same.

### Why not use SAM or a generic segmentation model?

A generic segmentation model can help produce a mask, but it does not automatically provide Aadhaar identity, document classes, or hard-negative behavior. Some such models are also interactive or computationally heavier.

Our requirement is not only "segment a rectangle." It is "recognize an Aadhaar, isolate the correct instance, reject similar cards, and produce a safe output." A task-specific segmentation model is a better contract for that requirement.

### Why not build one end-to-end model that detects and redacts everything?

An end-to-end model might be smaller in terms of visible pipeline steps, but it would be harder to explain, test, debug, and audit. A model error could directly become an unsafe output.

The current modular design allows each stage to be tested independently and allows deterministic rules to stop the pipeline before a PDF is returned.

### Why is the system fail-closed?

For privacy redaction, an unsafe false acceptance is more serious than a safe rejection. If the document is blurry, the boundary is uncertain, OCR is incomplete, or the safety audit finds a readable full number, the system returns no PDF.

### Why run OCR after masking?

The first OCR pass helps locate and validate the number. The second pass is a safety audit of the actual output.

This catches repeated number occurrences, insufficient mask width, and other cases where the first pass appeared successful but a complete Aadhaar number remains readable.

### Why create a flattened PDF?

A normal PDF annotation or overlay may be removable or may leave the original text layer available. The application renders the processed document as pixels and creates a new image-only PDF, so the original selectable source text is not carried into the result.

## 6. Concrete model and code contract

### Document model

- Runtime: `UltralyticsDocumentSegmenter`
- Task: YOLO instance segmentation
- Classes: front, back, and letter
- Output: class, confidence, bounding box, and polygon
- Weight: `backend/models/aadhaar-document-seg.pt`
- Training base: `yolo26n-seg.pt`

### Number model

- Runtime: `UltralyticsNumberDetector`
- Task: YOLO object detection
- Class: `aadhaar_number`
- Output: confidence and bounding box
- Weight: `backend/models/aadhaar-number-det.pt`
- Training base: `yolo26n.pt`

### Supporting modules

- `backend/pipeline.py` - orchestration and fail-closed decisions
- `backend/detectors.py` - Ultralytics YOLO integration
- `backend/candidates.py` - OpenCV geometric proposals
- `backend/geometry.py` - corner ordering and perspective correction
- `backend/ocr.py` - PaddleOCR integration
- `backend/validation.py` - identity signals and Verhoeff validation
- `backend/redaction.py` - exact first-eight-digit masking
- `backend/server.py` - FastAPI endpoints

## 7. Short answer for a CTO meeting

> We use Ultralytics YOLO because it gives a practical local deployment path for our detection tasks. We use YOLO segmentation for the document because we need the Aadhaar polygon, not just a rectangle, to remove nearby objects and correct perspective. We train custom weights because generic models do not understand our Aadhaar classes and hard negatives. We keep the number detector, OpenCV geometry, OCR, checksum validation, and post-mask audit as separate checks so one model failure does not directly create an unsafe output. The system accepts only when all required evidence agrees; otherwise it fails closed.

## 8. Architecture trade-offs and release conditions

The current architecture favors privacy, explainability, modular testing, and safe rejection over the smallest possible model or the highest possible acceptance rate.

Before an internet-facing or commercial release, the team should validate the reviewed weights on a locked, identity-disjoint real-world dataset, measure false Aadhaar detections and missed number occurrences, test unsafe-output rate, isolate Word conversion, add authentication and HTTPS, and confirm the Ultralytics licensing option.

