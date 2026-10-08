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

## 7. Model provenance, training history, size, and time

### Where the starting weights came from

We did **not** create the YOLO architecture or train the initial networks from random weights. The training code starts from Ultralytics YOLO26 nano checkpoints:

| Task | Starting checkpoint | Public pretraining | Local file size |
|---|---|---|---:|
| Aadhaar document segmentation | `yolo26n-seg.pt` | COCO-pretrained instance segmentation | 6,719,965 bytes (6.41 MiB) |
| Aadhaar number detection | `yolo26n.pt` | COCO-pretrained object detection | 5,544,453 bytes (5.29 MiB) |

Ultralytics documents these as official YOLO26 checkpoints, distributed through its model assets and downloaded by the `ultralytics` package when a named weight is not already present. Its YOLO26 documentation says the COCO checkpoints were themselves fine-tuned from Objects365 checkpoints. The project does not use Roboflow-hosted inference. See [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26), [YOLO instance segmentation](https://docs.ultralytics.com/tasks/segment), and the [official assets release](https://github.com/ultralytics/assets/releases).

The exact starting files in this workspace are identified by these SHA-256 digests:

```text
yolo26n-seg.pt  361FBFABAB285C3237700B6BB91D7ECFA602CD945FFFDA8DBE1242829B71E73F
yolo26n.pt      9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF
```

The version installed for the recorded runs is `ultralytics==8.4.172`. In `training/train.py`, `YOLO(base_model)` loads the starter checkpoint and `model.train(data=...)` fine-tunes it. “Fine-tuning” means retaining general visual features learned during public pretraining and adapting the model weights and task head to our own labels. The result is a task-specific checkpoint; it is not a new architecture and it is not the untouched public model.

The root of this workspace also contains `yolo26s-seg.pt` (23,467,933 bytes, 22.37 MiB). The current training configuration does not use it; the deployed document model is the nano segmentation checkpoint listed above.

### Why the first approach did not work well

An unchanged COCO model does not have an `aadhaar_front`, `aadhaar_back`, `aadhaar_letter`, or `aadhaar_number` class. It may find generic objects, but that is not enough to identify Aadhaar reliably or to draw the full physical boundary needed here. The public Roboflow dataset reviewed during early exploration had five entity classes with ordinary boxes; it did not provide the complete document polygon required by the rectification pipeline. It was not used to produce the current model pair.

The first task-specific experiments were also too weak to be treated as usable models. The document run used `yolo26n-seg.pt`, CPU, 512 px, batch 4, one recorded epoch, and freeze 10; it achieved validation mask mAP50 of 0.054. The first number run used `yolo26n.pt`, CPU, 384 px, batch 8, and freeze 5. Its arguments requested 15 epochs, but the results file records five and validation box mAP50 of 0.055. The run record does not establish why the number run ended after five epochs, so that cause should not be guessed in a client discussion.

We then generated a privacy-safe synthetic dataset, trained a longer document model, and continued number-model training from its first custom checkpoint. On the synthetic validation split, the later runs reached mask mAP50 0.881 for documents and box mAP50 0.995 for number regions. This fixed the prototype’s task and training problem on generated examples. It does **not** establish equivalent accuracy on real cameras, real print layouts, or every Aadhaar design.

### What data and labels were used

The local training generator created all current training, validation, and test images procedurally. The local dataset manifest records seed `20261005`, 768-pixel document scenes, and `contains_real_personal_data: false`.

The dataset can be regenerated with the recorded settings:

```powershell
python training\generate_synthetic_dataset.py --train 180 --val 36 --test 36 --scene-size 768 --seed 20261005
```

| Dataset | Train | Validation | Test | Labels |
|---|---:|---:|---:|---|
| Document segmentation | 180 images | 36 images | 36 images | 114 / 21 / 23 Aadhaar polygons; 81 / 18 / 15 negative images |
| Number detection | 180 images | 36 images | 36 images | 160 / 35 / 39 number boxes; 84 / 15 / 19 negative images |

The generated image files occupy 49,296,499 bytes in total (about 47.0 MiB); YOLO label files add very little. This dataset size is separate from the 11.31 MiB deployed checkpoint pair.

The generated layouts contain synthetic names and numbers, synthetic visual details, and generated hard negatives. The number generator creates checksum-valid synthetic values so the layout resembles the target task; they are not real issued identities. Images and labels were written by `training/generate_synthetic_dataset.py`, rather than manually annotated. The supplied private examples were kept out of these datasets and were used only as a small engineering acceptance/regression set, as described in [ARCHITECTURE.md](ARCHITECTURE.md).

The labels tell each model exactly what its output should be:

- The document model learns three classes and a polygon covering the complete front, back, or letter.
- The number model learns one class and a box around each printed 12-digit number occurrence.
- Negative examples have empty labels, teaching the model that a card-like rectangle or nearby text is not automatically Aadhaar.

Training, validation, and test examples were generated with separate deterministic seeds. This avoids putting the same generated image into more than one split, but it is not equivalent to splitting real people, source documents, cameras, or capture sessions. A client production dataset still needs consent, de-identification, identity/source-based separation, and reviewed annotations.

### Recorded fine-tuning runs

The local run argument files under `runs/` preserve the model names, settings, and dataset paths. Both final runs used CPU, PyTorch 2.8.0, Ultralytics 8.4.172, deterministic seed 42, and the generated 180-image training split. The `datasets/` and `runs/` directories are ignored by Git; the recipes and summary facts below were transcribed from this workspace and should be archived with future release evidence.

| Model | Starting point | Recorded training recipe | Approx. run time |
|---|---|---|---:|
| Document segmentation | `yolo26n-seg.pt` | 20 epochs, 384 px, batch 8, first 5 layers frozen; rotation 180°, translation 0.12, scale 0.45, perspective 0.0008, mosaic 0.35 | about 25 minutes |
| Number detection | First custom number checkpoint from `runs/aadhaar-number-det/weights/best.pt` | 15 more epochs, 512 px, batch 4, all layers trainable; rotation 6°, translation 0.04, scale 0.15, perspective 0.0001, mosaic 0.05 | about 27 minutes |

The different augmentation settings are intentional: document detection must tolerate arbitrary scene rotation and perspective; number detection sees a rectified document, so aggressive scene-level rotation would be a poor match. The current training workflow evaluates the selected `best.pt` checkpoint on the test split and copies it into `backend/models/`.

### Checkpoint size and integrity

These are the exact local `.pt` file sizes and SHA-256 digests observed in this workspace. The digest identifies the file; it is not a vendor signature. The two files loaded by the application total 11,855,650 bytes, or about 11.31 MiB on disk.

| Runtime checkpoint | Size | SHA-256 |
|---|---:|---|
| `backend/models/aadhaar-document-seg.pt` | 6,502,941 bytes (6.20 MiB) | `C86F5908C092E837A5B903F038966FEAE8B1629695D7BBD584F76E7F36EA4F5F` |
| `backend/models/aadhaar-number-det.pt` | 5,352,709 bytes (5.10 MiB) | `62CFBE8E1D125AA3A4AA83903ED14ED427DBBA1D92F77223C7E1598055BC65F7` |

The `.pt` sizes are checkpoint storage, not the full installation size or runtime memory requirement. PyTorch, PaddlePaddle, PaddleOCR model files, image buffers, and inference activations use additional disk space and memory.

The custom runtime `.pt` files are ignored by Git (`backend/models/*.pt`) and are not part of a normal source checkout. The files exist on this machine, but a fresh clone will need the separately delivered model artifacts or a training run before the API can become ready. The generated datasets and raw run folders are also ignored, so they are not a complete archived training package. For a client handoff, deliver approved weights through a controlled artifact channel and preserve the run arguments, dataset manifest, evaluation report, and verified hashes alongside each released model version.

### What the measured scores mean

The project report records these results for the held-out **synthetic** test split:

| Model | Metric | Reported value |
|---|---|---:|
| Document segmentation | Mask precision / recall | 0.981 / 0.946 |
| Document segmentation | Mask mAP50 / mAP50-95 | 0.982 / 0.959 |
| Number detection | Box precision / recall | 0.998 / 1.000 |
| Number detection | Box mAP50 / mAP50-95 | 0.995 / 0.667 |

The epoch-by-epoch `results.csv` files also record validation history. Their final-epoch metrics are lower for document masks (precision 0.834, recall 0.877, mAP50 0.881, mAP50-95 0.822) and are precision 0.997, recall 1.000, mAP50 0.995, mAP50-95 0.645 for number boxes. These figures describe different splits/checkpoint evaluations; they are not a real-world accuracy guarantee. The repository preserves training CSVs and test plots, but does not contain a per-image production evaluation report or machine-readable test metrics export.

The four supplied acceptance examples are useful regression cases, not a representative benchmark. The current evidence does not support a claim such as “99.5% accurate on Aadhaar documents.” A release claim needs an independently collected and locked real-world test set, including hard negatives, capture conditions, and every printed-number occurrence.

### How long training takes

On the CPU used for the recorded runs, training took about 25 minutes for the 20-epoch document run and 27 minutes for the 15-epoch number run—about 52 minutes in total for these small synthetic datasets at 384/512 pixels. The recorded test-evaluation folders were created in under 30 seconds in total. Dataset preparation, human annotation, dependency installation, and a future client’s dataset are not included.

The saved run arguments identify the device as CPU but do not record the processor model or system RAM, so the timing cannot be generalized to another machine from these logs alone. At the same CPU speed, data size, and image sizes, a simple linear estimate for 150 epochs is roughly 3.1 hours for documents plus 4.5 hours for numbers. Treat that only as a planning baseline: the current script defaults to 1280/960-pixel training and real datasets may be much larger, while a GPU can change throughput substantially. For a client estimate, run a short benchmark on the target training machine and multiply observed time per epoch by the planned epoch count and dataset size. Early stopping may shorten a run; training should be repeated when the data distribution or labels change.

### Direct answers a client may ask

**Did we build our own neural network?** No. We selected official Ultralytics YOLO26 nano starting checkpoints and fine-tuned them for our labels.

**Where did the base knowledge come from?** The document and number starters are Ultralytics’ COCO-pretrained YOLO26 nano models. Ultralytics documents the COCO checkpoints as fine-tuned from Objects365 checkpoints. Our Aadhaar-specific labels and data were added during local fine-tuning.

**Did we use real Aadhaar records for training?** No. The current model weights were trained on generated synthetic examples. The private examples were used for application regression checks and were not placed in the model training dataset.

**Are these models ready for production?** They are working bootstrap checkpoints, not production-certified models. Production readiness requires a consented real dataset, identity-disjoint test split, a locked evaluation report, security/deployment controls, and license review.

**Will the models send data to Roboflow or a cloud service?** No. Runtime inference loads the local `.pt` files. Roboflow is not part of the runtime path, and the explored public dataset was not used for these final weights.

**What does the model do when uncertain?** The wider pipeline cross-checks OCR, document identity, layout, checksum, and post-mask OCR. If it cannot establish a safe result, it rejects the file rather than returning a possibly unmasked PDF.

**What remains before client production use?** Collect and label representative consented examples, evaluate false positives and missed occurrences on a locked real-world test set, preserve a signed/versioned model report, benchmark the client’s hardware, and review the [Ultralytics license options](https://www.ultralytics.com/license) for commercial use.

## 8. Short answer for a CTO meeting

> We use Ultralytics YOLO because it gives a practical local deployment path for our detection tasks. We use YOLO segmentation for the document because we need the Aadhaar polygon, not just a rectangle, to remove nearby objects and correct perspective. We train custom weights because generic models do not understand our Aadhaar classes and hard negatives. We keep the number detector, OpenCV geometry, OCR, checksum validation, and post-mask audit as separate checks so one model failure does not directly create an unsafe output. The system accepts only when all required evidence agrees; otherwise it fails closed.

## 9. Architecture trade-offs and release conditions

The current architecture favors privacy, explainability, modular testing, and safe rejection over the smallest possible model or the highest possible acceptance rate.

Before an internet-facing or commercial release, the team should validate the reviewed weights on a locked, identity-disjoint real-world dataset, measure false Aadhaar detections and missed number occurrences, test unsafe-output rate, isolate Word conversion, add authentication and HTTPS, and confirm the Ultralytics licensing option.

