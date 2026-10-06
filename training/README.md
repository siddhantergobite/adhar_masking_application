# Training the production models

The runtime is complete but deliberately refuses to process documents until two task-specific, reviewed weights are present. A generic rectangle detector or an OCR-only rule would recreate the debit-card false-positive problem.

## 1. Build the datasets

Keep all training data outside source control under:

```text
datasets/
  aadhaar-document-seg/
    images/{train,val,test}/
    labels/{train,val,test}/
  aadhaar-number/
    images/{train,val,test}/
    labels/{train,val,test}/
```

For `aadhaar-document-seg`, draw a tight polygon around the complete physical Aadhaar card or complete Aadhaar letter/page. Use only these classes:

- `0 aadhaar_front`
- `1 aadhaar_back`
- `2 aadhaar_letter`

For `aadhaar-number`, draw a tight box around each printed 12-digit Aadhaar number and use only `0 aadhaar_number`. A full letter may contain several instances; label every one.

Non-Aadhaar images must have an empty `.txt` label file. Include debit/credit cards, PAN cards, licences, visiting cards, blank rectangles and screenshots. Mixed scenes such as Aadhaar + debit card contain an Aadhaar polygon but no label around the debit card. Add rotations, perspective, glare, blur, compression, shadows, partial occlusion and difficult backgrounds in every split.

Split by identity/source capture rather than randomly by augmented image. Near-duplicates and images of the same document must never cross train/validation/test boundaries. Use consented and de-identified material; do not publish or commit real Aadhaar or payment-card data.

The public Roboflow project previously examined has five numeric entity classes and ordinary detection boxes. It can be research input after a privacy review, but it does not provide the outer segmentation contract required here.

For a privacy-safe local bootstrap (development and integration testing only), generate procedural Aadhaar-like layouts and hard negatives:

```powershell
.\.venv\Scripts\python.exe training\generate_synthetic_dataset.py
```

The generated images are explicitly marked synthetic and contain no source Aadhaar or payment-card data. They are useful for proving the complete pipeline and catching regressions, but they are not an acceptable substitute for a consented, identity-disjoint real-world production evaluation set.

## 2. Audit labels

```powershell
.\.venv\Scripts\python.exe training\validate_dataset.py datasets\aadhaar-document-seg --task segment --classes 3
.\.venv\Scripts\python.exe training\validate_dataset.py datasets\aadhaar-number --task detect --classes 1
```

## 3. Train and validate

Use a CUDA machine for serious training:

```powershell
.\.venv\Scripts\python.exe training\train.py --task all --device 0 --epochs 150 --batch 8
```

For a CPU-only bootstrap run:

```powershell
.\.venv\Scripts\python.exe training\train.py --task all --device cpu --epochs 30 --batch 4 --workers 0 --document-imgsz 512 --number-imgsz 512
```

The script validates the final checkpoints on the test split and copies them to:

```text
backend/models/aadhaar-document-seg.pt
backend/models/aadhaar-number-det.pt
```

Restart FastAPI after installing new weights. The API checks both filenames, task outputs and class names before becoming ready.

## 4. Release gate

Do not select weights from mAP alone. On a locked, identity-disjoint test set, separately report:

- Aadhaar document recall by front/back/letter and by capture condition.
- False Aadhaar detections on debit cards and all other hard negatives.
- Boundary mask IoU and four-corner rectification quality.
- Recall for every printed Aadhaar-number occurrence, especially full letters.
- End-to-end unsafe-output rate: any returned file containing a readable unmasked Aadhaar number.
- Rejection rate for blur, cut-off documents, glare and partial numbers.

Tune environment thresholds only against that locked set. A production system should reject uncertain inputs instead of lowering thresholds until every demo image passes.

Ultralytics is distributed under AGPL-3.0 with a separate enterprise licensing option. Confirm that the selected license fits the deployment before commercial release.

