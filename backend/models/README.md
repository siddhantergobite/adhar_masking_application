# Runtime model contract

Place the two reviewed production weights in this directory:

- `aadhaar-document-seg.pt` — YOLO instance-segmentation model with one or more of these classes: `aadhaar_front`, `aadhaar_back`, `aadhaar_letter`.
- `aadhaar-number-det.pt` — YOLO detection model with the class `aadhaar_number`.

The API intentionally remains unavailable when either file is absent or when the class names do not match. A generic COCO model, a card detector, or the public entity dataset shown in the project discussion is not an Aadhaar boundary model and will be rejected.

Use `training/train.py` to create weights from reviewed, consented, de-identified datasets. Never commit model training images or real identity documents to this repository.

