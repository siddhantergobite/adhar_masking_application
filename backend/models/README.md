# Runtime model contract and current status

The two `.pt` files presently in this working directory are synthetic bootstrap checkpoints. They were fine-tuned from official Ultralytics YOLO26 nano COCO-pretrained starting weights on procedurally generated examples. No real Aadhaar records were used to train them. They support local prototype and regression work but are not production-certified. Git ignores `*.pt` here, so a fresh checkout will not include these local checkpoints.

The runtime expects these two files and class contracts:

- `aadhaar-document-seg.pt` — YOLO instance-segmentation model with one or more of these classes: `aadhaar_front`, `aadhaar_back`, `aadhaar_letter`.
- `aadhaar-number-det.pt` — YOLO detection model with the class `aadhaar_number`.

The API intentionally remains unavailable when either file is absent or when the class names do not match. A generic COCO model, a card detector, or the public entity dataset shown in the project discussion is not an Aadhaar boundary model and will be rejected.

Use `training/train.py` to create replacement weights from reviewed, consented, de-identified datasets. Never commit model training images or real identity documents to this repository. See [CTO_APPLICATION_GUIDE.md](../../CTO_APPLICATION_GUIDE.md) for provenance, recorded training parameters, sizes, digests, and measured timings.

