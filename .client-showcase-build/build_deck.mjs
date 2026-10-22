import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const root = process.cwd();
const buildDir = path.join(root, ".client-showcase-build");
const outputDir = path.join(root, "output");
const previewDir = path.join(buildDir, "draft-previews");
const draftPath = path.join(buildDir, "Aadhaar_Masking_Equitas_Client_Showcase.draft.pptx");

const C = {
  bg: "#F6F8F5",
  paper: "#FFFFFF",
  ink: "#17342F",
  muted: "#5D716A",
  green: "#206B4D",
  green2: "#16844C",
  pale: "#E8F1EA",
  line: "#D5E2D9",
  soft: "#EFF4F0",
  coral: "#B55243",
  coralPale: "#F9ECE8",
  white: "#FFFFFF",
};

const FONT = "Arial";

function addText(slide, name, x, y, w, h, value, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    typeface: FONT,
    fontSize: 20,
    color: C.ink,
    alignment: "left",
    verticalAlignment: "middle",
    wrap: "square",
    autoFit: "shrinkText",
    insets: 0,
    ...style,
  };
  return shape;
}

function addRect(slide, name, x, y, w, h, fill, lineFill = "none", radius = 0) {
  const shape = slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: lineFill, width: lineFill === "none" ? 0 : 1.2 },
    ...(radius ? { borderRadius: radius } : {}),
  });
  return shape;
}

function addLine(slide, name, x1, y1, x2, y2, color = C.line, width = 1) {
  return slide.shapes.add({
    geometry: "line",
    name,
    position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function addArrow(slide, name, x, y, w = 34, h = 22) {
  return slide.shapes.add({
    geometry: "rightArrow",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: C.green2,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function connect(slide, a, b, fromSide = "right", toSide = "left", kind = "straight") {
  return slide.shapes.connect(a, b, {
    kind,
    fromSide,
    toSide,
    line: { style: "solid", fill: C.green2, width: 2 },
    tail: { type: "arrow", width: "sm", length: "sm" },
  });
}

function addHeader(slide, title, subtitle) {
  addText(slide, "slide-title", 64, 40, 1152, 58, title, {
    fontSize: 45,
    bold: true,
    color: C.ink,
    verticalAlignment: "middle",
  });
  addText(slide, "slide-subtitle", 66, 100, 1148, 32, subtitle, {
    fontSize: 20,
    color: C.muted,
  });
}

function addFooter(slide, page) {
  addLine(slide, "footer-rule", 64, 666, 1216, 666, C.line, 1);
  addText(slide, "footer-label", 64, 675, 820, 20, "AADHAAR MASKING WORKBENCH  ·  CLIENT SHOWCASE", {
    fontSize: 13,
    bold: true,
    color: C.muted,
  });
  addText(slide, "page-number", 1150, 675, 66, 20, `${String(page).padStart(2, "0")} / 05`, {
    fontSize: 13,
    bold: true,
    color: C.green,
    alignment: "right",
  });
}

function addProcessNode(slide, name, x, y, w, h, step, title, detail, options = {}) {
  const fill = options.fill ?? C.paper;
  const stroke = options.stroke ?? C.line;
  const shape = addRect(slide, name, x, y, w, h, fill, stroke, 12);
  addText(slide, `${name}-step`, x + 14, y + 9, 42, 22, step, {
    fontSize: 14,
    bold: true,
    color: options.stepColor ?? C.green,
  });
  addText(slide, `${name}-title`, x + 14, y + 31, w - 28, 28, title, {
    fontSize: options.titleSize ?? 20,
    bold: true,
    color: options.textColor ?? C.ink,
  });
  addText(slide, `${name}-detail`, x + 14, y + 60, w - 28, h - 67, detail, {
    fontSize: options.detailSize ?? 17,
    color: options.detailColor ?? C.muted,
    verticalAlignment: "top",
  });
  return shape;
}

async function addLocalImage(slide, name, filePath, alt, x, y, w, h) {
  const bytes = new Uint8Array(await fs.readFile(filePath));
  addRect(slide, `${name}-frame`, x, y, w, h, C.paper, C.line, 9);
  return slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit: "contain",
    geometry: "roundRect",
    borderRadius: 7,
    position: { left: x + 5, top: y + 5, width: w - 10, height: h - 10 },
  });
}

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });

// 1 — Project overview
{
  const slide = deck.slides.add();
  slide.background.fill = C.bg;
  addRect(slide, "cover-accent", 0, 0, 18, 720, C.green, "none");
  addText(slide, "cover-kicker", 72, 76, 700, 26, "CLIENT SHOWCASE  ·  EQUITAS BANK DISCUSSION", {
    fontSize: 16,
    bold: true,
    color: C.green,
  });
  addText(slide, "cover-title", 72, 146, 850, 150, "Aadhaar document\nmasking", {
    fontSize: 66,
    bold: true,
    color: C.ink,
    verticalAlignment: "top",
  });
  addRect(slide, "cover-rule", 74, 326, 112, 6, C.green2, "none", 3);
  addText(slide, "cover-summary", 72, 364, 770, 92,
    "A local AI pipeline that finds Aadhaar documents, verifies number lines and masks the first eight digits before PDF delivery.",
    { fontSize: 25, color: C.muted, verticalAlignment: "top" });
  addText(slide, "cover-stack-label", 72, 526, 450, 26, "PROJECT STACK", {
    fontSize: 14,
    bold: true,
    color: C.green,
  });
  addText(slide, "cover-stack", 72, 557, 730, 42, "YOLO26  ·  OpenCV  ·  PaddleOCR  ·  Verhoeff validation", {
    fontSize: 20,
    color: C.ink,
  });
  addLine(slide, "cover-footer-rule", 72, 650, 1208, 650, C.line, 1);
  addText(slide, "cover-footer", 72, 666, 900, 20, "Working prototype  ·  Synthetic training data  ·  Local processing", {
    fontSize: 14,
    color: C.muted,
  });
  addText(slide, "cover-page-number", 1138, 666, 70, 20, "01 / 05", {
    fontSize: 13,
    bold: true,
    color: C.green,
    alignment: "right",
  });
  slide.speakerNotes.text =
    "Introduce this as the current local Aadhaar masking prototype. It detects the full document, verifies the number through OCR and checks, masks the first eight digits, then returns a flattened PDF. The model weights were trained on generated examples, not real Aadhaar records; production readiness is not claimed.";
}

// 2 — Two user supplied before/after samples
{
  const slide = deck.slides.add();
  slide.background.fill = C.bg;
  addHeader(slide, "Before and masked output", "Two sample captures processed by the current local pipeline.");

  const examples = [
    {
      x: 64,
      name: "single-card",
      heading: "01  SINGLE CARD PHOTO",
      before: path.join(buildDir, "before_single_card.png"),
      after: path.join(buildDir, "single_card_masked.png"),
      score: "76 / 100",
      caption: "One Aadhaar document · one number line redacted",
      beforeAlt: "Original user-provided Aadhaar card photo, cropped to the card.",
      afterAlt: "Masked PDF page created by the local application from the single-card photo.",
    },
    {
      x: 656,
      name: "mixed-capture",
      heading: "02  MIXED CARD SCENE",
      before: path.join(buildDir, "before_mixed_cards_aadhaar_crop.png"),
      after: path.join(buildDir, "mixed_cards_masked.png"),
      score: "64 / 100",
      caption: "Aadhaar isolated · one number line redacted",
      beforeAlt: "Original user-provided mixed-scene photo, cropped to the Aadhaar document.",
      afterAlt: "Masked Aadhaar-only PDF page; the unrelated debit card is excluded.",
    },
  ];

  for (const ex of examples) {
    addText(slide, `${ex.name}-heading`, ex.x, 154, 552, 30, ex.heading, {
      fontSize: 19,
      bold: true,
      color: C.green,
    });
    const leftX = ex.x + 16;
    const frameY = 224;
    const frameW = 222;
    const frameH = 238;
    const rightX = ex.x + 316;
    addText(slide, `${ex.name}-before-label`, leftX, 190, frameW, 24, "BEFORE  ·  ORIGINAL", {
      fontSize: 14,
      bold: true,
      color: C.muted,
    });
    addText(slide, `${ex.name}-after-label`, rightX, 190, frameW, 24, "AFTER  ·  MASKED COPY", {
      fontSize: 14,
      bold: true,
      color: C.muted,
    });
    await addLocalImage(slide, `${ex.name}-before`, ex.before, ex.beforeAlt, leftX, frameY, frameW, frameH);
    addArrow(slide, `${ex.name}-arrow`, ex.x + 266, frameY + 106, 34, 24);
    await addLocalImage(slide, `${ex.name}-after`, ex.after, ex.afterAlt, rightX, frameY, frameW, frameH);
    addText(slide, `${ex.name}-score`, rightX, 476, frameW, 30, `Pipeline score  ${ex.score}`, {
      fontSize: 19,
      bold: true,
      color: C.green,
    });
    addText(slide, `${ex.name}-caption`, ex.x + 16, 510, 520, 42, ex.caption, {
      fontSize: 18,
      color: C.ink,
      verticalAlignment: "top",
    });
  }
  addText(slide, "before-after-footnote", 66, 614, 1146, 24,
    "The mixed-scene result contains the Aadhaar page only; the unrelated debit card is not included in the output.",
    { fontSize: 16, color: C.muted });
  addFooter(slide, 2);
  slide.speakerNotes.text =
    "The two input photographs were supplied by the user. The output pages were generated locally from those photographs by the current application pipeline; each run produced one Aadhaar document and redacted one number line. The displayed 76/100 and 64/100 values are the actual conservative pipeline scores for these two runs. The mixed-scene output contains the Aadhaar only, not the debit card. Personal document imagery is included at the user's request for this client showcase.";
}

// 3 — Editable, native-object system flow
{
  const slide = deck.slides.add();
  slide.background.fill = C.bg;
  addHeader(slide, "System processing flow", "Document detection, independent number checks and a fail-closed masking audit.");
  addText(slide, "flow-lane-one", 66, 145, 290, 24, "01  FIND AND RECTIFY", {
    fontSize: 15,
    bold: true,
    color: C.green,
  });
  const upload = addProcessNode(slide, "upload", 64, 174, 170, 98, "01", "Upload + decode", "Image, PDF or document");
  const detect = addProcessNode(slide, "boundary", 270, 174, 190, 98, "02", "Aadhaar boundary", "YOLO segmentation + OpenCV quadrilaterals");
  const rectify = addProcessNode(slide, "rectify", 496, 174, 170, 98, "03", "Rectify + quality", "Perspective, size and orientation checks");
  const read = addProcessNode(slide, "read", 702, 174, 190, 98, "04", "Read number", "OCR positions + YOLO boxes");
  const verify = addProcessNode(slide, "verify", 928, 174, 288, 98, "05", "Verhoeff + identity gate", "12 digits, first-digit rule and Aadhaar context", {
    fill: C.pale,
    stroke: "#B7D4BF",
  });
  connect(slide, upload, detect);
  connect(slide, detect, rectify);
  connect(slide, rectify, read);
  connect(slide, read, verify);

  addText(slide, "flow-lane-two", 66, 371, 360, 24, "02  MASK, THEN VERIFY THE OUTPUT", {
    fontSize: 15,
    bold: true,
    color: C.green,
  });
  const locate = addProcessNode(slide, "locate-digits", 976, 404, 240, 98, "06", "Locate digit boxes", "Use OCR word-level coordinates");
  const mask = addProcessNode(slide, "mask-digits", 712, 404, 240, 98, "07", "Mask digits 1–8", "Leave only the final four visible", {
    fill: C.pale,
    stroke: "#B7D4BF",
  });
  const audit = addProcessNode(slide, "audit", 448, 404, 240, 98, "08", "Post-mask OCR audit", "A full number must no longer be readable");
  const pdf = addProcessNode(slide, "pdf-output", 184, 404, 240, 98, "09", "Flattened PDF", "Preview and download", {
    fill: C.green,
    stroke: C.green,
    textColor: C.white,
    detailColor: "#E8F2EC",
    stepColor: "#D2E9D8",
  });
  connect(slide, verify, locate, "bottom", "top", "elbow");
  connect(slide, locate, mask, "left", "right");
  connect(slide, mask, audit, "left", "right");
  connect(slide, audit, pdf, "left", "right");

  addRect(slide, "reject-callout", 390, 550, 520, 54, C.coralPale, "#E6C5BC", 10);
  addText(slide, "reject-callout-text", 412, 559, 476, 36,
    "Uncertain verification or a readable full number after masking → reject; no PDF is returned.",
    { fontSize: 17, bold: true, color: C.coral, alignment: "center" });
  addFooter(slide, 3);
  slide.speakerNotes.text =
    "Source: SYSTEM_DIAGRAM.md and the current backend pipeline. YOLO segmentation proposes Aadhaar document boundaries; OpenCV proposes geometric quadrilaterals and helps rectify perspective. PaddleOCR supplies text and word-level coordinates, while the separate YOLO number detector supplies an additional number-region proposal. The verification stage checks the twelve-digit shape, the first-digit rule, Verhoeff checksum and Aadhaar-specific context. The checksum alone does not prove identity. The app masks digits 1–8, runs OCR on the masked result, and returns a flattened PDF only when the audit passes; uncertain inputs fail closed.";
}

// 4 — Fine-tuning and checkpoint storage
{
  const slide = deck.slides.add();
  slide.background.fill = C.bg;
  addHeader(slide, "YOLO26 fine-tuning and model size", "COCO-pretrained nano starters were updated on task-specific synthetic labels; the network architecture stayed YOLO26.");

  const row1y = 151;
  const row2y = 277;
  addText(slide, "seg-task-label", 66, row1y + 5, 192, 27, "DOCUMENT SEGMENTATION", {
    fontSize: 15,
    bold: true,
    color: C.green,
  });
  addText(slide, "seg-task-classes", 66, row1y + 36, 192, 65, "Classes: aadhaar_front\naadhaar_back · aadhaar_letter", {
    fontSize: 17,
    color: C.muted,
    verticalAlignment: "top",
  });
  addRect(slide, "seg-base-box", 280, row1y, 285, 102, C.paper, C.line, 10);
  addText(slide, "seg-base-label", 298, row1y + 12, 250, 22, "COCO STARTER", {
    fontSize: 14,
    bold: true,
    color: C.muted,
  });
  addText(slide, "seg-base-model", 298, row1y + 37, 250, 31, "yolo26n-seg.pt", {
    fontSize: 24,
    bold: true,
    color: C.ink,
  });
  addText(slide, "seg-base-size", 298, row1y + 70, 250, 23, "6.41 MiB", {
    fontSize: 17,
    color: C.muted,
  });
  addArrow(slide, "seg-finetune-arrow", 586, row1y + 39, 42, 24);
  addRect(slide, "seg-custom-box", 648, row1y, 310, 102, C.pale, "#B7D4BF", 10);
  addText(slide, "seg-custom-label", 666, row1y + 12, 275, 22, "FINE-TUNED CHECKPOINT", {
    fontSize: 14,
    bold: true,
    color: C.green,
  });
  addText(slide, "seg-custom-model", 666, row1y + 37, 275, 31, "aadhaar-document-seg.pt", {
    fontSize: 21,
    bold: true,
    color: C.ink,
  });
  addText(slide, "seg-custom-size", 666, row1y + 70, 275, 23, "6.20 MiB", {
    fontSize: 17,
    color: C.green,
  });
  addText(slide, "seg-labeled-polygons", 978, row1y + 18, 236, 72, "Learns full-card or letter polygons for front, back and letter documents.", {
    fontSize: 18,
    color: C.muted,
    verticalAlignment: "top",
  });

  addText(slide, "number-task-label", 66, row2y + 5, 192, 27, "NUMBER DETECTION", {
    fontSize: 15,
    bold: true,
    color: C.green,
  });
  addText(slide, "number-task-classes", 66, row2y + 36, 192, 65, "Class: aadhaar_number", {
    fontSize: 17,
    color: C.muted,
    verticalAlignment: "top",
  });
  addRect(slide, "num-base-box", 280, row2y, 285, 102, C.paper, C.line, 10);
  addText(slide, "num-base-label", 298, row2y + 12, 250, 22, "COCO STARTER", {
    fontSize: 14,
    bold: true,
    color: C.muted,
  });
  addText(slide, "num-base-model", 298, row2y + 37, 250, 31, "yolo26n.pt", {
    fontSize: 24,
    bold: true,
    color: C.ink,
  });
  addText(slide, "num-base-size", 298, row2y + 70, 250, 23, "5.29 MiB", {
    fontSize: 17,
    color: C.muted,
  });
  addArrow(slide, "num-finetune-arrow", 586, row2y + 39, 42, 24);
  addRect(slide, "num-custom-box", 648, row2y, 310, 102, C.pale, "#B7D4BF", 10);
  addText(slide, "num-custom-label", 666, row2y + 12, 275, 22, "FINE-TUNED CHECKPOINT", {
    fontSize: 14,
    bold: true,
    color: C.green,
  });
  addText(slide, "num-custom-model", 666, row2y + 37, 275, 31, "aadhaar-number-det.pt", {
    fontSize: 21,
    bold: true,
    color: C.ink,
  });
  addText(slide, "num-custom-size", 666, row2y + 70, 275, 23, "5.10 MiB", {
    fontSize: 17,
    color: C.green,
  });
  addText(slide, "number-boxes", 978, row2y + 18, 236, 72, "Locates each printed Aadhaar number line; this final run continued from an earlier custom checkpoint.", {
    fontSize: 18,
    color: C.muted,
    verticalAlignment: "top",
  });

  addLine(slide, "model-detail-divider", 64, 399, 1216, 399, C.line, 1);
  addText(slide, "storage-total", 66, 418, 250, 33, "11.31 MiB", {
    fontSize: 29,
    bold: true,
    color: C.green,
  });
  addText(slide, "storage-caption", 66, 453, 250, 50, "Combined size of the two custom .pt checkpoints.\nCheckpoint size only; runtime dependencies add storage.", {
    fontSize: 16,
    color: C.muted,
    verticalAlignment: "top",
  });
  addText(slide, "training-data-heading", 352, 418, 330, 28, "TRAINING DATA", {
    fontSize: 15,
    bold: true,
    color: C.green,
  });
  addText(slide, "training-data-copy", 352, 451, 330, 119,
    "Each task: 180 train · 36 validation · 36 test images.\n47.0 MiB of generated images in total.\nNo real Aadhaar records used for training.",
    { fontSize: 17, color: C.ink, verticalAlignment: "top" });
  addText(slide, "recipe-heading", 720, 418, 460, 28, "RECORDED FINAL RUNS", {
    fontSize: 15,
    bold: true,
    color: C.green,
  });
  addText(slide, "recipe-copy", 720, 451, 496, 140,
    "Segmentation: 20 epochs · 384 px · batch 8 · first 5 layers frozen · ~25 min.\nNumber detector: 15 more epochs from prior custom weights · 512 px · batch 4 · all layers trainable · ~27 min.\nFine-tuning updates learned weights; it does not append another model.",
    { fontSize: 17, color: C.ink, verticalAlignment: "top" });
  addText(slide, "pt-note", 66, 612, 1150, 22, "The .pt extension marks a PyTorch checkpoint. These are synthetic bootstrap weights, not production-certified models.", {
    fontSize: 15,
    color: C.muted,
  });
  addFooter(slide, 4);
  slide.speakerNotes.text =
    "Exact checkpoint sizes come from backend/models and CTO_APPLICATION_GUIDE.md. The base files are the official Ultralytics YOLO26 nano segmentation and detection weights. Ultralytics model.train() fine-tuned the task-specific weights and heads. The document model used 20 epochs, 384 px, batch 8, freezing the first five layers. The final number model continued from the earlier custom number checkpoint for 15 more epochs at 512 px and batch 4 with all layers trainable. Per task, the generated set had 180 train, 36 validation and 36 test images; no real Aadhaar records were used. The .pt extension marks a PyTorch checkpoint, not pytest. The reported disk size covers only the checkpoint files, not PyTorch, PaddleOCR and other runtime components. Check Ultralytics commercial licensing before external deployment. Official documentation: https://docs.ultralytics.com/models/yolo26 ; https://docs.ultralytics.com/tasks/segment ; https://docs.ultralytics.com/tasks/detect .";
}

// 5 — Model metrics, confidence meaning and a proposed Equitas pilot
{
  const slide = deck.slides.add();
  slide.background.fill = C.bg;
  addHeader(slide, "AI metrics and Equitas pilot", "Prototype evidence today; a measured, human-reviewed document-intelligence pilot next.");

  addText(slide, "metrics-heading", 66, 148, 540, 30, "HELD-OUT SYNTHETIC MODEL METRICS", {
    fontSize: 16,
    bold: true,
    color: C.green,
  });
  addText(slide, "metrics-columns", 66, 185, 544, 24, "TASK                     PRECISION / RECALL            mAP50  /  mAP50–95", {
    fontSize: 14,
    bold: true,
    color: C.muted,
  });
  addLine(slide, "metric-row-rule-1", 66, 216, 608, 216, C.line, 1);
  addText(slide, "metric-doc", 66, 222, 544, 40, "Document masks       .981 / .946                      .982 / .959", {
    fontSize: 18,
    color: C.ink,
  });
  addLine(slide, "metric-row-rule-2", 66, 266, 608, 266, C.line, 1);
  addText(slide, "metric-number", 66, 272, 544, 40, "Number boxes           .998 / 1.000                    .995 / .667", {
    fontSize: 18,
    color: C.ink,
  });
  addText(slide, "metrics-caveat", 66, 320, 540, 34, "36 synthetic test images per task; not real-world accuracy.", {
    fontSize: 16,
    color: C.muted,
  });

  addLine(slide, "score-divider", 650, 146, 650, 354, C.line, 1);
  addText(slide, "score-heading", 688, 148, 510, 30, "CURRENT UI CONFIDENCE SCORE", {
    fontSize: 16,
    bold: true,
    color: C.green,
  });
  addText(slide, "score-formula", 688, 188, 500, 60, "100 × min(document confidence,\nverification confidence)", {
    fontSize: 25,
    bold: true,
    color: C.ink,
    verticalAlignment: "top",
  });
  addText(slide, "score-exceptions", 688, 258, 500, 40, "Rejected / no safe output = 0  ·  missing score data = unavailable", {
    fontSize: 17,
    color: C.muted,
    verticalAlignment: "top",
  });
  addText(slide, "score-caveat", 688, 302, 500, 48, "A conservative pipeline signal, not a calibrated probability or end-to-end accuracy claim.", {
    fontSize: 17,
    bold: true,
    color: C.coral,
    verticalAlignment: "top",
  });

  addLine(slide, "opportunity-rule", 64, 374, 1216, 374, C.line, 1);
  addText(slide, "equitas-opportunity-heading", 66, 393, 560, 34, "PROPOSED AI / ML EXTENSIONS FOR EQUITAS", {
    fontSize: 19,
    bold: true,
    color: C.green,
  });
  addText(slide, "equitas-opportunity-context", 66, 427, 1146, 28,
    "Mapped to Equitas’s public digital onboarding, lending and API context; these are proposals, not existing Equitas features.",
    { fontSize: 16, color: C.muted });

  const columns = [
    {
      x: 66,
      title: "KYC / loan-pack intake",
      text: "Classify documents, extract fields with OCR and flag missing or unreadable pages.",
    },
    {
      x: 456,
      title: "Privacy masking",
      text: "Extend redaction to PAN and account values in review copies and case attachments.",
    },
    {
      x: 846,
      title: "Guided human review",
      text: "Surface cross-document mismatches and duplicates for staff review; do not auto-decide credit.",
    },
  ];
  for (const [index, item] of columns.entries()) {
    addText(slide, `equitas-feature-${index + 1}-title`, item.x, 474, 345, 30, item.title, {
      fontSize: 20,
      bold: true,
      color: C.ink,
    });
    addText(slide, `equitas-feature-${index + 1}-body`, item.x, 510, 345, 68, item.text, {
      fontSize: 17,
      color: C.muted,
      verticalAlignment: "top",
    });
  }
  addLine(slide, "feature-divider-1", 430, 472, 430, 580, C.line, 1);
  addLine(slide, "feature-divider-2", 820, 472, 820, 580, C.line, 1);
  addRect(slide, "pilot-recommendation", 66, 590, 1148, 48, C.pale, "#B7D4BF", 9);
  addText(slide, "pilot-recommendation-text", 84, 597, 1112, 34,
    "First pilot: KYC pack completeness + PII masking. Measure field accuracy, unsafe outputs, false accepts/rejects and review minutes.",
    { fontSize: 16, bold: true, color: C.ink, alignment: "center" });
  addFooter(slide, 5);
  slide.speakerNotes.text =
    "Prototype metrics are from CTO_APPLICATION_GUIDE.md and ARCHITECTURE.md. The final synthetic test evaluation reported document mask precision/recall 0.981/0.946, mask mAP50 0.982 and mask mAP50–95 0.959; number-box precision/recall 0.998/1.000, mAP50 0.995 and mAP50–95 0.667. These are synthetic-set results, not estimates of real camera or production performance. The UI score is round(100 × min(document detector confidence, verification confidence)); no safe output maps to zero, while unavailable inputs remain unscored. It is not calibrated. Equitas context is based on its FY2024–25 Integrated Annual Report and public Developer Portal API catalogue. The suggested KYC/loan-pack pilot is a proposal subject to Equitas security, legal and architecture review, and should keep human operators responsible for decisions. Sources: https://ir.equitas.bank.in/wp-content/uploads/2025/08/Equitas-AR-2024-25-1.pdf ; https://developer.equitas.bank.in/ .";
}

for (const [index, slide] of deck.slides.items.entries()) {
  const png = await deck.export({ slide, format: "png", scale: 1 });
  const output = path.join(previewDir, `slide-${String(index + 1).padStart(2, "0")}.png`);
  await fs.writeFile(output, new Uint8Array(await png.arrayBuffer()));
}

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(draftPath);
console.log(JSON.stringify({ draftPath, previewDir, slides: deck.slides.items.length }, null, 2));
