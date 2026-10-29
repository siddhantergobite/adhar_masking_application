from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np

from .domain import NumberDetection, OcrLine, ProcessedDocument, ProcessingResult, VisionDetection
from .errors import NoAadhaarDetected, NoAadhaarNumber, UnsafeDocument
from .geometry import (
    box_iou,
    enhance_for_ocr,
    overlap_over_first,
    polygon_box,
    rectify_polygon,
    rotate_quarter,
    validate_region_quality,
)
from .ocr import read_ocr_lines
from .redaction import mask_first_eight
from .settings import Settings
from .validation import (
    extract_aadhaar_digits,
    has_aadhaar_identity_anchor,
    has_pan_identity_anchor,
    qr_present,
    verify_aadhaar,
)


class _NotAadhaarCandidate(Exception):
    """Internal signal for a card-shaped object without Aadhaar identity text."""


class AadhaarPipeline:
    def __init__(
        self,
        document_segmenter: Any,
        number_detector: Any,
        ocr_engine: Any,
        settings: Settings,
        document_proposer: Any | None = None,
    ) -> None:
        self.document_segmenter = document_segmenter
        self.number_detector = number_detector
        self.ocr_engine = ocr_engine
        self.settings = settings
        self.document_proposer = document_proposer

    def process(self, pages: list[np.ndarray]) -> ProcessingResult:
        detected_any = False
        processed: list[ProcessedDocument] = []
        first_failure: UnsafeDocument | NoAadhaarNumber | None = None

        for page in pages:
            primary_detections = self._deduplicate_documents(self.document_segmenter.detect(page))
            proposals = (
                self._deduplicate_documents(self.document_proposer.detect(page))
                if self.document_proposer is not None
                else []
            )
            using_geometric_fallback = not primary_detections and bool(proposals)
            if primary_detections:
                detections = self._refine_document_boundaries(primary_detections, proposals)
                # Keep independent quadrilateral proposals available when a
                # low-confidence YOLO hypothesis fails.  A proposal is still
                # treated as untrusted by _process_detection; it can only
                # produce output after the same Aadhaar verification checks.
                detections = self._deduplicate_documents([*detections, *proposals])
            elif using_geometric_fallback:
                detections = proposals
            else:
                detections = []

            # When the strict YOLO result and the sole OpenCV proposal agree
            # on one strong document boundary, the verified crop already has
            # independent geometry confirmation. Keep the full-page OCR pass
            # for ambiguous scenes, multiple candidates, and weaker matches.
            single_reconciled_candidate = (
                len(primary_detections) == 1
                and primary_detections[0].confidence >= self.settings.document_confidence
                and len(proposals) == 1
                and len(detections) == 1
                and box_iou(primary_detections[0].box, proposals[0].box) >= 0.80
            )

            # A model box can be valid Aadhaar identity evidence while still
            # clipping the physical page. Treat the complete image as a
            # last-chance candidate, after the learned/geometric candidates.
            page_results: list[ProcessedDocument] = []
            page_failures: list[UnsafeDocument | NoAadhaarNumber] = []
            page_has_aadhaar_back_without_number = False
            for detection in detections:
                try:
                    result = self._process_detection(page, detection)
                    if result is not None:
                        page_results.append(result)
                    elif detection.class_name == "aadhaar_back":
                        page_has_aadhaar_back_without_number = True
                except _NotAadhaarCandidate:
                    # A card-shaped PAN or other document is not an Aadhaar
                    # failure. Ignore it so the final message says no Aadhaar
                    # was found when no Aadhaar candidate passes OCR checks.
                    continue
                except NoAadhaarNumber as exc:
                    # _process_detection raises this only after OCR finds
                    # Aadhaar-specific identity text.
                    if not using_geometric_fallback and detection.confidence >= self.settings.document_confidence:
                        page_failures.append(exc)
                except UnsafeDocument as exc:
                    # Non-Aadhaar proposals are filtered before this point;
                    # retain safety failures for any geometry source.
                    if not using_geometric_fallback and detection.confidence >= self.settings.document_confidence:
                        page_failures.append(exc)

            # A verified crop is authoritative.  Running a full-frame rescue
            # after it succeeds can replace the crop with the entire camera
            # photo (for example, an Aadhaar beside a debit card).  Full-frame
            # recovery is only appropriate when no bounded candidate passed.
            fallback_failure: UnsafeDocument | NoAadhaarNumber | None = None
            used_full_frame_fallback = False
            if self.document_proposer is not None and not page_results:
                try:
                    fallback_result = self._process_detection(
                        page,
                        self._full_frame_detection(page),
                    )
                except _NotAadhaarCandidate:
                    fallback_result = None
                except NoAadhaarNumber as exc:
                    fallback_result = None
                    fallback_failure = exc
                except UnsafeDocument as exc:
                    fallback_result = None
                    fallback_failure = exc
                if fallback_result is not None:
                    # A complete-page result supersedes partial candidates;
                    # otherwise a clipped result could leave a repeated
                    # Aadhaar number outside the returned crop.
                    page_results = [fallback_result]
                    page_failures = []
                    used_full_frame_fallback = True

            if (
                self.document_proposer is not None
                and len(page_results) == 1
                and not page_failures
                and not used_full_frame_fallback
                and not single_reconciled_candidate
            ):
                # A full-page candidate is considered only when it proves
                # that the bounded crop missed an additional verified number.
                # Equal-count results stay cropped so nearby debit cards and
                # the surrounding camera scene never enter the PDF.
                try:
                    broader_result = self._process_detection(
                        page,
                        self._full_frame_detection(page),
                    )
                except (_NotAadhaarCandidate, NoAadhaarNumber, UnsafeDocument):
                    broader_result = None
                if (
                    broader_result is not None
                    and (
                        broader_result.masked_regions > page_results[0].masked_regions
                        or (
                            broader_result.masked_regions == page_results[0].masked_regions
                            and page_has_aadhaar_back_without_number
                            and self._detections_tile_page(primary_detections, page.shape)
                        )
                    )
                ):
                    page_results = [broader_result]

            if page_results:
                processed.extend(page_results)
                detected_any = True
                if page_failures:
                    first_failure = first_failure or page_failures[0]
            elif page_failures or fallback_failure is not None:
                detected_any = True
                if page_failures:
                    first_failure = first_failure or page_failures[0]
                elif fallback_failure is not None:
                    first_failure = first_failure or fallback_failure
            elif page_has_aadhaar_back_without_number:
                detected_any = True
                first_failure = first_failure or NoAadhaarNumber()

        if not detected_any:
            raise NoAadhaarDetected()
        if first_failure is not None:
            # Never return a partially processed upload when another Aadhaar
            # instance could not be redacted safely.
            raise first_failure
        if not processed:
            raise NoAadhaarNumber()
        return ProcessingResult(documents=tuple(processed))

    @staticmethod
    def _full_frame_detection(image: np.ndarray) -> VisionDetection:
        height, width = image.shape[:2]
        polygon = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        long_edge = max(width, height)
        short_edge = max(1, min(width, height))
        class_name = "aadhaar_front" if long_edge / short_edge >= 1.45 else "aadhaar_letter"
        return VisionDetection(
            class_name=class_name,
            confidence=0.50,
            polygon=polygon,
            box=np.array([0, 0, width - 1, height - 1], dtype=np.float32),
            source="full_frame_fallback",
        )

    @staticmethod
    def _deduplicate_documents(detections: list[VisionDetection]) -> list[VisionDetection]:
        accepted: list[VisionDetection] = []
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(box_iou(candidate.box, existing.box) >= 0.72 for existing in accepted):
                continue
            accepted.append(candidate)
        return sorted(accepted, key=lambda item: (float(item.box[1]), float(item.box[0])))

    @staticmethod
    def _detections_tile_page(
        detections: list[VisionDetection],
        image_shape: tuple[int, ...],
    ) -> bool:
        """Identify aligned Aadhaar panels that together fill one upload.

        A two-sided Aadhaar may be photographed as front-over-back. When the
        detector returns one box per panel, the full-page OCR crop can retain
        both sides even if it finds the same number of redaction lines as the
        lower panel crop. This exception only applies when confident YOLO
        boxes are adjacent, horizontally aligned, and span almost the page.
        """

        candidates = [item for item in detections if item.confidence >= 0.50]
        if len(candidates) < 2:
            return False

        image_height, image_width = image_shape[:2]
        ordered = sorted(candidates, key=lambda item: (float(item.box[1]), float(item.box[0])))
        left = min(float(item.box[0]) for item in ordered)
        top = min(float(item.box[1]) for item in ordered)
        right = max(float(item.box[2]) for item in ordered)
        bottom = max(float(item.box[3]) for item in ordered)
        union_width = max(0.0, right - left)
        union_height = max(0.0, bottom - top)
        page_area = max(1.0, float(image_width * image_height))
        if union_width * union_height / page_area < 0.86:
            return False
        if (
            left > image_width * 0.08
            or top > image_height * 0.08
            or image_width - right > image_width * 0.08
            or image_height - bottom > image_height * 0.08
        ):
            return False

        for first, second in zip(ordered, ordered[1:]):
            first_width = max(1.0, float(first.box[2] - first.box[0]))
            second_width = max(1.0, float(second.box[2] - second.box[0]))
            horizontal_overlap = max(
                0.0,
                min(float(first.box[2]), float(second.box[2]))
                - max(float(first.box[0]), float(second.box[0])),
            )
            aligned = horizontal_overlap / min(first_width, second_width) >= 0.65
            min_height = max(
                1.0,
                min(
                    float(first.box[3] - first.box[1]),
                    float(second.box[3] - second.box[1]),
                ),
            )
            vertical_gap = max(0.0, float(second.box[1] - first.box[3]))
            if not aligned or vertical_gap > min_height * 0.12:
                return False
        return True

    @staticmethod
    def _refine_document_boundaries(
        detections: list[VisionDetection],
        proposals: list[VisionDetection],
    ) -> list[VisionDetection]:
        """Use a corroborating physical rectangle for cleaner four corners.

        YOLO still decides that the object is Aadhaar. OpenCV only replaces
        its often-truncated mask polygon when a strongly overlapping outer
        card/page boundary is available.
        """

        refined: list[VisionDetection] = []
        for detection in detections:
            compatible: list[tuple[float, VisionDetection]] = []
            for proposal in proposals:
                same_shape = (
                    proposal.class_name == "aadhaar_letter"
                    if detection.class_name == "aadhaar_letter"
                    else proposal.class_name == "aadhaar_front"
                )
                if not same_shape:
                    continue
                overlap = box_iou(detection.box, proposal.box)
                if overlap >= 0.55:
                    compatible.append((overlap, proposal))
            if not compatible:
                refined.append(detection)
                continue
            best_overlap, proposal = max(compatible, key=lambda item: item[0])
            refined.append(
                VisionDetection(
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    polygon=proposal.polygon,
                    box=proposal.box,
                    source=detection.source,
                    boundary_agreement=best_overlap,
                )
            )
        return refined

    def _orientation_order(self, image: np.ndarray, class_name: str) -> tuple[int, ...]:
        height, width = image.shape[:2]
        expects_landscape = class_name in {"aadhaar_front", "aadhaar_back"}
        already_expected = (width >= height) if expects_landscape else (height >= width)
        return (0, 2, 1, 3) if already_expected else (1, 3, 0, 2)

    def _process_detection(
        self,
        page: np.ndarray,
        detection: VisionDetection,
    ) -> ProcessedDocument | None:
        rectified = rectify_polygon(page, detection.polygon)
        quality_score = validate_region_quality(
            rectified.image,
            min_edge=self.settings.min_region_edge,
            # Blur is a useful warning signal, but it is not a reliable
            # decision when a bad inner contour has been selected. Exact OCR
            # coordinates plus the post-mask audit are the authoritative
            # safety checks, so defer the blur rejection until after OCR.
            min_blur_score=0.0,
        )
        blur_below_threshold = quality_score < self.settings.min_blur_score

        saw_valid_number = False
        saw_number_region = False
        saw_verification_failure = False
        saw_aadhaar_identity = False
        boundary_confidence = float(detection.confidence)
        if detection.boundary_agreement is not None:
            boundary_confidence = min(boundary_confidence, detection.boundary_agreement)
        blur_threshold = float(self.settings.min_blur_score)
        scan_quality_score = min(quality_score / blur_threshold, 1.0) if blur_threshold > 0 else 1.0

        orientation_order = self._orientation_order(rectified.image, detection.class_name)
        orientation_attempts = max(1, min(4, int(getattr(self.settings, "orientation_attempts", 4))))
        ocr_max_long_edge = max(320, int(getattr(self.settings, "ocr_max_long_edge", 1200)))
        for turns in orientation_order[:orientation_attempts]:
            oriented = rotate_quarter(rectified.image, turns)
            enhanced = enhance_for_ocr(oriented)
            lines = read_ocr_lines(
                self.ocr_engine,
                enhanced,
                max_long_edge=ocr_max_long_edge,
            )
            valid_lines = self._valid_number_lines(lines)
            aadhaar_identity_anchor = has_aadhaar_identity_anchor(lines)
            if has_pan_identity_anchor(lines) and not aadhaar_identity_anchor:
                continue
            if not aadhaar_identity_anchor and not valid_lines:
                # Do not classify a generic card-shaped proposal as Aadhaar
                # from its boundary alone. OCR may miss the Hindi Aadhaar
                # label, so a Verhoeff-valid number proceeds to independent
                # layout, model, and masking checks below.
                continue
            saw_aadhaar_identity = True
            saw_valid_number = saw_valid_number or bool(valid_lines)

            if not valid_lines:
                # Number detection only adds value after OCR has found a
                # checksum-valid candidate. Skipping it on failed OCR passes
                # avoids several expensive CPU model calls for one document.
                continue

            number_regions = self._deduplicate_numbers(self.number_detector.detect(oriented))
            saw_number_region = saw_number_region or bool(number_regions)

            number_detector_supported = False
            number_detector_confidence: float | None = None
            if number_regions:
                matched_lines, _, number_detector_confidence = self._match_number_regions(
                    valid_lines,
                    number_regions,
                )
                # OCR already supplied checksum-valid text and word-level
                # coordinates. The number detector is useful corroboration,
                # but a low-confidence/spurious box must not override exact
                # OCR coordinates or block a safe mask.
                number_detector_supported = bool(matched_lines)
            verification = verify_aadhaar(
                lines,
                detector_confidence=detection.confidence,
                has_qr=qr_present(oriented),
                valid_number_count=len(valid_lines),
                threshold=self.settings.verification_confidence,
                detection_source=detection.source,
                number_detector_supported=number_detector_supported,
            )
            if not verification.accepted:
                saw_verification_failure = True
                continue

            masked = oriented.copy()
            # Every checksum-valid OCR line is masked using its own word
            # coordinates. Detector overlap is independent corroboration, not
            # a reason to leave another valid occurrence exposed.
            for line in valid_lines:
                mask_first_eight(masked, line)

            self._assert_no_readable_aadhaar_remains(masked)
            # OCR may need a quarter-turn to read a difficult capture, but
            # the returned document should stay in the rectified boundary's
            # canonical orientation. This also turns a rotated replacement
            # label back into the normal `XXXX XXXX 1234` presentation.
            canonical_masked = rotate_quarter(masked, (-turns) % 4)
            return ProcessedDocument(
                image=canonical_masked,
                document_class=detection.class_name,
                document_confidence=detection.confidence,
                verification_confidence=verification.confidence,
                masked_regions=len(valid_lines),
                boundary_confidence=boundary_confidence,
                scan_quality_score=scan_quality_score,
                ocr_confidence=min(line.confidence for line in valid_lines),
                number_detector_confidence=number_detector_confidence,
                raw_blur_score=quality_score,
            )

        if not saw_aadhaar_identity:
            raise _NotAadhaarCandidate()
        if detection.class_name == "aadhaar_back" and not saw_valid_number and not saw_number_region:
            # A back-only card is recognized as Aadhaar, but contains no number
            # line this service can safely redact.
            return None
        if saw_verification_failure:
            raise UnsafeDocument(
                "The boundary model found a card, but Aadhaar identity checks did not agree strongly enough."
            )
        if saw_valid_number and not saw_number_region:
            raise UnsafeDocument(
                "A valid Aadhaar number was visible, but the Aadhaar-number detector did not localize it."
            )
        if saw_number_region and not saw_valid_number:
            raise UnsafeDocument(
                "The number region was found, but OCR could not validate all twelve digits and their checksum."
            )
        if blur_below_threshold:
            raise UnsafeDocument("The Aadhaar is too blurred to mask safely.")
        raise NoAadhaarNumber()

    def _valid_number_lines(self, lines: list[OcrLine]) -> list[OcrLine]:
        # PaddleOCR can split a large, slightly blurred Aadhaar number into
        # three separate text lines even when the fragments share one visual
        # baseline. Reassemble only numeric-only fragments that have their
        # own word boxes; the normal checksum and identity checks still decide
        # whether the merged candidate is usable.
        candidate_lines = [*lines, *self._merge_split_number_lines(lines)]
        accepted: list[OcrLine] = []
        for line in sorted(candidate_lines, key=lambda item: item.confidence, reverse=True):
            if line.confidence < self.settings.ocr_confidence:
                continue
            if extract_aadhaar_digits(line.text) is None:
                continue
            line_bounds = polygon_box(line.polygon)
            if any(box_iou(line_bounds, polygon_box(existing.polygon)) >= 0.78 for existing in accepted):
                continue
            accepted.append(line)
        return sorted(accepted, key=lambda item: (float(polygon_box(item.polygon)[1]), float(polygon_box(item.polygon)[0])))

    @staticmethod
    def _numeric_fragment(line: OcrLine) -> str | None:
        """Return the digits in a coordinate-safe numeric OCR fragment."""

        if not line.words:
            return None
        digits: list[str] = []
        for character in line.text:
            try:
                digits.append(str(unicodedata.decimal(character)))
            except (TypeError, ValueError):
                if character not in " \t\r\n-\u2010\u2011\u2012\u2013\u2014":
                    return None

        if not 2 <= len(digits) <= 8:
            return None

        word_digits: list[str] = []
        for word in line.words:
            for character in word.text:
                try:
                    word_digits.append(str(unicodedata.decimal(character)))
                except (TypeError, ValueError):
                    if character not in " \t\r\n-\u2010\u2011\u2012\u2013\u2014":
                        return None
        if "".join(digits) != "".join(word_digits):
            return None
        return "".join(digits)

    @classmethod
    def _merge_split_number_lines(cls, lines: list[OcrLine]) -> list[OcrLine]:
        fragments: list[tuple[OcrLine, str, np.ndarray]] = []
        for line in lines:
            digits = cls._numeric_fragment(line)
            if digits is None:
                continue
            bounds = polygon_box(line.polygon)
            fragments.append((line, digits, bounds))
        fragments.sort(key=lambda item: (float(item[2][1]), float(item[2][0])))

        merged: list[OcrLine] = []
        for first_index, (first_line, _, first_bounds) in enumerate(fragments):
            group = [(first_line, first_bounds)]
            total_digits = len(cls._numeric_fragment(first_line) or "")
            previous_bounds = first_bounds
            for line, digits, bounds in fragments[first_index + 1 :]:
                if float(bounds[0]) <= float(previous_bounds[0]):
                    continue
                previous_height = max(1.0, float(previous_bounds[3] - previous_bounds[1]))
                current_height = max(1.0, float(bounds[3] - bounds[1]))
                vertical_tolerance = max(previous_height, current_height) * 0.45
                same_baseline = abs(float(bounds[1] - previous_bounds[1])) <= vertical_tolerance
                horizontal_gap = max(0.0, float(bounds[0] - previous_bounds[2]))
                horizontal_limit = max(previous_height, current_height) * 2.0
                if not same_baseline or horizontal_gap > horizontal_limit:
                    if float(bounds[1]) > float(previous_bounds[3]):
                        break
                    continue
                if total_digits + len(digits) > 12:
                    break
                group.append((line, bounds))
                total_digits += len(digits)
                previous_bounds = bounds
                if total_digits == 12:
                    break

            if len(group) < 2 or total_digits != 12:
                continue
            group_lines = [item[0] for item in group]
            points = np.concatenate(
                [np.asarray(item.polygon, dtype=np.float32).reshape(-1, 2) for item in group_lines],
                axis=0,
            )
            merged.append(
                OcrLine(
                    text=" ".join(item.text for item in group_lines),
                    confidence=min(item.confidence for item in group_lines),
                    polygon=np.array(
                        [
                            [points[:, 0].min(), points[:, 1].min()],
                            [points[:, 0].max(), points[:, 1].min()],
                            [points[:, 0].max(), points[:, 1].max()],
                            [points[:, 0].min(), points[:, 1].max()],
                        ],
                        dtype=np.float32,
                    ),
                    words=tuple(word for item in group_lines for word in item.words),
                )
            )
        return merged

    @staticmethod
    def _deduplicate_numbers(detections: list[NumberDetection]) -> list[NumberDetection]:
        accepted: list[NumberDetection] = []
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(box_iou(candidate.box, existing.box) >= 0.70 for existing in accepted):
                continue
            accepted.append(candidate)
        return accepted

    @staticmethod
    def _expanded_box(box: np.ndarray, image_shape: tuple[int, ...] | None = None) -> np.ndarray:
        left, top, right, bottom = np.asarray(box, dtype=np.float32).reshape(4)
        width = right - left
        height = bottom - top
        expanded = np.array(
            [left - width * 0.08, top - height * 0.30, right + width * 0.08, bottom + height * 0.30],
            dtype=np.float32,
        )
        if image_shape is not None:
            image_height, image_width = image_shape[:2]
            expanded[[0, 2]] = np.clip(expanded[[0, 2]], 0, image_width - 1)
            expanded[[1, 3]] = np.clip(expanded[[1, 3]], 0, image_height - 1)
        return expanded

    def _match_number_regions(
        self,
        lines: list[OcrLine],
        detections: list[NumberDetection],
    ) -> tuple[list[OcrLine], int, float | None]:
        matched: list[OcrLine] = []
        used_line_ids: set[int] = set()
        unresolved = 0
        support_scores: list[float] = []
        for detection in detections:
            region = self._expanded_box(detection.box)
            choices: list[tuple[float, float, int, OcrLine]] = []
            for index, line in enumerate(lines):
                if index in used_line_ids:
                    continue
                line_bounds = polygon_box(line.polygon)
                overlap = overlap_over_first(line_bounds, region)
                if overlap >= 0.42:
                    choices.append((overlap + line.confidence * 0.1, overlap, index, line))
            if not choices:
                unresolved += 1
                continue
            _, overlap, index, line = max(choices, key=lambda item: item[0])
            used_line_ids.add(index)
            matched.append(line)
            support_scores.append(min(float(detection.confidence), overlap))

        if not matched or not lines:
            return matched, unresolved, None
        coverage = len(used_line_ids) / len(lines)
        return matched, unresolved, min(min(support_scores), coverage)

    def _assert_no_readable_aadhaar_remains(self, masked: np.ndarray) -> None:
        audit_lines = read_ocr_lines(
            self.ocr_engine,
            enhance_for_ocr(masked),
            max_long_edge=max(320, int(getattr(self.settings, "ocr_audit_max_long_edge", 1200))),
        )
        if any(
            line.confidence >= self.settings.ocr_confidence
            and extract_aadhaar_digits(line.text) is not None
            for line in audit_lines
        ):
            raise UnsafeDocument(
                "The post-mask safety check could still read an Aadhaar number. No output was created."
            )

