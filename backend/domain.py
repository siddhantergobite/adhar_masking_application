from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class VisionDetection:
    class_name: str
    confidence: float
    polygon: np.ndarray
    box: np.ndarray
    source: str = "yolo"


@dataclass(frozen=True)
class NumberDetection:
    confidence: float
    box: np.ndarray


@dataclass(frozen=True)
class OcrWord:
    text: str
    box: np.ndarray


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    polygon: np.ndarray
    words: tuple[OcrWord, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Verification:
    accepted: bool
    confidence: float
    signals: tuple[str, ...]


@dataclass(frozen=True)
class ProcessedDocument:
    image: np.ndarray
    document_class: str
    document_confidence: float
    verification_confidence: float
    masked_regions: int


@dataclass(frozen=True)
class ProcessingResult:
    documents: tuple[ProcessedDocument, ...]

    @property
    def masked_regions(self) -> int:
        return sum(document.masked_regions for document in self.documents)

