from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .domain import ProcessedDocument
from .settings import Settings


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
}


try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass


def decode_image(data: bytes, settings: Settings) -> list[np.ndarray]:
    try:
        with Image.open(io.BytesIO(data)) as source:
            pages: list[np.ndarray] = []
            frame_count = int(getattr(source, "n_frames", 1))
            if frame_count > settings.max_pages:
                raise HTTPException(413, f"Please upload a document with {settings.max_pages} pages or fewer.")
            for frame_index in range(frame_count):
                source.seek(frame_index)
                frame = ImageOps.exif_transpose(source.copy()).convert("RGB")
                if frame.width * frame.height > settings.max_page_pixels:
                    raise HTTPException(413, "An image page is too large to process safely.")
                pages.append(np.asarray(frame))
            return pages
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            415,
            "This image could not be decoded. Try JPEG, PNG, WebP, TIFF, BMP, GIF, or HEIC.",
        ) from exc


def render_pdf(data: bytes, settings: Settings) -> list[np.ndarray]:
    document = None
    try:
        document = pdfium.PdfDocument(data)
        if len(document) > settings.max_pages:
            raise HTTPException(413, f"Please upload a PDF with {settings.max_pages} pages or fewer.")
        pages: list[np.ndarray] = []
        for page_index in range(len(document)):
            page = document[page_index]
            page_width, page_height = page.get_size()
            estimated_pixels = int(page_width * settings.pdf_scale) * int(page_height * settings.pdf_scale)
            if estimated_pixels > settings.max_page_pixels:
                page.close()
                raise HTTPException(413, "A PDF page is too large to process safely.")
            bitmap = page.render(scale=settings.pdf_scale)
            pil_image = bitmap.to_pil().convert("RGB")
            pages.append(np.asarray(pil_image))
            bitmap.close()
            page.close()
        return pages
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(415, "This PDF could not be opened. It may be damaged or password-protected.") from exc
    finally:
        if document is not None:
            document.close()


def convert_word_to_pdf(data: bytes, extension: str) -> bytes:
    office = shutil.which("soffice") or shutil.which("soffice.exe") or shutil.which("libreoffice")
    if not office:
        raise HTTPException(
            503,
            "Word conversion needs LibreOffice on this computer. Install LibreOffice or save the document as PDF.",
        )
    with tempfile.TemporaryDirectory(prefix="aadhaar-mask-") as work_dir:
        root = Path(work_dir)
        source = root / f"source{extension}"
        output_dir = root / "converted"
        profile_dir = root / "office-profile"
        output_dir.mkdir()
        source.write_bytes(data)
        command = [
            office,
            "--headless",
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(source),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=90)
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(408, "Word document conversion took too long. Save it as PDF and try again.") from exc
        except subprocess.CalledProcessError as exc:
            raise HTTPException(415, "This Word document could not be converted. Save it as PDF and try again.") from exc
        converted = output_dir / "source.pdf"
        if not converted.is_file():
            raise HTTPException(415, "This Word document could not be converted. Save it as PDF and try again.")
        return converted.read_bytes()


def input_pages(data: bytes, filename: str, settings: Settings) -> list[np.ndarray]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, "Use a JPEG, PNG, WebP, TIFF, BMP, GIF, HEIC, PDF, or Word document.")
    if extension == ".pdf":
        return render_pdf(data, settings)
    if extension in {".doc", ".docx"}:
        return render_pdf(convert_word_to_pdf(data, extension), settings)
    return decode_image(data, settings)


def make_flattened_pdf(documents: tuple[ProcessedDocument, ...]) -> bytes:
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pageCompression=1)
    pdf.setTitle("Masked Aadhaar document")
    pdf.setAuthor("Private Aadhaar masking service")
    pdf.setSubject("Flattened redacted copy")
    for document in documents:
        image = document.image
        height, width = image.shape[:2]
        page_width = width * 0.72
        page_height = height * 0.72
        pdf.setPageSize((page_width, page_height))
        pdf.drawImage(
            ImageReader(Image.fromarray(image)),
            0,
            0,
            width=page_width,
            height=page_height,
        )
        pdf.showPage()
    pdf.save()
    return output.getvalue()

