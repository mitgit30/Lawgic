import logging
from io import BytesIO

import fitz
import pdfplumber
import pytesseract
from PIL import Image

from src.core.config import get_settings

logger = logging.getLogger(__name__)


def _extract_page_text(page: pdfplumber.page.Page) -> str:
    # 1) layout-aware extraction handles many multi-column legal PDFs.
    text = page.extract_text(layout=True) or ""
    if text.strip():
        return text

    # 2) fallback to default extraction.
    text = page.extract_text() or ""
    if text.strip():
        return text

    # 3) final fallback: reconstruct from words list.
    words = page.extract_words(use_text_flow=True) or []
    if words:
        return " ".join(w.get("text", "") for w in words if w.get("text", "").strip())
    return ""


def _pdf_has_embedded_text(file_bytes: bytes) -> bool:
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            if page.chars:
                return True
            text = page.extract_text() or ""
            if text.strip():
                return True
    return False


def _extract_text_with_pdfplumber(file_bytes: bytes) -> tuple[str, list[int]]:
    pages: list[str] = []
    empty_page_indexes: list[int] = []
    total_pages = 0
    empty_pages = 0

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages):
            total_pages += 1
            text = _extract_page_text(page)
            if text.strip():
                pages.append(text)
            else:
                empty_pages += 1
                empty_page_indexes.append(idx)

    full_text = "\n\n".join(pages)
    logger.info(
        "PDF text extraction done via pdfplumber | chars=%s | total_pages=%s | empty_pages=%s",
        len(full_text),
        total_pages,
        empty_pages,
    )
    return full_text, empty_page_indexes


def _ocr_page(doc: fitz.Document, page_index: int, *, ocr_dpi: int, ocr_lang: str) -> str:
    page = doc[page_index]
    zoom = max(ocr_dpi / 72.0, 1.0)
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pix.n >= 3 else "L"
    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(image, lang=ocr_lang) or ""


def _extract_text_with_ocr(file_bytes: bytes, page_indexes: list[int] | None = None) -> str:
    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    pages: list[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        indices = page_indexes if page_indexes is not None else list(range(len(doc)))
        for page_index in indices:
            text = _ocr_page(
                doc,
                page_index,
                ocr_dpi=settings.ocr_dpi,
                ocr_lang=settings.ocr_lang,
            )
            if text.strip():
                pages.append(text)

        full_text = "\n\n".join(pages)
        logger.info(
            "PDF text extraction done via OCR | chars=%s | ocr_pages=%s | total_pages=%s",
            len(full_text),
            len(indices),
            len(doc),
        )
    return full_text


def extract_text_from_pdf(file_bytes: bytes) -> str:
    has_text = _pdf_has_embedded_text(file_bytes)
    logger.info("PDF text detection result: has_embedded_text=%s", has_text)

    if has_text:
        plumber_text, empty_page_indexes = _extract_text_with_pdfplumber(file_bytes)
        if not empty_page_indexes:
            return plumber_text

        logger.info("Running OCR fallback for %s empty pages.", len(empty_page_indexes))
        try:
            ocr_text = _extract_text_with_ocr(file_bytes, page_indexes=empty_page_indexes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR fallback failed for empty pages: %s", exc)
            ocr_text = ""

        merged_text = "\n\n".join([part for part in [plumber_text, ocr_text] if part.strip()])
        logger.info("Hybrid extraction complete | chars=%s", len(merged_text))
        return merged_text

    logger.info("No embedded PDF text detected. Using OCR with pytesseract.")
    try:
        return _extract_text_with_ocr(file_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("OCR extraction failed")
        raise RuntimeError(
            "OCR failed. Ensure Tesseract is installed and TESSERACT_CMD is configured."
        ) from exc
