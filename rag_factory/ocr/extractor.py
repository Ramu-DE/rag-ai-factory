# -*- coding: utf-8 -*-
"""
Unified document extractor
===========================
Chooses the right extraction strategy automatically:

  Clean digital PDF  →  PyMuPDF (free, fast, structure-aware blocks)
  Scanned PDF        →  PyMuPDF render → Textract (OCR per page)
  Image (PNG/JPG)    →  Textract directly
  Forced mode        →  caller passes extraction_mode="textract"|"pymupdf"

Strategy decision for PDFs:
  1. Try PyMuPDF extraction
  2. Count extractable chars per page
  3. If avg < 100 chars/page → classify as scanned → route to Textract
  4. Otherwise keep PyMuPDF result

Returned ExtractedDocument:
  pages        : List[ExtractedPage]   (one per physical page)
  doc_type     : inferred before skill routing
  is_scanned   : bool
  method       : "pymupdf" | "textract_text" | "textract_forms" | ...
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── output types ─────────────────────────────────────────────────────────────
@dataclass
class ExtractedPage:
    page_num:    int
    text:        str          # clean text for chunking
    rich_text:   str          # text + tables + forms (for embedding)
    char_count:  int
    tables:      List[Any] = field(default_factory=list)   # Table objects
    forms:       List[Any] = field(default_factory=list)   # FormField objects
    metadata:    Dict[str, Any] = field(default_factory=dict)
    ner:         Optional[Any] = None  # NERResult — populated by NER normalizer


@dataclass
class ExtractedDocument:
    file_name:      str
    pages:          List[ExtractedPage]
    is_scanned:     bool
    method:         str
    expense_fields: List[Any] = field(default_factory=list)
    id_fields:      Dict[str, str] = field(default_factory=dict)
    page_count:     int = 0

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def full_rich_text(self) -> str:
        return "\n\n".join(p.rich_text for p in self.pages)

    @property
    def all_tables(self):
        tables = []
        for p in self.pages:
            tables.extend(p.tables)
        return tables

    @property
    def all_forms(self):
        forms = []
        for p in self.pages:
            forms.extend(p.forms)
        return forms


# ─── PyMuPDF extractor ────────────────────────────────────────────────────────
def _extract_pymupdf(pdf_path: str) -> List[ExtractedPage]:
    import fitz
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            # get_text("blocks") returns (x0,y0,x1,y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
            # Sort top-to-bottom, left-to-right
            blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))
            paragraphs = [b[4].strip() for b in blocks if b[4].strip()]
            text = "\n".join(paragraphs)

            pages.append(ExtractedPage(
                page_num=i,
                text=text,
                rich_text=text,
                char_count=len(text),
                metadata={"page_label": page.get_label() or str(i + 1)},
            ))
    return pages


def _is_scanned(pages: List[ExtractedPage], threshold: int = 100) -> bool:
    if not pages:
        return True
    avg = sum(p.char_count for p in pages) / len(pages)
    return avg < threshold


# ─── OCR backend selector ─────────────────────────────────────────────────────
def _ocr_backend() -> str:
    """Return 'textract' | 'local' depending on AWS availability."""
    return os.getenv("OCR_BACKEND", "auto")  # auto = try textract, fall back to local


def _run_ocr(source_path: str, mode: str = "forms", is_bytes: bool = False):
    """
    Try Textract first; fall back to local engine automatically.
    Returns a TextractResult-compatible object.
    """
    backend = _ocr_backend()

    # Force local
    if backend == "local":
        from .local_engine import ocr_file_local, ocr_bytes_local
        if is_bytes:
            with open(source_path, "rb") as f:
                return ocr_bytes_local(f.read(), mode=mode)
        return ocr_file_local(source_path, mode=mode)

    # Try Textract (default / auto)
    try:
        from .textract import ocr_file
        return ocr_file(source_path, mode=mode)
    except Exception as e:
        err = str(e)
        if "AccessDeniedException" in err or "credentials" in err.lower() \
                or "not authorized" in err.lower() or "UnrecognizedClientException" in err:
            # Silently fall back to local engine
            from .local_engine import ocr_file_local
            return ocr_file_local(source_path, mode=mode)
        raise


def _run_ocr_bytes(data: bytes, mode: str = "forms"):
    """Run OCR on raw bytes — Textract first, local fallback."""
    backend = _ocr_backend()
    if backend == "local":
        from .local_engine import ocr_bytes_local
        return ocr_bytes_local(data, mode=mode)
    try:
        from .textract import ocr_bytes
        return ocr_bytes(data, mode=mode)
    except Exception as e:
        err = str(e)
        if "AccessDeniedException" in err or "credentials" in err.lower() \
                or "not authorized" in err.lower() or "UnrecognizedClientException" in err:
            from .local_engine import ocr_bytes_local
            return ocr_bytes_local(data, mode=mode)
        raise


# ─── Textract / local extractor ───────────────────────────────────────────────
def _extract_textract_pages(pdf_path: str, mode: str = "forms") -> List[ExtractedPage]:
    """Extract pages using Textract if available, local engine as fallback."""
    backend = _ocr_backend()
    try:
        if backend == "local":
            raise RuntimeError("local forced")
        from .textract import ocr_pdf_pages
        results = ocr_pdf_pages(pdf_path, mode=mode)
    except Exception as e:
        err = str(e)
        if backend == "local" or "AccessDeniedException" in err \
                or "not authorized" in err.lower() or "credentials" in err.lower():
            from .local_engine import get_local_engine
            eng     = get_local_engine()
            results = eng.process_pdf(pdf_path, mode=mode)
        else:
            raise

    pages = []
    for i, r in enumerate(results):
        rich = r.structured_text()
        pages.append(ExtractedPage(
            page_num=i,
            text=r.raw_text,
            rich_text=rich,
            char_count=len(r.raw_text),
            tables=r.tables,
            forms=r.forms,
        ))
    return pages


def _extract_image_ocr(img_path: str, mode: str = "forms") -> ExtractedDocument:
    """Extract image using Textract if available, local engine as fallback."""
    r = _run_ocr(img_path, mode=mode)
    page = ExtractedPage(
        page_num=0,
        text=r.raw_text,
        rich_text=r.structured_text(),
        char_count=len(r.raw_text),
        tables=r.tables,
        forms=r.forms,
    )
    return ExtractedDocument(
        file_name=os.path.basename(img_path),
        pages=[page],
        is_scanned=True,
        method=r.method,
        expense_fields=r.expense_fields,
        id_fields=r.id_fields,
        page_count=1,
    )


# Keep old name as alias for compatibility
_extract_image_textract = _extract_image_ocr


# ─── NER enrichment ──────────────────────────────────────────────────────────
def _enrich_ner(doc: "ExtractedDocument") -> "ExtractedDocument":
    """Run post-OCR NER normalization on every page (in-place)."""
    try:
        from .ner_normalizer import get_normalizer
        nn = get_normalizer()
        for page in doc.pages:
            if page.text and not page.ner:
                page.ner = nn.run(page.text)
    except Exception:
        pass   # NER enrichment is optional — never block extraction
    return doc


# ─── main entrypoint ─────────────────────────────────────────────────────────
def extract_document(
    file_path:       str,
    extraction_mode: str = "auto",   # "auto" | "pymupdf" | "textract" | "expense" | "id"
    textract_mode:   str = "forms",  # "text" | "forms" | "expense" | "id"
    scan_threshold:  int = 100,      # avg chars/page below this → treat as scanned
) -> ExtractedDocument:
    """
    Extract all content from a PDF or image file.

    extraction_mode="auto" (default):
      - PDFs: try PyMuPDF first; fall back to Textract if scanned
      - Images: always Textract
    extraction_mode="textract":
      - Force Textract for all files (use for low-quality scans)
    extraction_mode="expense":
      - Force Textract AnalyzeExpense (invoices, receipts)
    extraction_mode="id":
      - Force Textract AnalyzeID
    """
    ext      = os.path.splitext(file_path)[1].lower()
    fname    = os.path.basename(file_path)
    is_image = ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif")

    # ── image files ───────────────────────────────────────────────────────────
    if is_image:
        mode = extraction_mode if extraction_mode in ("expense","id","forms","text") else textract_mode
        return _extract_image_ocr(file_path, mode=mode)

    # ── forced modes ─────────────────────────────────────────────────────────
    if extraction_mode == "expense":
        r = _run_ocr(file_path, mode="expense")
        page = ExtractedPage(
            page_num=0, text=r.raw_text, rich_text=r.structured_text(),
            char_count=len(r.raw_text), tables=r.tables, forms=r.forms,
        )
        return _enrich_ner(ExtractedDocument(
            file_name=fname, pages=[page], is_scanned=False,
            method=r.method, expense_fields=r.expense_fields, page_count=1,
        ))

    if extraction_mode == "id":
        r = _run_ocr(file_path, mode="id")
        page = ExtractedPage(
            page_num=0, text=r.raw_text, rich_text=r.structured_text(),
            char_count=len(r.raw_text), tables=[], forms=[],
        )
        return _enrich_ner(ExtractedDocument(
            file_name=fname, pages=[page], is_scanned=False,
            method=r.method, id_fields=r.id_fields, page_count=1,
        ))

    if extraction_mode == "textract":
        pages = _extract_textract_pages(file_path, mode=textract_mode)
        method = pages[0].metadata.get("method", f"ocr_{textract_mode}") if pages else f"ocr_{textract_mode}"
        return _enrich_ner(ExtractedDocument(
            file_name=fname, pages=pages, is_scanned=True,
            method=method, page_count=len(pages),
        ))

    # ── auto mode (PDF) ───────────────────────────────────────────────────────
    try:
        import fitz
        pymupdf_pages = _extract_pymupdf(file_path)
        if not _is_scanned(pymupdf_pages, threshold=scan_threshold):
            return _enrich_ner(ExtractedDocument(
                file_name=fname, pages=pymupdf_pages, is_scanned=False,
                method="pymupdf", page_count=len(pymupdf_pages),
            ))
        # Scanned → fall through to OCR
    except ImportError:
        pass

    # OCR fallback (Textract → local engine auto-fallback)
    pages = _extract_textract_pages(file_path, mode=textract_mode)
    method = pages[0].metadata.get("method", f"ocr_{textract_mode}") if pages else f"ocr_{textract_mode}"
    return _enrich_ner(ExtractedDocument(
        file_name=fname, pages=pages, is_scanned=True,
        method=method, page_count=len(pages),
    ))
