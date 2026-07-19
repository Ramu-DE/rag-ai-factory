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


# ─── Textract extractor ───────────────────────────────────────────────────────
def _extract_textract_pages(pdf_path: str, mode: str = "forms") -> List[ExtractedPage]:
    from .textract import ocr_pdf_pages
    results = ocr_pdf_pages(pdf_path, mode=mode)
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


def _extract_image_textract(img_path: str, mode: str = "forms") -> ExtractedDocument:
    from .textract import ocr_file
    r = ocr_file(img_path, mode=mode)
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
        return _extract_image_textract(file_path, mode=mode)

    # ── forced modes ─────────────────────────────────────────────────────────
    if extraction_mode == "expense":
        from .textract import ocr_file
        r = ocr_file(file_path, mode="expense")
        page = ExtractedPage(
            page_num=0, text="", rich_text=r.structured_text(),
            char_count=0, tables=[], forms=[],
        )
        return ExtractedDocument(
            file_name=fname, pages=[page], is_scanned=False,
            method="textract_expense", expense_fields=r.expense_fields, page_count=1,
        )

    if extraction_mode == "id":
        from .textract import ocr_file
        r = ocr_file(file_path, mode="id")
        page = ExtractedPage(
            page_num=0, text=r.raw_text, rich_text=r.structured_text(),
            char_count=len(r.raw_text), tables=[], forms=[],
        )
        return ExtractedDocument(
            file_name=fname, pages=[page], is_scanned=False,
            method="textract_id", id_fields=r.id_fields, page_count=1,
        )

    if extraction_mode == "textract":
        pages = _extract_textract_pages(file_path, mode=textract_mode)
        return ExtractedDocument(
            file_name=fname, pages=pages, is_scanned=True,
            method=f"textract_{textract_mode}", page_count=len(pages),
        )

    # ── auto mode (PDF) ───────────────────────────────────────────────────────
    try:
        import fitz
        pymupdf_pages = _extract_pymupdf(file_path)
        if not _is_scanned(pymupdf_pages, threshold=scan_threshold):
            return ExtractedDocument(
                file_name=fname, pages=pymupdf_pages, is_scanned=False,
                method="pymupdf", page_count=len(pymupdf_pages),
            )
        # Scanned → fall through to Textract
    except ImportError:
        pass

    # Textract fallback for scanned / no PyMuPDF
    pages = _extract_textract_pages(file_path, mode=textract_mode)
    return ExtractedDocument(
        file_name=fname, pages=pages, is_scanned=True,
        method=f"textract_{textract_mode}", page_count=len(pages),
    )
