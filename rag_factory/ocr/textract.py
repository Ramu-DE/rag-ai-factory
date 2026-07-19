# -*- coding: utf-8 -*-
"""
AWS Textract adapter
====================
Handles three Textract APIs:
  detect_document_text   — raw OCR (scans, photos)
  analyze_document       — tables + forms + key-value pairs
  analyze_expense        — invoice / receipt fields
  analyze_id             — identity documents

All calls are synchronous (single-page) for notebooks / API.
Multi-page PDFs are split to individual pages by the caller.

Returns structured TextractResult with:
  - pages          : List[PageResult]  (text blocks, tables, forms, per page)
  - raw_text       : str               (all text concatenated)
  - tables         : List[Table]
  - forms          : List[FormField]   (key/value pairs)
  - expense_fields : List[ExpenseField]
  - id_fields      : Dict[str, str]
"""
from __future__ import annotations
import os, json, base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─── result types ─────────────────────────────────────────────────────────────
@dataclass
class TextBlock:
    text:       str
    block_type: str          # LINE | WORD | CELL | KEY_VALUE_SET
    confidence: float
    page:       int
    bbox:       Dict[str, float] = field(default_factory=dict)


@dataclass
class TableCell:
    row:        int
    col:        int
    text:       str
    confidence: float
    is_header:  bool = False


@dataclass
class Table:
    page:       int
    rows:       int
    cols:       int
    cells:      List[TableCell] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render table as markdown for embedding."""
        if not self.cells:
            return ""
        grid: Dict[Tuple[int,int], str] = {(c.row, c.col): c.text for c in self.cells}
        lines = []
        for r in range(self.rows):
            row_vals = [grid.get((r, c), "") for c in range(self.cols)]
            lines.append("| " + " | ".join(row_vals) + " |")
            if r == 0:
                lines.append("|" + "|".join(["---"] * self.cols) + "|")
        return "\n".join(lines)

    def to_text(self) -> str:
        """Flat text version for chunking."""
        return " | ".join(
            f"({c.row},{c.col}): {c.text}" for c in self.cells
        )


@dataclass
class FormField:
    key:              str
    value:            str
    key_confidence:   float
    value_confidence: float
    page:             int


@dataclass
class ExpenseField:
    field_type:  str
    label:       str
    value:       str
    confidence:  float
    page:        int = 1


@dataclass
class TextractResult:
    raw_text:       str
    tables:         List[Table]       = field(default_factory=list)
    forms:          List[FormField]   = field(default_factory=list)
    expense_fields: List[ExpenseField]= field(default_factory=list)
    id_fields:      Dict[str, str]    = field(default_factory=dict)
    page_count:     int               = 0
    method:         str               = "textract"
    metadata:       Dict[str, Any]    = field(default_factory=dict)

    def structured_text(self) -> str:
        """
        Combines raw text + table markdown + form fields
        into a single rich string for chunking + embedding.
        """
        parts = [self.raw_text]
        if self.tables:
            parts.append("\n\n--- TABLES ---")
            for i, tbl in enumerate(self.tables):
                parts.append(f"\nTable {i+1} (page {tbl.page}):\n{tbl.to_markdown()}")
        if self.forms:
            parts.append("\n\n--- FORM FIELDS ---")
            for f in self.forms:
                parts.append(f"{f.key}: {f.value}")
        if self.expense_fields:
            parts.append("\n\n--- DOCUMENT FIELDS ---")
            for e in self.expense_fields:
                parts.append(f"{e.label}: {e.value}")
        if self.id_fields:
            parts.append("\n\n--- ID FIELDS ---")
            for k, v in self.id_fields.items():
                parts.append(f"{k}: {v}")
        return "\n".join(parts)


# ─── Textract client ──────────────────────────────────────────────────────────
def _get_textract_client():
    import boto3
    return boto3.client(
        "textract",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ─── block parsers ────────────────────────────────────────────────────────────
def _extract_text_from_blocks(blocks: List[Dict]) -> str:
    lines = [
        b["Text"] for b in blocks
        if b.get("BlockType") == "LINE" and "Text" in b
    ]
    return "\n".join(lines)


def _extract_tables_from_blocks(blocks: List[Dict]) -> List[Table]:
    block_map = {b["Id"]: b for b in blocks}
    tables: List[Table] = []

    for block in blocks:
        if block.get("BlockType") != "TABLE":
            continue

        cells: List[TableCell] = []
        max_row = max_col = 0

        for rel in block.get("Relationships", []):
            if rel["Type"] != "CHILD":
                continue
            for cell_id in rel["Ids"]:
                cell_block = block_map.get(cell_id, {})
                if cell_block.get("BlockType") != "CELL":
                    continue
                row = cell_block.get("RowIndex", 1) - 1
                col = cell_block.get("ColumnIndex", 1) - 1
                max_row = max(max_row, row)
                max_col = max(max_col, col)
                is_header = cell_block.get("EntityTypes", []) == ["COLUMN_HEADER"]

                # get cell text from WORD children
                words = []
                for c_rel in cell_block.get("Relationships", []):
                    if c_rel["Type"] == "CHILD":
                        for wid in c_rel["Ids"]:
                            wb = block_map.get(wid, {})
                            if wb.get("BlockType") in ("WORD", "SELECTION_ELEMENT"):
                                words.append(wb.get("Text", ""))
                conf = cell_block.get("Confidence", 0.0)
                cells.append(TableCell(row=row, col=col, text=" ".join(words),
                                       confidence=conf, is_header=is_header))

        if cells:
            page = block.get("Page", 1)
            tables.append(Table(page=page, rows=max_row+1, cols=max_col+1, cells=cells))

    return tables


def _extract_forms_from_blocks(blocks: List[Dict]) -> List[FormField]:
    block_map  = {b["Id"]: b for b in blocks}
    forms: List[FormField] = []

    for block in blocks:
        if block.get("BlockType") != "KEY_VALUE_SET":
            continue
        if "KEY" not in block.get("EntityTypes", []):
            continue

        # get key text
        key_words = []
        val_block_id = None
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for cid in rel["Ids"]:
                    cb = block_map.get(cid, {})
                    if cb.get("BlockType") == "WORD":
                        key_words.append(cb.get("Text",""))
            elif rel["Type"] == "VALUE":
                val_block_id = rel["Ids"][0] if rel["Ids"] else None

        key_text  = " ".join(key_words)
        key_conf  = block.get("Confidence", 0.0)
        val_text  = ""
        val_conf  = 0.0

        if val_block_id:
            vb = block_map.get(val_block_id, {})
            val_words = []
            for rel in vb.get("Relationships", []):
                if rel["Type"] == "CHILD":
                    for cid in rel["Ids"]:
                        wb = block_map.get(cid, {})
                        if wb.get("BlockType") == "WORD":
                            val_words.append(wb.get("Text",""))
            val_text = " ".join(val_words)
            val_conf = vb.get("Confidence", 0.0)

        if key_text:
            page = block.get("Page", 1)
            forms.append(FormField(key=key_text, value=val_text,
                                   key_confidence=key_conf,
                                   value_confidence=val_conf, page=page))
    return forms


def _extract_expense_fields(response: Dict) -> List[ExpenseField]:
    fields: List[ExpenseField] = []
    for doc in response.get("ExpenseDocuments", []):
        for sf in doc.get("SummaryFields", []):
            label = sf.get("LabelDetection", {}).get("Text", sf.get("Type", {}).get("Text",""))
            value = sf.get("ValueDetection", {}).get("Text","")
            conf  = sf.get("ValueDetection", {}).get("Confidence", 0.0)
            ftype = sf.get("Type",{}).get("Text","")
            fields.append(ExpenseField(field_type=ftype, label=label, value=value, confidence=conf))

        for item in doc.get("LineItemGroups", []):
            for li in item.get("LineItems", []):
                parts = []
                for lif in li.get("LineItemExpenseFields", []):
                    lbl = lif.get("Type",{}).get("Text","")
                    val = lif.get("ValueDetection",{}).get("Text","")
                    if lbl and val:
                        parts.append(f"{lbl}: {val}")
                if parts:
                    fields.append(ExpenseField(
                        field_type="LINE_ITEM", label="LineItem",
                        value=" | ".join(parts), confidence=90.0
                    ))
    return fields


def _extract_id_fields(response: Dict) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for doc in response.get("IdentityDocuments", []):
        for f in doc.get("IdentityDocumentFields", []):
            ftype = f.get("Type",{}).get("Text","")
            value = f.get("ValueDetection",{}).get("Text","")
            if ftype and value:
                result[ftype] = value
    return result


# ─── public API ───────────────────────────────────────────────────────────────
def ocr_file(file_path: str, mode: str = "auto") -> TextractResult:
    """
    Extract text + structure from a file (PDF, PNG, JPG, TIFF).

    mode:
      "text"    — DetectDocumentText only (fastest, cheapest)
      "forms"   — AnalyzeDocument FORMS + TABLES
      "expense" — AnalyzeExpense (invoices, receipts)
      "id"      — AnalyzeID (passports, driver's licences)
      "auto"    — "text" mode (caller switches based on doc_type)
    """
    doc_bytes = _read_bytes(file_path)
    return ocr_bytes(doc_bytes, mode=mode)


def ocr_bytes(doc_bytes: bytes, mode: str = "text") -> TextractResult:
    """Run Textract on raw bytes."""
    client   = _get_textract_client()
    document = {"Bytes": doc_bytes}

    if mode == "expense":
        resp   = client.analyze_expense(Document=document)
        text   = ""
        tables = []
        forms  = []
        exp    = _extract_expense_fields(resp)
        return TextractResult(raw_text=text, tables=tables, forms=forms,
                               expense_fields=exp, page_count=1, method="textract_expense")

    if mode == "id":
        resp     = client.analyze_id(DocumentPages=[document])
        id_fields = _extract_id_fields(resp)
        return TextractResult(raw_text=" ".join(id_fields.values()),
                               id_fields=id_fields, page_count=1, method="textract_id")

    if mode in ("forms", "auto"):
        resp   = client.analyze_document(
            Document=document,
            FeatureTypes=["TABLES", "FORMS"],
        )
        blocks = resp.get("Blocks", [])
        text   = _extract_text_from_blocks(blocks)
        tables = _extract_tables_from_blocks(blocks)
        forms  = _extract_forms_from_blocks(blocks)
        pages  = max((b.get("Page",1) for b in blocks), default=1)
        return TextractResult(raw_text=text, tables=tables, forms=forms,
                               page_count=pages, method="textract_forms")

    # default: detect text only
    resp   = client.detect_document_text(Document=document)
    blocks = resp.get("Blocks", [])
    text   = _extract_text_from_blocks(blocks)
    pages  = max((b.get("Page",1) for b in blocks), default=1)
    return TextractResult(raw_text=text, page_count=pages, method="textract_text")


def ocr_pdf_pages(pdf_path: str, mode: str = "forms") -> List[TextractResult]:
    """
    Split PDF into pages and OCR each one.
    Uses PyMuPDF to render pages as PNG → Textract.
    Falls back to single-call if PyMuPDF unavailable.
    """
    try:
        import fitz
        results = []
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat  = fitz.Matrix(2.0, 2.0)   # 2x zoom for better OCR accuracy
                pix  = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                result = ocr_bytes(img_bytes, mode=mode)
                results.append(result)
        return results
    except ImportError:
        # No PyMuPDF — send whole PDF (only works for single-page)
        return [ocr_file(pdf_path, mode=mode)]
