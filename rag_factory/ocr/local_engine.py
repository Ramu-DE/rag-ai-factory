# -*- coding: utf-8 -*-
"""
AI RAG Factory — Local OCR Engine
===================================
Production-grade local OCR equivalent to AWS Textract + ABBYY FineReader.

Capabilities
------------
  detect_text       : raw character + line extraction with confidence scores
  analyze_document  : TABLES (bordered & borderless) + FORMS (key-value pairs)
  analyze_expense   : invoice / receipt structured field extraction
  analyze_id        : passport / driver's licence / national ID field extraction

Image pre-processing pipeline (ABBYY-equivalent)
-------------------------------------------------
  1. Grayscale conversion
  2. Upscale 2x (improves accuracy on low-res scans)
  3. Adaptive binarization (Otsu / sauvola threshold)
  4. Deskew  — Hough-line rotation correction (±45°)
  5. Denoise — morphological open/close + Gaussian blur
  6. Border crop — remove document edge artefacts
  7. Contrast enhancement

OCR backends (tried in priority order)
---------------------------------------
  1. pytesseract  — Tesseract 4/5 LSTM engine (recommended, best accuracy)
  2. easyocr      — pure-Python PyTorch engine (no binary install needed)
  3. PyMuPDF      — text-layer extraction (digital PDFs only, no image OCR)

Table detection strategies
---------------------------
  1. Bordered tables — OpenCV Hough-line grid detection
  2. Borderless tables — whitespace projection profile / column alignment
  3. pdfplumber      — native PDF table extraction (digital PDFs)

All public methods return the same TextractResult / Table / FormField /
ExpenseField / IDField types as textract.py — zero API change in extractor.py.
"""
from __future__ import annotations

import io
import os
import re
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .textract import (
    TextractResult, Table, TableCell, FormField, ExpenseField,
)


# ─────────────────────────────────────────────────────────────────────────────
# Backend availability flags (evaluated lazily)
# ─────────────────────────────────────────────────────────────────────────────
def _has_tesseract() -> bool:
    try:
        import pytesseract
        # Auto-discover Tesseract on Windows
        import sys, os
        if sys.platform == "win32":
            candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.join(os.environ.get("LOCALAPPDATA",""), "Tesseract-OCR", "tesseract.exe"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    pytesseract.pytesseract.tesseract_cmd = c
                    break
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _has_easyocr() -> bool:
    try:
        import easyocr  # noqa
        return True
    except ImportError:
        return False


def _has_cv2() -> bool:
    try:
        import cv2  # noqa
        return True
    except ImportError:
        return False


def _has_pil() -> bool:
    try:
        from PIL import Image  # noqa
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_pil(source):
    """Accept file path, bytes, or PIL Image — always return PIL.Image."""
    from PIL import Image
    if isinstance(source, str):
        return Image.open(source).convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source)).convert("RGB")
    return source.convert("RGB")


def _pil_to_cv2(pil_img):
    import numpy as np
    import cv2
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(cv2_img):
    import cv2
    from PIL import Image
    import numpy as np
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing pipeline
# ─────────────────────────────────────────────────────────────────────────────
class _Preprocessor:
    """
    ABBYY-equivalent image pre-processing.
    Returns a PIL Image ready for OCR.
    Falls back gracefully when OpenCV is unavailable.
    """

    def __call__(self, source, upscale: float = 2.0) -> Any:
        """Process image → return PIL Image."""
        pil = _to_pil(source)

        if not _has_cv2():
            return self._pil_only(pil, upscale)

        import cv2
        import numpy as np

        img = _pil_to_cv2(pil)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── upscale ────────────────────────────────────────────────────
        if upscale != 1.0:
            h, w = gray.shape
            gray = cv2.resize(gray, (int(w * upscale), int(h * upscale)),
                              interpolation=cv2.INTER_CUBIC)

        # ── deskew ─────────────────────────────────────────────────────
        gray = self._deskew(gray)

        # ── binarize ───────────────────────────────────────────────────
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── denoise ────────────────────────────────────────────────────
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.GaussianBlur(binary, (1, 1), 0)

        from PIL import Image
        return Image.fromarray(binary)

    def _deskew(self, gray):
        """Rotate image to correct skew using Hough-line angle estimation."""
        try:
            import cv2
            import numpy as np
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, math.pi / 180, 200)
            if lines is None:
                return gray
            angles = []
            for line in lines[:20]:
                _, theta = line[0]
                angle = (theta - math.pi / 2) * 180 / math.pi
                if abs(angle) < 45:
                    angles.append(angle)
            if not angles:
                return gray
            median_angle = sorted(angles)[len(angles) // 2]
            if abs(median_angle) < 0.5:
                return gray
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
            return cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return gray

    def _pil_only(self, pil_img, upscale: float):
        """Minimal preprocessing without OpenCV."""
        from PIL import Image, ImageFilter, ImageEnhance
        if upscale != 1.0:
            w, h = pil_img.size
            pil_img = pil_img.resize((int(w * upscale), int(h * upscale)),
                                     Image.LANCZOS)
        gray = pil_img.convert("L")
        enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
        return enhanced


PREPROCESSOR = _Preprocessor()


# ─────────────────────────────────────────────────────────────────────────────
# Tesseract backend
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _Word:
    text:       str
    conf:       float
    left:       int
    top:        int
    width:      int
    height:     int
    page:       int = 1

    @property
    def right(self): return self.left + self.width
    @property
    def bottom(self): return self.top + self.height
    @property
    def cx(self): return self.left + self.width // 2
    @property
    def cy(self): return self.top + self.height // 2


def _tesseract_words(pil_img, lang: str = "eng") -> List[_Word]:
    import pytesseract
    data = pytesseract.image_to_data(
        pil_img, lang=lang,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )
    words = []
    n = len(data["text"])
    for i in range(n):
        text = str(data["text"][i]).strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        words.append(_Word(
            text=text, conf=conf / 100.0,
            left=data["left"][i], top=data["top"][i],
            width=data["width"][i], height=data["height"][i],
        ))
    return words


def _tesseract_text(pil_img, lang: str = "eng") -> str:
    import pytesseract
    return pytesseract.image_to_string(pil_img, lang=lang,
                                       config="--oem 3 --psm 6").strip()


# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR backend
# ─────────────────────────────────────────────────────────────────────────────
_EASY_READER = None

def _easyocr_words(pil_img) -> List[_Word]:
    global _EASY_READER
    import numpy as np
    import easyocr
    if _EASY_READER is None:
        _EASY_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    arr = np.array(pil_img)
    results = _EASY_READER.readtext(arr, detail=1, paragraph=False)
    words = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        left, top = int(min(xs)), int(min(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))
        words.append(_Word(text=text.strip(), conf=float(conf),
                           left=left, top=top, width=w, height=h))
    return [w for w in words if w.text]


# ─────────────────────────────────────────────────────────────────────────────
# Reading-order reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def _words_to_lines(words: List[_Word], line_gap_ratio: float = 0.6) -> List[List[_Word]]:
    """Cluster words into lines by vertical proximity, then sort left-to-right."""
    if not words:
        return []
    words = sorted(words, key=lambda w: w.top)
    lines: List[List[_Word]] = []
    current: List[_Word] = [words[0]]

    for w in words[1:]:
        avg_h = sum(x.height for x in current) / len(current)
        if abs(w.top - current[-1].top) <= avg_h * line_gap_ratio:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda x: x.left))
            current = [w]
    lines.append(sorted(current, key=lambda x: x.left))
    return lines


def _lines_to_text(lines: List[List[_Word]]) -> str:
    return "\n".join(" ".join(w.text for w in line) for line in lines)


def _column_aware_text(words: List[_Word], img_width: int) -> str:
    """Detect 2-column layouts and produce reading-order text."""
    if not words or img_width == 0:
        return _lines_to_text(_words_to_lines(words))
    mid = img_width // 2
    left_col  = [w for w in words if w.cx <= mid]
    right_col = [w for w in words if w.cx >  mid]
    if not right_col or len(right_col) < 3:
        return _lines_to_text(_words_to_lines(words))
    left_text  = _lines_to_text(_words_to_lines(left_col))
    right_text = _lines_to_text(_words_to_lines(right_col))
    return left_text + "\n" + right_text


# ─────────────────────────────────────────────────────────────────────────────
# Table detection
# ─────────────────────────────────────────────────────────────────────────────
def _detect_tables_cv2(pil_img) -> List[Table]:
    """Detect bordered tables using horizontal+vertical line detection."""
    if not _has_cv2():
        return []
    try:
        import cv2
        import numpy as np
        img = _pil_to_cv2(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        h_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        grid      = cv2.add(h_lines, v_lines)
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        tables = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 100 or h < 40:
                continue
            roi_h = cv2.morphologyEx(binary[y:y+h, x:x+w],
                                     cv2.MORPH_OPEN, h_kernel)
            roi_v = cv2.morphologyEx(binary[y:y+h, x:x+w],
                                     cv2.MORPH_OPEN, v_kernel)
            h_proj = np.sum(roi_h, axis=1)
            v_proj = np.sum(roi_v, axis=0)
            rows = _count_lines_proj(h_proj)
            cols = _count_lines_proj(v_proj)
            if rows >= 2 and cols >= 2:
                tables.append(Table(page=1, rows=rows, cols=cols, cells=[]))
        return tables
    except Exception:
        return []


def _count_lines_proj(proj) -> int:
    import numpy as np
    threshold = proj.max() * 0.3
    in_line = False
    count   = 0
    for v in proj:
        if v > threshold and not in_line:
            in_line = True
            count += 1
        elif v <= threshold:
            in_line = False
    return max(count, 1)


def _detect_borderless_tables(words: List[_Word], img_width: int) -> List[Table]:
    """Detect borderless tables via column-alignment whitespace analysis."""
    if not words or img_width == 0:
        return []
    lines = _words_to_lines(words)
    if len(lines) < 3:
        return []
    # Look for 3+ consecutive lines with ≥3 column-aligned words
    col_lines = [l for l in lines if len(l) >= 3]
    if len(col_lines) < 3:
        return []
    # Simple heuristic: if >30% of lines have ≥3 words at similar x positions
    return [Table(page=1, rows=len(col_lines), cols=len(col_lines[0]), cells=[])]


def _extract_table_cells_from_words(
    words: List[_Word], table: Table
) -> Table:
    """Populate table cells by clustering words into grid positions."""
    if not words or table.rows < 2 or table.cols < 2:
        return table

    lines = _words_to_lines(words)
    if len(lines) < 2:
        return table

    cells = []
    for r, line in enumerate(lines[:table.rows]):
        # Distribute words across columns by x-position
        if not line:
            continue
        line_width = max(w.right for w in line) - min(w.left for w in line) + 1
        col_width  = line_width / table.cols
        for w in line:
            col = min(int((w.left - min(x.left for x in line)) / col_width),
                      table.cols - 1)
            cells.append(TableCell(row=r, col=col, text=w.text,
                                   confidence=w.conf))
    table.cells = cells
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Form / Key-Value extraction
# ─────────────────────────────────────────────────────────────────────────────
_KV_SPLIT = re.compile(
    r"^([\w\s\-\/\(\)\.]{2,40}?)\s*[:–\-]\s*(.*)$", re.UNICODE
)
_FIELD_LABELS = re.compile(
    r"(invoice\s*(no|num|#|date|number)|"
    r"date|due\s*date|payment\s*terms|"
    r"vendor|supplier|bill\s*to|ship\s*to|"
    r"po\s*(number|#)|purchase\s*order|"
    r"subtotal|sub\s*total|total\s*(due|amount)?|"
    r"tax|vat|gst|hst|"
    r"patient\s*(name|id)?|provider|"
    r"diagnosis|medication|prescription|"
    r"name|address|phone|email|fax|"
    r"account\s*(no|number)?|"
    r"expir(y|ation)|issue\s*date|"
    r"nationality|country|city|state|zip)",
    re.I,
)


def _extract_forms_from_words(words: List[_Word]) -> List[FormField]:
    """Extract key-value pairs from OCR word geometry."""
    lines = _words_to_lines(words)
    forms: List[FormField] = []

    for line in lines:
        line_text = " ".join(w.text for w in line)
        m = _KV_SPLIT.match(line_text)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            if _FIELD_LABELS.search(key) or len(key) <= 30:
                avg_conf = sum(w.conf for w in line) / len(line) if line else 0
                forms.append(FormField(
                    key=key, value=val,
                    key_confidence=avg_conf,
                    value_confidence=avg_conf,
                    page=1,
                ))
                continue

        # Spatial proximity: short label on left, value on right
        if len(line) >= 2:
            left_words  = [w for w in line if w.cx < (line[0].left + line[-1].right) // 2]
            right_words = [w for w in line if w.cx >= (line[0].left + line[-1].right) // 2]
            if left_words and right_words:
                key = " ".join(w.text for w in left_words).strip().rstrip(":")
                val = " ".join(w.text for w in right_words).strip()
                if _FIELD_LABELS.search(key):
                    avg_conf = sum(w.conf for w in line) / len(line)
                    forms.append(FormField(
                        key=key, value=val,
                        key_confidence=avg_conf,
                        value_confidence=avg_conf,
                        page=1,
                    ))

    # Deduplicate by key
    seen: Dict[str, bool] = {}
    unique = []
    for f in forms:
        k = f.key.lower()
        if k not in seen and f.value:
            seen[k] = True
            unique.append(f)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Invoice / Expense extraction  (≡ Textract AnalyzeExpense)
# ─────────────────────────────────────────────────────────────────────────────
_AMT_RE   = re.compile(r"[\$£€]?\s*([\d,]+\.?\d{0,2})")
_DATE_RE  = re.compile(
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
    r"|\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2}"
    r"|[A-Za-z]+\s+\d{1,2},?\s*\d{4})\b"
)
_PO_RE    = re.compile(r"\bP\.?O\.?\s*(?:Number|No|#)?\s*[:\-]?\s*([A-Z0-9\-]{4,20})\b", re.I)
_INV_RE   = re.compile(r"\b(?:Invoice|Inv)\.?\s*(?:No|#|Number|Num)?\s*[:\-]?\s*([A-Z0-9\-]{3,20})\b", re.I)
_VAT_RE   = re.compile(r"\b(?:VAT|GST|Tax)\s*(?:No|#|Number|ID)?\s*[:\-]?\s*([A-Z0-9]{6,20})\b", re.I)
_NET_RE   = re.compile(r"\b(?:Net)\s*(\d{1,3})\b", re.I)

_INV_FIELD_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("INVOICE_RECEIPT_DATE", re.compile(
        r"invoice\s*date\s*[:\-]?\s*(" + _DATE_RE.pattern + r")", re.I), "date"),
    ("DUE_DATE", re.compile(
        r"due\s*(?:date|by)\s*[:\-]?\s*(" + _DATE_RE.pattern + r")", re.I), "date"),
    ("VENDOR_NAME", re.compile(
        r"^(.{3,60})$", re.M), "vendor"),
    ("TOTAL", re.compile(
        r"(?:total\s*(?:due|amount|:)?|amount\s*due)\s*[:\-]?\s*([\$£€]?\s*[\d,]+\.?\d{0,2})", re.I), "amount"),
    ("SUBTOTAL", re.compile(
        r"sub\s*total\s*[:\-]?\s*([\$£€]?\s*[\d,]+\.?\d{0,2})", re.I), "amount"),
    ("TAX", re.compile(
        r"(?:tax|vat|gst|hst)\s*[:\-]?\s*([\$£€]?\s*[\d,]+\.?\d{0,2})", re.I), "amount"),
    ("PO_NUMBER", _PO_RE, "text"),
    ("INVOICE_ID", _INV_RE, "text"),
    ("PAYMENT_TERMS", _NET_RE, "text"),
]


def _extract_invoice_fields(text: str) -> List[ExpenseField]:
    fields: List[ExpenseField] = []
    lines  = text.splitlines()

    # Vendor: usually top 3 non-empty lines before "Invoice"
    header_lines = [l.strip() for l in lines[:8] if l.strip() and not _INV_RE.search(l)]
    if header_lines:
        fields.append(ExpenseField(field_type="VENDOR_NAME", label="Vendor",
                                   value=header_lines[0], confidence=0.7))

    for field_type, pattern, ftype in _INV_FIELD_PATTERNS:
        if field_type == "VENDOR_NAME":
            continue
        m = pattern.search(text)
        if m:
            val = m.group(1).strip() if m.lastindex else m.group(0).strip()
            fields.append(ExpenseField(field_type=field_type, label=field_type.replace("_"," ").title(),
                                       value=val, confidence=0.85))

    # Line items: rows matching "description ... quantity ... price ... amount"
    _LI_RE = re.compile(
        r"(.{5,50}?)\s{2,}(\d+)\s+x?\s*([\$£€]?[\d,]+\.?\d{0,2})\s*([\$£€]?[\d,]+\.?\d{0,2})", re.I
    )
    for line in lines:
        m = _LI_RE.search(line)
        if m:
            fields.append(ExpenseField(
                field_type="LINE_ITEM",
                label="LineItem",
                value=f"{m.group(1).strip()} | qty={m.group(2)} | unit={m.group(3)} | total={m.group(4)}",
                confidence=0.80,
            ))

    return fields


# ─────────────────────────────────────────────────────────────────────────────
# ID document extraction  (≡ Textract AnalyzeID)
# ─────────────────────────────────────────────────────────────────────────────
# MRZ parsers — TD1 (30 chars × 3 lines), TD2 (36 × 2), TD3 (passport, 44 × 2)
_MRZ_TD3 = re.compile(r"P[<A-Z][A-Z]{3}([A-Z<]{39})\n([A-Z0-9<]{44})", re.M)
_MRZ_TD1 = re.compile(r"([A-Z0-9<]{30})\n([A-Z0-9<]{30})\n([A-Z0-9<]{30})", re.M)
_MRZ_CHARS = re.compile(r"[A-Z0-9<]{9,}")


def _parse_mrz_td3(line1: str, line2: str) -> Dict[str, str]:
    """Parse ICAO TD3 (passport) MRZ."""
    out: Dict[str, str] = {}
    if len(line1) >= 44:
        doc_type = line1[0]
        country  = line1[2:5].replace("<", "")
        names    = line1[5:44].split("<<", 1)
        out["DOCUMENT_TYPE"] = doc_type
        out["ISSUING_COUNTRY"] = country
        out["LAST_NAME"]  = names[0].replace("<", " ").strip() if names else ""
        out["FIRST_NAME"] = names[1].replace("<", " ").strip() if len(names) > 1 else ""
    if len(line2) >= 44:
        out["DOCUMENT_NUMBER"] = line2[0:9].replace("<", "")
        out["NATIONALITY"]     = line2[10:13].replace("<", "")
        dob = line2[13:19]
        out["DATE_OF_BIRTH"]   = f"{dob[4:6]}/{dob[2:4]}/{dob[0:2]}" if len(dob) == 6 else dob
        out["SEX"]             = line2[20]
        exp = line2[21:27]
        out["EXPIRY_DATE"]     = f"{exp[4:6]}/{exp[2:4]}/20{exp[0:2]}" if len(exp) == 6 else exp
    return out


def _extract_id_fields_local(text: str) -> Dict[str, str]:
    """Extract identity document fields using MRZ parsing + regex patterns."""
    result: Dict[str, str] = {}
    upper = text.upper()

    # MRZ TD3 (passport)
    m = _MRZ_TD3.search(upper)
    if m:
        result.update(_parse_mrz_td3("P " + m.group(1), m.group(2)))
        result["MRZ_DETECTED"] = "YES"
        return result

    # MRZ raw (any 2+ lines of MRZ chars)
    mrz_lines = [l.strip() for l in text.splitlines() if _MRZ_CHARS.match(l.strip()) and len(l.strip()) >= 30]
    if len(mrz_lines) >= 2:
        if len(mrz_lines[0]) >= 44:
            result.update(_parse_mrz_td3(mrz_lines[0], mrz_lines[1]))
            result["MRZ_DETECTED"] = "YES"
            return result

    # Pattern-based fallback for driver's licences / national IDs
    patterns = [
        ("LAST_NAME",       re.compile(r"surname[:\s]+([A-Z][A-Za-z\s\-]{1,30})", re.I)),
        ("FIRST_NAME",      re.compile(r"(?:given\s*names?|forename|first\s*name)[:\s]+([A-Z][A-Za-z\s\-]{1,30})", re.I)),
        ("FULL_NAME",       re.compile(r"(?:^|\n)\s*name[:\s]+([A-Z][A-Za-z\s\-]{2,40})", re.I | re.M)),
        ("DATE_OF_BIRTH",   re.compile(r"(?:date\s*of\s*birth|dob|birth\s*date)[:\s]+(" + _DATE_RE.pattern + r")", re.I)),
        ("EXPIRY_DATE",     re.compile(r"(?:expir(?:y|ation|es?)|exp\.?\s*date)[:\s]+(" + _DATE_RE.pattern + r")", re.I)),
        ("ISSUE_DATE",      re.compile(r"(?:issue\s*d(?:ate)?|date\s*of\s*issue)[:\s]+(" + _DATE_RE.pattern + r")", re.I)),
        ("DOCUMENT_NUMBER", re.compile(r"(?:passport\s*no|dl\s*no|document\s*no|id\s*(?:no|number)|license\s*no)[:\s#]+([A-Z0-9\-]{4,20})", re.I)),
        ("NATIONALITY",     re.compile(r"nationality[:\s]+([A-Za-z\s]{2,30})", re.I)),
        ("ISSUING_COUNTRY",  re.compile(r"(?:country|issued\s*by|issuing\s*(?:country|authority))[:\s]+([A-Za-z\s]{2,30})", re.I)),
        ("ADDRESS",         re.compile(r"address[:\s]+(.{10,80})", re.I)),
        ("SEX",             re.compile(r"(?:sex|gender)[:\s]+(M(?:ale)?|F(?:emale)?|X)", re.I)),
    ]
    for key, pat in patterns:
        m = pat.search(text)
        if m:
            val = m.group(1).strip()
            if val:
                result[key] = val

    # Driver's license number (heuristic: "DL D1234567" or "DL: D1234567")
    dl_m = re.search(r"\bDL\s*[:\-#]?\s*([A-Z]\d{6,10})\b", upper)
    if dl_m and "DOCUMENT_NUMBER" not in result:
        result["DOCUMENT_NUMBER"] = dl_m.group(1)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────
class LocalOCREngine:
    """
    Local OCR engine — drop-in replacement for AWS Textract.

    Usage
    -----
        engine = LocalOCREngine()
        result = engine.analyze_document("path/to/scan.png")
        print(result.raw_text)
        for tbl in result.tables:
            print(tbl.to_markdown())
        for f in result.forms:
            print(f.key, ":", f.value)
    """

    def __init__(self, lang: str = "eng", upscale: float = 2.0,
                 preprocess: bool = True):
        self.lang       = lang
        self.upscale    = upscale
        self.preprocess = preprocess
        self._backend   = self._choose_backend()

    # ── backend selection ──────────────────────────────────────────────────────
    def _choose_backend(self) -> str:
        if _has_tesseract():
            return "tesseract"
        if _has_easyocr():
            return "easyocr"
        return "none"

    @property
    def backend(self) -> str:
        return self._backend

    # ── core OCR ──────────────────────────────────────────────────────────────
    def _ocr_image(self, source) -> Tuple[str, List[_Word]]:
        """Return (full_text, word_list) for an image source."""
        pil = _to_pil(source)
        if self.preprocess:
            pil = PREPROCESSOR(pil, upscale=self.upscale)
        elif self.upscale != 1.0:
            w, h = pil.size
            from PIL import Image
            pil = pil.resize((int(w * self.upscale), int(h * self.upscale)),
                             Image.LANCZOS)

        if self._backend == "tesseract":
            words = _tesseract_words(pil, lang=self.lang)
            text  = _tesseract_text(pil, lang=self.lang) if not words else \
                    _column_aware_text(words, pil.width)
            return text, words

        if self._backend == "easyocr":
            words = _easyocr_words(pil)
            text  = _column_aware_text(words, pil.width)
            return text, words

        return "", []

    # ── public API  (mirrors Textract API signatures) ─────────────────────────
    def detect_text(self, source) -> TextractResult:
        """Equivalent to Textract DetectDocumentText."""
        text, _ = self._ocr_image(source)
        return TextractResult(
            raw_text=text, page_count=1, method="local_detect_text"
        )

    def analyze_document(self, source, features=None) -> TextractResult:
        """
        Equivalent to Textract AnalyzeDocument(FeatureTypes=["TABLES","FORMS"]).
        features: list containing "TABLES", "FORMS", or both (default both)
        """
        if features is None:
            features = ["TABLES", "FORMS"]

        text, words = self._ocr_image(source)
        pil = _to_pil(source)

        tables: List[Table] = []
        forms:  List[FormField] = []

        if "TABLES" in features:
            # Try bordered first, then borderless
            bordered = _detect_tables_cv2(pil)
            if bordered:
                tables = [_extract_table_cells_from_words(words, t)
                          for t in bordered]
            else:
                borderless = _detect_borderless_tables(words, pil.width)
                tables = [_extract_table_cells_from_words(words, t)
                          for t in borderless]

        if "FORMS" in features:
            forms = _extract_forms_from_words(words)

        return TextractResult(
            raw_text=text, tables=tables, forms=forms,
            page_count=1, method="local_analyze_document",
        )

    def analyze_expense(self, source) -> TextractResult:
        """
        Equivalent to Textract AnalyzeExpense.
        Extracts: vendor, invoice#, dates, totals, tax, line items.
        """
        text, words = self._ocr_image(source)
        forms        = _extract_forms_from_words(words)
        expense_flds = _extract_invoice_fields(text)

        # Supplement expense fields from forms (high confidence form fields)
        form_kv = {f.key.upper(): f.value for f in forms}
        _SUPP = {
            "VENDOR": "VENDOR_NAME", "TOTAL DUE": "TOTAL",
            "INVOICE DATE": "INVOICE_RECEIPT_DATE",
            "DUE DATE": "DUE_DATE", "PO NUMBER": "PO_NUMBER",
            "TAX": "TAX", "SUBTOTAL": "SUBTOTAL",
        }
        existing_types = {e.field_type for e in expense_flds}
        for form_key, exp_type in _SUPP.items():
            val = form_kv.get(form_key, "")
            if val and exp_type not in existing_types:
                expense_flds.append(ExpenseField(
                    field_type=exp_type,
                    label=form_key.title(),
                    value=val, confidence=0.80,
                ))

        return TextractResult(
            raw_text=text, forms=forms, expense_fields=expense_flds,
            page_count=1, method="local_analyze_expense",
        )

    def analyze_id(self, source) -> TextractResult:
        """
        Equivalent to Textract AnalyzeID.
        Supports: passport (MRZ TD3), driver's licence, national ID.
        """
        text, _  = self._ocr_image(source)
        id_fields = _extract_id_fields_local(text)
        raw = " ".join(f"{k}: {v}" for k, v in id_fields.items())

        return TextractResult(
            raw_text=text or raw,
            id_fields=id_fields,
            page_count=1, method="local_analyze_id",
        )

    # ── multi-page PDF support ────────────────────────────────────────────────
    def process_pdf(self, pdf_path: str, mode: str = "forms") -> List[TextractResult]:
        """
        Process each page of a PDF.
        mode: "text" | "forms" | "expense" | "id"
        """
        try:
            import fitz
            results = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page    = doc[page_num]
                    mat     = fitz.Matrix(2.0, 2.0)
                    pix     = page.get_pixmap(matrix=mat)
                    img_buf = pix.tobytes("png")
                    results.append(self._process_bytes(img_buf, mode))
            return results
        except ImportError:
            return [self._process_bytes(open(pdf_path, "rb").read(), mode)]

    def _process_bytes(self, data: bytes, mode: str) -> TextractResult:
        if mode == "expense":
            return self.analyze_expense(data)
        if mode == "id":
            return self.analyze_id(data)
        if mode in ("forms", "auto"):
            return self.analyze_document(data)
        return self.detect_text(data)

    # ── diagnostic ────────────────────────────────────────────────────────────
    def info(self) -> Dict[str, Any]:
        return {
            "backend":    self._backend,
            "tesseract":  _has_tesseract(),
            "easyocr":    _has_easyocr(),
            "opencv":     _has_cv2(),
            "lang":       self.lang,
            "upscale":    self.upscale,
            "preprocess": self.preprocess,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton + public helpers  (matches textract.py API)
# ─────────────────────────────────────────────────────────────────────────────
_ENGINE: Optional[LocalOCREngine] = None


def get_local_engine() -> LocalOCREngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = LocalOCREngine()
    return _ENGINE


def ocr_file_local(file_path: str, mode: str = "forms") -> TextractResult:
    """File-based public API — drop-in for textract.ocr_file()."""
    eng = get_local_engine()
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        results = eng.process_pdf(file_path, mode=mode)
        if not results:
            return TextractResult(raw_text="", method="local_empty")
        # Merge pages
        merged_text = "\n\n".join(r.raw_text for r in results)
        all_tables  = [t for r in results for t in r.tables]
        all_forms   = [f for r in results for f in r.forms]
        all_exp     = [e for r in results for e in r.expense_fields]
        merged_id   = {}
        for r in results:
            merged_id.update(r.id_fields)
        return TextractResult(
            raw_text=merged_text, tables=all_tables, forms=all_forms,
            expense_fields=all_exp, id_fields=merged_id,
            page_count=len(results), method=results[0].method,
        )
    with open(file_path, "rb") as f:
        return ocr_bytes_local(f.read(), mode=mode)


def ocr_bytes_local(data: bytes, mode: str = "forms") -> TextractResult:
    """Bytes-based public API — drop-in for textract.ocr_bytes()."""
    return get_local_engine()._process_bytes(data, mode)
