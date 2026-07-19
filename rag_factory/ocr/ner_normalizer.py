# -*- coding: utf-8 -*-
"""
Post-OCR NER + Field Normalizer
================================
Runs after raw OCR text extraction to:
  1. Detect named entity spans (dates, amounts, phone numbers, emails,
     addresses, IDs, names, organizations)
  2. Normalize detected values to canonical forms
  3. Return annotated text + structured entity dict for downstream use

Why this matters (Mindee / ABBYY analogy)
------------------------------------------
Leading OCR APIs differentiate themselves not just by raw character accuracy,
but by semantic understanding of extracted values. A date "Jan 15 2024",
"15/01/24", "2024-01-15" are all the same date — normalizing them to ISO 8601
makes downstream matching, validation, and embedding much more reliable.

NER types and normalizations
------------------------------
  DATE         → ISO 8601 (YYYY-MM-DD) or best-effort YYYY-MM
  AMOUNT       → canonical "$1,234.56" with currency symbol
  PHONE        → E.164-like "+1-555-123-4567" (or local canonical)
  EMAIL        → lowercase
  ADDRESS      → single-line, comma-separated
  ID_NUMBER    → uppercase, hyphens preserved
  PERSON_NAME  → Title Case
  ORG_NAME     → as-detected (no normalization, just extraction)
  ICD_CODE     → uppercase, dot-separated (e.g. "J06.9")
  CPT_CODE     → 5-digit zero-padded

This module is pure Python (no ML model, no network calls).
Uses regex + dateutil for parsing.

Usage
-----
  from rag_factory.ocr.ner_normalizer import NERNormalizer
  nn = NERNormalizer()
  result = nn.run("Invoice Date: Jan 15, 2024  Total Due: $1,800.00")
  result.entities      # List[Entity]
  result.normalized    # {"DATE": ["2024-01-15"], "AMOUNT": ["$1,800.00"], ...}
  result.annotated     # "Invoice Date: [DATE:2024-01-15]  Total Due: [AMOUNT:$1,800.00]"
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Entity dataclass ─────────────────────────────────────────────────────────
@dataclass
class Entity:
    entity_type: str    # DATE | AMOUNT | PHONE | EMAIL | ADDRESS | ...
    raw:         str    # original text span
    normalized:  str    # canonical form
    start:       int    # char offset in source text
    end:         int
    confidence:  float  # 0.0–1.0

    def __repr__(self):
        return f"Entity({self.entity_type}, raw={self.raw!r}, norm={self.normalized!r})"


@dataclass
class NERResult:
    text:       str
    entities:   List[Entity]       = field(default_factory=list)
    normalized: Dict[str, List[str]] = field(default_factory=dict)
    annotated:  str = ""

    def get(self, entity_type: str) -> List[str]:
        """Return all normalized values for a given entity type."""
        return self.normalized.get(entity_type, [])

    def first(self, entity_type: str, default: str = "") -> str:
        vals = self.get(entity_type)
        return vals[0] if vals else default


# ─── Pattern library ──────────────────────────────────────────────────────────

# Dates — covers the main formats seen in invoices, medical records, reports
_DATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # ISO 8601: 2024-01-15
    (re.compile(r"\b(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b"), "iso"),
    # US: 01/15/2024 or 1-15-24
    (re.compile(r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b"), "us"),
    # Long form: January 15, 2024 / Jan 15 2024 / 15 January 2024
    (re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[.\s]+(\d{1,2})[,\s]+(\d{4})\b", re.I), "long_mdy"),
    (re.compile(
        r"\b(\d{1,2})[.\s]+"
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[.\s]+(\d{4})\b", re.I), "long_dmy"),
    # Year-only: 2024 (lower confidence)
    (re.compile(r"\b(20[0-9]{2}|19[5-9][0-9])\b"), "year"),
]

_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    "january": "01", "february": "02", "march": "03", "april": "04",
    "june": "06", "july": "07", "august": "08", "september": "09",
    "october": "10", "november": "11", "december": "12",
}

# Currency amounts
_AMOUNT_PAT = re.compile(
    r"(?:(?:USD|EUR|GBP|CAD|AUD|INR|JPY)\s*)?"
    r"[\$€£₹¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?|\d+(?:\.\d{1,4})?)"
    r"\s*(?:USD|EUR|GBP|CAD|AUD|INR|JPY)?",
    re.I,
)
_CURRENCY_LEADER = re.compile(r"[\$€£₹¥]|(?:USD|EUR|GBP|CAD|AUD|INR)")

# Phone numbers (US-centric + international)
_PHONE_PAT = re.compile(
    r"(?:\+?1[\s\-\.]?)?"
    r"\(?(\d{3})\)?[\s\-\.]?"
    r"(\d{3})[\s\-\.]?"
    r"(\d{4})"
    r"(?:\s*(?:ext|x|ext\.)\s*\d{1,5})?",
    re.I,
)

# Email
_EMAIL_PAT = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# ICD-10 codes
_ICD_PAT = re.compile(r"\b([A-TV-Z][0-9]{2})\.?([0-9A-TV-Z]{0,4})\b")

# CPT codes (5-digit standalone)
_CPT_PAT = re.compile(r"\b(\d{5})\b")

# NPI / Tax ID / SSN-like patterns
_NPI_PAT  = re.compile(r"\bNPI\s*[:\-#]?\s*(\d{10})\b", re.I)
_EIN_PAT  = re.compile(r"\b(\d{2})\s*[\-]?\s*(\d{7})\b")

# Address block (very light — just enough to detect multi-token addresses)
_ADDR_PAT = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9\s]{3,40}"
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|"
    r"Court|Ct|Circle|Cir|Way|Place|Pl|Suite|Ste|Floor|Fl|Apt|Unit)"
    r"[\.,\s]{0,3}[A-Za-z\s]{2,30}(?:,\s*[A-Z]{2})?\s*\d{5}(?:\-\d{4})?",
    re.I,
)

# Person name heuristic (Title Case 2-4 word sequence not matching known labels)
_NAME_PAT = re.compile(
    r"\b(?:Dr\.?\s+|Mr\.?\s+|Mrs\.?\s+|Ms\.?\s+|Prof\.?\s+)?"
    r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})\b"
)
_NOT_NAME = re.compile(
    r"\b(Invoice|Total|Amount|Date|Payment|Terms|Bill|Ship|Address|"
    r"Street|Avenue|Phone|Email|Fax|Company|Corporation|Inc|LLC|Ltd|"
    r"Department|Section|Chapter|Table|Figure|Appendix|January|February|"
    r"March|April|June|July|August|September|October|November|December)\b",
    re.I,
)

# Organization heuristic
_ORG_SUFFIXES = re.compile(
    r"\b([A-Z][A-Za-z0-9\s&\.\-]{2,50})"
    r"\s+(?:Inc|LLC|Ltd|Corp|Co\b|PLC|LLP|GmbH|SA|SAS|NV|BV|AG|Pty|"
    r"Incorporated|Limited|Corporation|Company)\b",
    re.I,
)


# ─── Normalizers ──────────────────────────────────────────────────────────────

def _normalize_date(m: re.Match, fmt: str) -> Tuple[str, float]:
    """Try to parse a date match into ISO 8601. Returns (iso_string, confidence)."""
    try:
        g = m.groups()
        if fmt == "iso":
            y, mo, d = int(g[0]), int(g[1]), int(g[2])
        elif fmt == "us":
            mo, d, y = int(g[0]), int(g[1]), int(g[2])
            if y < 100:
                y += 2000 if y < 50 else 1900
        elif fmt == "long_mdy":
            mo = int(_MONTH_MAP.get(g[0].lower()[:3], "0"))
            d, y = int(g[1]), int(g[2])
        elif fmt == "long_dmy":
            d = int(g[0])
            mo = int(_MONTH_MAP.get(g[1].lower()[:3], "0"))
            y = int(g[2])
        elif fmt == "year":
            return g[0], 0.55   # year-only is low confidence

        if not (1 <= mo <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100):
            return "", 0.0
        return f"{y:04d}-{mo:02d}-{d:02d}", 0.92
    except (ValueError, IndexError):
        return "", 0.0


def _normalize_amount(raw: str) -> str:
    """Normalize a currency amount string."""
    # Extract currency symbol
    sym_m = _CURRENCY_LEADER.search(raw)
    sym   = sym_m.group(0) if sym_m else "$"
    if sym.upper() in ("USD",):
        sym = "$"
    # Extract numeric part
    num = re.sub(r"[^\d\.,]", "", raw)
    # Standardise thousands separators → remove, then reformat
    try:
        # Preserve decimal places from original if present
        has_decimal = "." in num
        val = float(num.replace(",", ""))
        if has_decimal:
            # Keep original decimal precision (at least 2 places)
            dec_digits = len(num.split(".")[-1]) if "." in num else 0
            dec_digits = max(dec_digits, 2)
            formatted = f"{val:,.{dec_digits}f}"
        else:
            formatted = f"{int(val):,}"
        return f"{sym}{formatted}"
    except ValueError:
        return raw.strip()


def _normalize_phone(m: re.Match) -> str:
    parts = m.groups()[:3]
    return f"({parts[0]}) {parts[1]}-{parts[2]}"


# ─── Main NER class ───────────────────────────────────────────────────────────

class NERNormalizer:
    """
    Post-OCR named entity recognizer and value normalizer.

    Runs a cascade of regex patterns over extracted text, normalizes
    detected values, and returns annotated output.
    """

    def run(self, text: str) -> NERResult:
        if not text or not text.strip():
            return NERResult(text=text)

        # Normalize Unicode (OCR sometimes produces odd quotes/dashes)
        text = unicodedata.normalize("NFKC", text)

        entities: List[Entity] = []
        covered:  set          = set()   # char offsets already claimed

        def _add(etype, m, norm, conf):
            s, e = m.start(), m.end()
            if any(i in covered for i in range(s, e)):
                return   # overlap — skip
            if not norm:
                return
            covered.update(range(s, e))
            entities.append(Entity(etype, text[s:e], norm, s, e, conf))

        # ── Addresses (before phone — addresses contain digit sequences) ───────
        for m in _ADDR_PAT.finditer(text):
            addr = " ".join(m.group(0).split())
            _add("ADDRESS", m, addr, 0.75)

        # ── Dates ─────────────────────────────────────────────────────────────
        for pat, fmt in _DATE_PATTERNS:
            for m in pat.finditer(text):
                iso, conf = _normalize_date(m, fmt)
                if iso:
                    _add("DATE", m, iso, conf)

        # ── Currency amounts ───────────────────────────────────────────────────
        for m in _AMOUNT_PAT.finditer(text):
            raw = m.group(0).strip()
            # Must have a digit and look like a monetary value
            if not re.search(r"\d", raw):
                continue
            # Skip if it's already been claimed (e.g. part of date)
            if any(i in covered for i in range(m.start(), m.end())):
                continue
            # Require context: currency symbol OR nearby monetary keyword
            has_sym = bool(_CURRENCY_LEADER.search(raw))
            # Check surrounding context (±40 chars) for amount keywords
            ctx = text[max(0, m.start()-40):min(len(text), m.end()+10)].lower()
            has_ctx = bool(re.search(
                r"(total|amount|due|paid|price|cost|fee|charge|balance|invoice|subtotal|tax|vat|gst)",
                ctx,
            ))
            if has_sym or has_ctx:
                norm = _normalize_amount(raw)
                _add("AMOUNT", m, norm, 0.88 if has_sym else 0.72)

        # ── Phone numbers ──────────────────────────────────────────────────────
        for m in _PHONE_PAT.finditer(text):
            norm = _normalize_phone(m)
            _add("PHONE", m, norm, 0.82)

        # ── Emails ────────────────────────────────────────────────────────────
        for m in _EMAIL_PAT.finditer(text):
            _add("EMAIL", m, m.group(0).lower(), 0.98)

        # ── ICD-10 codes ───────────────────────────────────────────────────────
        for m in _ICD_PAT.finditer(text):
            code = m.group(1) + ("." + m.group(2) if m.group(2) else "")
            _add("ICD_CODE", m, code.upper(), 0.90)

        # ── NPI ────────────────────────────────────────────────────────────────
        for m in _NPI_PAT.finditer(text):
            _add("ID_NUMBER", m, f"NPI:{m.group(1)}", 0.95)

        # ── Organizations ──────────────────────────────────────────────────────
        for m in _ORG_SUFFIXES.finditer(text):
            org = m.group(0).strip()
            _add("ORG_NAME", m, org, 0.70)

        # ── Person names (lower priority — many false positives) ──────────────
        for m in _NAME_PAT.finditer(text):
            name = m.group(1)
            if _NOT_NAME.search(name):
                continue
            if len(name.split()) < 2:
                continue
            # Skip if already covered
            if any(i in covered for i in range(m.start(), m.end())):
                continue
            _add("PERSON_NAME", m, name.title(), 0.60)

        # ── Sort entities by position ─────────────────────────────────────────
        entities.sort(key=lambda e: e.start)

        # ── Build normalized dict ─────────────────────────────────────────────
        normalized: Dict[str, List[str]] = {}
        for e in entities:
            normalized.setdefault(e.entity_type, []).append(e.normalized)

        # ── Build annotated text ───────────────────────────────────────────────
        annotated = _build_annotated(text, entities)

        return NERResult(
            text=text,
            entities=entities,
            normalized=normalized,
            annotated=annotated,
        )


def _build_annotated(text: str, entities: List[Entity]) -> str:
    """Replace entity spans with [TYPE:normalized_value] tags."""
    parts = []
    cursor = 0
    for e in entities:
        parts.append(text[cursor:e.start])
        parts.append(f"[{e.entity_type}:{e.normalized}]")
        cursor = e.end
    parts.append(text[cursor:])
    return "".join(parts)


# ─── Convenience wrapper ──────────────────────────────────────────────────────

_NORMALIZER: Optional[NERNormalizer] = None

def get_normalizer() -> NERNormalizer:
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = NERNormalizer()
    return _NORMALIZER


def normalize_ocr_text(text: str) -> NERResult:
    """
    Top-level convenience: normalize OCR output and return NERResult.
    Singleton normalizer (no model warm-up cost).
    """
    return get_normalizer().run(text)
