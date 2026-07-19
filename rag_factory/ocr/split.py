# -*- coding: utf-8 -*-
"""
Auto-Split — Split and categorize multi-page documents
=======================================================
Mirrors Mindee's "Split" feature: detects where one document ends and
the next begins within a multi-page PDF or image batch, then splits and
optionally classifies each segment.

Algorithm
---------
1. Feature extraction per page
   - Text density (chars / page area proxy)
   - Page visual fingerprint (perceptual hash via imagehash or pixel mean)
   - Layout type (from ML layout classifier)
   - Colon ratio, numeric ratio (form / table signals)
   - Is blank page? (whitespace ratio)

2. Boundary detection — three signals combined:
   a. Blank page separator  → hard boundary (confidence 0.95)
   b. Sharp visual discontinuity between consecutive pages
      (Δ perceptual hash > threshold)
   c. Document-type change signal
      (layout shifts from SINGLE_COLUMN → FORM, TABLE→SINGLE_COLUMN, etc.)

3. Segment building
   Each segment is a consecutive page range from one boundary to the next.

4. Optional classification
   Each segment is classified via the DocumentClassifier (heuristic → LLM).

Output
------
SplitResult:
  segments : List[DocumentSegment]  (one per detected document)
  boundaries: List[int]             (0-based page indices where splits occur)
  total_pages: int

DocumentSegment:
  segment_idx  : int
  page_start   : int    (0-based, inclusive)
  page_end     : int    (0-based, inclusive)
  page_count   : int
  doc_type     : str    (if classification requested, else "unknown")
  confidence   : float
  text         : str    (concatenated text of all pages in segment)
  layout_types : List[str]  (per-page layout types)

Usage
-----
  from rag_factory.ocr.split import AutoSplitter
  splitter = AutoSplitter()
  result   = splitter.split_pdf("multi_doc.pdf", classify=True)
  for seg in result.segments:
      print(seg.segment_idx, seg.page_start, seg.page_end, seg.doc_type)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─── result types ─────────────────────────────────────────────────────────────

@dataclass
class DocumentSegment:
    segment_idx:  int
    page_start:   int          # 0-based inclusive
    page_end:     int          # 0-based inclusive
    page_count:   int
    doc_type:     str = "unknown"
    confidence:   float = 0.0
    text:         str   = ""
    layout_types: List[str] = field(default_factory=list)
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "segment_idx":  self.segment_idx,
            "page_start":   self.page_start,
            "page_end":     self.page_end,
            "page_count":   self.page_count,
            "doc_type":     self.doc_type,
            "confidence":   round(self.confidence, 3),
            "layout_types": self.layout_types,
        }


@dataclass
class SplitResult:
    segments:     List[DocumentSegment]
    boundaries:   List[int]   # page indices where a new doc starts (0-based)
    total_pages:  int
    method:       str = "auto"

    def to_dict(self) -> dict:
        return {
            "total_pages":   self.total_pages,
            "num_segments":  len(self.segments),
            "boundaries":    self.boundaries,
            "method":        self.method,
            "segments":      [s.to_dict() for s in self.segments],
        }


# ─── page feature ─────────────────────────────────────────────────────────────

@dataclass
class _PageFeature:
    page_num:       int
    text:           str
    char_count:     int
    is_blank:       bool
    layout_type:    str
    colon_ratio:    float
    num_ratio:      float
    pixel_hash:     int = 0     # perceptual hash value (0 if unavailable)


# ─── main class ───────────────────────────────────────────────────────────────

class AutoSplitter:
    """
    Detects document boundaries in a multi-page PDF and splits into segments.

    Parameters
    ----------
    blank_threshold    : float  — text chars below this → blank page (default 20)
    visual_threshold   : float  — hash distance for visual discontinuity (default 15)
    layout_change_conf : float  — confidence required for layout-change boundary (default 0.75)
    classify           : bool   — auto-classify each segment (default True)
    min_segment_pages  : int    — minimum pages per segment, merges shorter ones (default 1)
    """

    def __init__(
        self,
        blank_threshold:    int   = 20,
        visual_threshold:   float = 15.0,
        layout_change_conf: float = 0.75,
        min_segment_pages:  int   = 1,
    ):
        self.blank_threshold    = blank_threshold
        self.visual_threshold   = visual_threshold
        self.layout_change_conf = layout_change_conf
        self.min_segment_pages  = min_segment_pages

    # ── public interface ──────────────────────────────────────────────────────

    def split_pdf(self, pdf_path: str, classify: bool = True,
                  dpi: int = 100) -> SplitResult:
        """
        Split a multi-page PDF into document segments.

        Parameters
        ----------
        pdf_path : path to the PDF file
        classify : run DocumentClassifier on each segment
        dpi      : render resolution for visual hash (lower = faster)
        """
        try:
            import fitz
        except ImportError:
            raise RuntimeError("PyMuPDF required — pip install pymupdf")

        doc   = fitz.open(pdf_path)
        n     = len(doc)
        feats = []

        for pno in range(n):
            page     = doc[pno]
            text     = page.get_text("text").strip()
            # Visual hash from low-res render
            phash    = 0
            try:
                from PIL import Image
                import io as _io
                mat  = fitz.Matrix(dpi / 72, dpi / 72)
                pix  = page.get_pixmap(matrix=mat)
                pil  = Image.open(_io.BytesIO(pix.tobytes("png"))).convert("L")
                phash = _pixel_hash(pil)
            except Exception:
                pass

            feats.append(self._extract_page_feature(pno, text, phash))

        doc.close()
        return self._build_split_result(feats, classify=classify)

    def split_pages(self, pages_text: List[str], classify: bool = True) -> SplitResult:
        """
        Split from a list of pre-extracted page text strings.
        Useful when you already have text from extractor.py.
        """
        feats = [self._extract_page_feature(i, t, 0) for i, t in enumerate(pages_text)]
        return self._build_split_result(feats, classify=classify)

    def split_extracted_doc(self, extracted_doc, classify: bool = True) -> SplitResult:
        """
        Split from an ExtractedDocument (output of extractor.extract_document).
        """
        texts = [p.text for p in extracted_doc.pages]
        return self.split_pages(texts, classify=classify)

    # ── internals ─────────────────────────────────────────────────────────────

    def _extract_page_feature(self, pno: int, text: str, phash: int) -> _PageFeature:
        import re as _re
        blank = len(text) < self.blank_threshold
        colon = sum(1 for w in text.split() if ":" in w) / max(len(text.split()), 1)
        num_re = _re.compile(r"^[\d,\.\$€£%\-\+\/]+$")
        num   = sum(1 for w in text.split() if num_re.match(w)) / max(len(text.split()), 1)

        # Layout classification using ML classifier (text-based, no image needed)
        layout_type = "SINGLE_COLUMN"
        try:
            from .ml_layout import classify_layout, words_from_tesseract
            # Heuristic-only path using text stats (no bounding boxes)
            from .ml_layout import LayoutResult, FORM, TABLE_HEAVY, MULTI_COLUMN, SINGLE_COLUMN
            if colon > 0.15:
                layout_type = FORM
            elif num > 0.20:
                layout_type = TABLE_HEAVY
            else:
                layout_type = SINGLE_COLUMN
        except Exception:
            pass

        return _PageFeature(
            page_num    = pno,
            text        = text,
            char_count  = len(text),
            is_blank    = blank,
            layout_type = layout_type,
            colon_ratio = colon,
            num_ratio   = num,
            pixel_hash  = phash,
        )

    def _build_split_result(self, feats: List[_PageFeature],
                             classify: bool) -> SplitResult:
        n = len(feats)
        if n == 0:
            return SplitResult(segments=[], boundaries=[], total_pages=0)

        # ── detect boundaries ─────────────────────────────────────────────────
        boundaries = [0]   # page 0 always starts the first segment

        for i in range(1, n):
            reason = self._is_boundary(feats, i)
            if reason:
                boundaries.append(i)

        # ── build segments ────────────────────────────────────────────────────
        segments: List[DocumentSegment] = []
        for bi, start in enumerate(boundaries):
            end  = (boundaries[bi + 1] - 1) if bi + 1 < len(boundaries) else n - 1
            text = "\n\n".join(f.text for f in feats[start:end + 1] if f.text)
            layouts = [f.layout_type for f in feats[start:end + 1]]

            seg = DocumentSegment(
                segment_idx  = bi,
                page_start   = start,
                page_end     = end,
                page_count   = end - start + 1,
                text         = text,
                layout_types = layouts,
            )

            # Optional classification
            if classify and text.strip():
                try:
                    from ..document_classifier import CLASSIFIER
                    cr = CLASSIFIER.classify(text[:3000])
                    seg.doc_type   = cr.doc_type
                    seg.confidence = cr.confidence
                    seg.metadata["extraction_mode"] = cr.extraction_mode
                    seg.metadata["heuristic"]       = cr.heuristic
                except Exception:
                    seg.doc_type   = "unknown"
                    seg.confidence = 0.0
            elif feats[start].is_blank:
                seg.doc_type = "blank"

            segments.append(seg)

        # ── merge tiny segments ───────────────────────────────────────────────
        if self.min_segment_pages > 1:
            segments = self._merge_short(segments, feats)

        return SplitResult(
            segments    = segments,
            boundaries  = boundaries,
            total_pages = n,
            method      = "heuristic+layout",
        )

    def _is_boundary(self, feats: List[_PageFeature], idx: int) -> Optional[str]:
        """Return a reason string if page `idx` starts a new document, else None."""
        prev = feats[idx - 1]
        curr = feats[idx]

        # Hard boundary: blank separator page
        if prev.is_blank or curr.is_blank:
            return "blank_page_separator"

        # Visual discontinuity
        if prev.pixel_hash and curr.pixel_hash:
            dist = bin(prev.pixel_hash ^ curr.pixel_hash).count("1")
            if dist > self.visual_threshold:
                return f"visual_discontinuity(d={dist})"

        # Layout type change between structurally distinct types
        _structurally_distinct = {
            ("SINGLE_COLUMN", "FORM"),
            ("FORM", "SINGLE_COLUMN"),
            ("SINGLE_COLUMN", "TABLE_HEAVY"),
            ("TABLE_HEAVY", "SINGLE_COLUMN"),
            ("FORM", "TABLE_HEAVY"),
            ("TABLE_HEAVY", "FORM"),
        }
        pair = (prev.layout_type, curr.layout_type)
        if pair in _structurally_distinct:
            # Only signal boundary if the density also drops/jumps significantly
            density_ratio = (curr.char_count + 1) / (prev.char_count + 1)
            if density_ratio < 0.25 or density_ratio > 4.0:
                return f"layout_change({prev.layout_type}→{curr.layout_type})"

        return None

    def _merge_short(self, segments: List[DocumentSegment],
                     feats: List[_PageFeature]) -> List[DocumentSegment]:
        """Merge segments shorter than min_segment_pages into neighbours."""
        merged: List[DocumentSegment] = []
        for seg in segments:
            if seg.page_count < self.min_segment_pages and merged:
                prev = merged[-1]
                prev.page_end    = seg.page_end
                prev.page_count  = prev.page_end - prev.page_start + 1
                prev.text       += "\n\n" + seg.text
                prev.layout_types.extend(seg.layout_types)
            else:
                merged.append(seg)
        # Re-index
        for i, seg in enumerate(merged):
            seg.segment_idx = i
        return merged


# ─── perceptual hash helper ───────────────────────────────────────────────────

def _pixel_hash(gray_pil, size: int = 16) -> int:
    """
    Compute a simple difference-hash of a grayscale PIL image.
    Returns an integer whose Hamming distance to another hash is a
    measure of visual similarity.
    """
    try:
        pil = gray_pil.resize((size + 1, size), resample=1)   # LANCZOS=1
        import struct
        pixels = list(pil.getdata())
        bits   = 0
        for row in range(size):
            for col in range(size):
                left  = pixels[row * (size + 1) + col]
                right = pixels[row * (size + 1) + col + 1]
                if left > right:
                    bits |= (1 << (row * size + col))
        return bits
    except Exception:
        return 0


# ─── convenience wrappers ─────────────────────────────────────────────────────

def split_pdf(pdf_path: str, classify: bool = True, **kwargs) -> SplitResult:
    """Top-level convenience: split a PDF into document segments."""
    return AutoSplitter(**kwargs).split_pdf(pdf_path, classify=classify)


def split_extracted(extracted_doc, classify: bool = True, **kwargs) -> SplitResult:
    """Top-level convenience: split an already-extracted document."""
    return AutoSplitter(**kwargs).split_extracted_doc(extracted_doc, classify=classify)
