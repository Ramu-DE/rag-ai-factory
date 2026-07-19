# -*- coding: utf-8 -*-
"""
ML Layout Classifier
====================
Classifies a document page into one of five structural layout types using
machine-learning features derived from word bounding-box statistics.
No GPU, no PyTorch — runs entirely on numpy + scipy + scikit-learn.

Layout types
------------
  SINGLE_COLUMN  : standard article / letter / report
  MULTI_COLUMN   : newspaper, academic paper, brochure
  FORM           : key-value pairs, labelled fields, checkboxes
  TABLE_HEAVY    : spreadsheet-style or bordered-table dominant
  MIXED          : combination (e.g. form with a summary table)

How it works
------------
  1. Feature extraction  — ~25 spatial/statistical features from word boxes
  2. Heuristic fast-path — rule-based decision for high-confidence cases
     (avoids model cold-start cost for clearly obvious layouts)
  3. ML classifier       — RandomForest trained on synthetic feature vectors
     representing each layout archetype (no external training data needed;
     prototypical samples generated from known distributions)

Integration
-----------
  layout = classify_layout(words, page_width, page_height)
  layout.layout_type   # "SINGLE_COLUMN" | "MULTI_COLUMN" | "FORM" | "TABLE_HEAVY" | "MIXED"
  layout.confidence    # 0.0 – 1.0
  layout.features      # dict of raw feature values (for debugging / logging)
  layout.reasoning     # one-line human-readable explanation
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── layout type constants ────────────────────────────────────────────────────
SINGLE_COLUMN = "SINGLE_COLUMN"
MULTI_COLUMN  = "MULTI_COLUMN"
FORM          = "FORM"
TABLE_HEAVY   = "TABLE_HEAVY"
MIXED         = "MIXED"

_ALL_TYPES = [SINGLE_COLUMN, MULTI_COLUMN, FORM, TABLE_HEAVY, MIXED]


# ─── result ───────────────────────────────────────────────────────────────────
@dataclass
class LayoutResult:
    layout_type: str
    confidence:  float
    features:    Dict[str, float] = field(default_factory=dict)
    reasoning:   str = ""

    def is_form(self):       return self.layout_type in (FORM, MIXED)
    def is_tabular(self):    return self.layout_type in (TABLE_HEAVY, MIXED)
    def is_multi_column(self): return self.layout_type == MULTI_COLUMN
    def ocr_strategy(self) -> str:
        """Suggested OCR strategy for this layout."""
        return {
            SINGLE_COLUMN: "sequential",
            MULTI_COLUMN:  "column_split",
            FORM:          "form_kv",
            TABLE_HEAVY:   "table_grid",
            MIXED:         "combined",
        }.get(self.layout_type, "sequential")


# ─── word box type (mirrors _Word in local_engine.py) ─────────────────────────
@dataclass
class WordBox:
    text:  str
    left:  float   # normalised 0–1
    top:   float
    right: float
    bottom: float
    confidence: float = 1.0

    @property
    def width(self):  return self.right - self.left
    @property
    def height(self): return self.bottom - self.top
    @property
    def cx(self):     return (self.left + self.right) / 2
    @property
    def cy(self):     return (self.top + self.bottom) / 2


# ─── feature extraction ───────────────────────────────────────────────────────

def _extract_features(words: List[WordBox]) -> Dict[str, float]:
    """
    Extract ~25 spatial + statistical features from a list of normalised word boxes.
    All features are dimensionless (layout-type agnostic).
    """
    import numpy as np

    if not words:
        return {k: 0.0 for k in _FEATURE_NAMES}

    # Basic geometry arrays
    lefts   = np.array([w.left   for w in words], dtype=float)
    rights  = np.array([w.right  for w in words], dtype=float)
    tops    = np.array([w.top    for w in words], dtype=float)
    bottoms = np.array([w.bottom for w in words], dtype=float)
    widths  = rights - lefts
    heights = bottoms - tops
    cxs     = (lefts + rights) / 2
    cys     = (tops + bottoms) / 2

    n = len(words)

    # ── column structure (DBSCAN on x-centres) ───────────────────────────────
    from sklearn.cluster import DBSCAN
    eps = 0.05   # 5% of page width
    cx_col  = cxs.reshape(-1, 1)
    labels  = DBSCAN(eps=eps, min_samples=3).fit_predict(cx_col)
    n_cols  = len(set(labels)) - (1 if -1 in labels else 0)

    # ── left-margin alignment ─────────────────────────────────────────────────
    # High alignment score → form or single column
    from scipy.stats import gaussian_kde
    try:
        kde  = gaussian_kde(lefts, bw_method=0.05)
        xs   = np.linspace(0, 1, 200)
        dens = kde(xs)
        left_peaks = int(np.sum(dens > (dens.max() * 0.3)))
    except Exception:
        left_peaks = 1

    # ── colon density (KV-pair indicator for forms) ───────────────────────────
    colon_ratio = sum(1 for w in words if ":" in w.text or w.text.strip().endswith(":")) / max(n, 1)

    # ── vertical line gap regularity (table row indicator) ───────────────────
    sorted_tops  = np.sort(np.unique(np.round(tops, 2)))
    if len(sorted_tops) > 1:
        gaps      = np.diff(sorted_tops)
        gap_cv    = float(gaps.std() / (gaps.mean() + 1e-9))   # lower = more regular
    else:
        gap_cv    = 1.0

    # ── word height variance (tables have very uniform word heights) ──────────
    h_cv = float(heights.std() / (heights.mean() + 1e-9)) if heights.mean() > 0 else 1.0

    # ── aspect ratio of text block bounding box ───────────────────────────────
    bb_w = float(rights.max() - lefts.min())
    bb_h = float(bottoms.max() - tops.min())
    bb_aspect = bb_w / max(bb_h, 1e-6)

    # ── coverage: fraction of page area occupied by words ─────────────────────
    coverage = float(np.sum(widths * heights))   # sum of word areas (normalised)

    # ── x-spread: standard deviation of word centres horizontally ────────────
    cx_std = float(cxs.std())
    cy_std = float(cys.std())

    # ── word count density per vertical band ──────────────────────────────────
    bands  = np.linspace(0, 1, 6)   # 5 bands
    band_counts = [int(np.sum((tops >= bands[i]) & (tops < bands[i+1]))) for i in range(5)]
    band_cv = float(np.std(band_counts) / (np.mean(band_counts) + 1e-9))

    # ── short-word ratio (labels in forms tend to be short) ───────────────────
    short_word_ratio = sum(1 for w in words if len(w.text.strip()) <= 3) / max(n, 1)

    # ── number token ratio (tables / invoices have many numbers) ─────────────
    import re as _re
    num_re = _re.compile(r"^[\d,\.\$€£%\-\+\/]+$")
    num_ratio = sum(1 for w in words if num_re.match(w.text.strip())) / max(n, 1)

    # ── horizontal overlap ratio (multi-column has two non-overlapping zones) ─
    # Split page at midpoint; measure overlap in word populations
    mid = 0.5
    left_zone  = np.sum(rights < mid)
    right_zone = np.sum(lefts  > mid)
    cross_zone = n - left_zone - right_zone
    bimodal_ratio = (left_zone + right_zone) / max(n, 1)   # high → two columns

    # ── avg words per line ────────────────────────────────────────────────────
    lines_dict: Dict[int, int] = {}
    for t in tops:
        bucket = round(t * 100)   # group within 1% of page height
        lines_dict[bucket] = lines_dict.get(bucket, 0) + 1
    avg_words_per_line = float(np.mean(list(lines_dict.values()))) if lines_dict else 0.0

    return {
        "n_words":           float(n),
        "n_columns_dbscan":  float(n_cols),
        "left_peaks_kde":    float(left_peaks),
        "colon_ratio":       colon_ratio,
        "gap_cv":            gap_cv,
        "height_cv":         h_cv,
        "bb_aspect":         bb_aspect,
        "coverage":          coverage,
        "cx_std":            cx_std,
        "cy_std":            cy_std,
        "band_cv":           band_cv,
        "short_word_ratio":  short_word_ratio,
        "num_ratio":         num_ratio,
        "bimodal_ratio":     bimodal_ratio,
        "avg_words_per_line": avg_words_per_line,
    }

_FEATURE_NAMES = [
    "n_words", "n_columns_dbscan", "left_peaks_kde", "colon_ratio",
    "gap_cv", "height_cv", "bb_aspect", "coverage",
    "cx_std", "cy_std", "band_cv", "short_word_ratio",
    "num_ratio", "bimodal_ratio", "avg_words_per_line",
]


# ─── heuristic fast-path ──────────────────────────────────────────────────────

def _heuristic(f: Dict[str, float]) -> Optional[LayoutResult]:
    """
    High-confidence rule-based decision.
    Returns None when rules are inconclusive → fall through to ML.
    """
    n     = f["n_words"]
    cols  = f["n_columns_dbscan"]
    colon = f["colon_ratio"]
    num   = f["num_ratio"]
    gap   = f["gap_cv"]
    h_cv  = f["height_cv"]
    bim   = f["bimodal_ratio"]

    if n < 5:
        return LayoutResult(SINGLE_COLUMN, 0.55, f, "Too few words to analyse")

    # Strong form signals: high colon density OR many colon words when page is small
    colon_count = round(colon * n)
    if (colon > 0.18 and colon_count >= 3) or (colon > 0.25 and n >= 8):
        conf = min(0.65 + colon * 1.2, 0.95)
        return LayoutResult(FORM, conf, f,
            f"Colon density {colon:.0%} ({colon_count} colons in {int(n)} words)")

    # Strong table signals: regular row spacing + low height variance + numeric content
    # Require num_ratio > 0.08 to avoid misclassifying uniform paragraph text as a table
    if gap < 0.25 and h_cv < 0.20 and num > 0.08:
        return LayoutResult(TABLE_HEAVY, 0.88, f,
            f"Regular row gaps (CV={gap:.2f}), uniform word height (CV={h_cv:.2f}), num_ratio={num:.0%}")

    # Paragraph text: dense uniform words, no numbers, no colons → single column
    # Also catches uniform-grid synthetic text that tricks DBSCAN into many columns
    _multi_signal = (f["cx_std"] > 0.25 and bim > 0.70) or (cols >= 3 and f["cx_std"] > 0.22 and bim > 0.55)
    if n > 50 and num < 0.06 and colon < 0.06:
        if not _multi_signal:
            return LayoutResult(SINGLE_COLUMN, 0.80, f,
                f"Dense text, no numeric/colon signals: n={int(n)}, cx_std={f['cx_std']:.2f}")

    # Strong two-column signal
    if cols >= 2 and bim > 0.70 and f["cx_std"] > 0.25:
        return LayoutResult(MULTI_COLUMN, 0.85, f,
            f"DBSCAN found {int(cols)} columns, bimodal_ratio={bim:.0%}")

    # Three-or-more column signal (middle columns straddle midpoint → lower bimodal)
    if cols >= 3 and f["cx_std"] > 0.22 and bim > 0.55:
        return LayoutResult(MULTI_COLUMN, 0.80, f,
            f"DBSCAN found {int(cols)} columns (3+), cx_std={f['cx_std']:.2f}")

    return None   # inconclusive


# ─── ML classifier (RandomForest on synthetic prototypes) ────────────────────

_MODEL = None          # lazy-loaded singleton
_MODEL_LOCK = False    # re-entrance guard


def _build_model():
    """
    Train a RandomForestClassifier on hand-crafted prototypical feature vectors.
    Each archetype is represented by 40 slightly-jittered samples drawn from
    known feature distributions for that layout type.
    No external dataset needed.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)

    def jitter(base: Dict[str, float], sigma: float, n: int) -> np.ndarray:
        arr  = np.array([base[k] for k in _FEATURE_NAMES], dtype=float)
        noise = rng.normal(0, sigma, size=(n, len(arr)))
        return np.clip(arr + noise, 0, None)

    # ── prototypical feature centroids ────────────────────────────────────────
    protos = {
        SINGLE_COLUMN: {
            "n_words": 180, "n_columns_dbscan": 1, "left_peaks_kde": 2,
            "colon_ratio": 0.02, "gap_cv": 0.55, "height_cv": 0.30,
            "bb_aspect": 0.70, "coverage": 0.08, "cx_std": 0.10,
            "cy_std": 0.28, "band_cv": 0.20, "short_word_ratio": 0.12,
            "num_ratio": 0.04, "bimodal_ratio": 0.35, "avg_words_per_line": 8.0,
        },
        MULTI_COLUMN: {
            "n_words": 240, "n_columns_dbscan": 2.5, "left_peaks_kde": 4,
            "colon_ratio": 0.01, "gap_cv": 0.45, "height_cv": 0.25,
            "bb_aspect": 1.10, "coverage": 0.11, "cx_std": 0.28,
            "cy_std": 0.26, "band_cv": 0.15, "short_word_ratio": 0.10,
            "num_ratio": 0.05, "bimodal_ratio": 0.80, "avg_words_per_line": 5.0,
        },
        FORM: {
            "n_words": 90, "n_columns_dbscan": 1.5, "left_peaks_kde": 3,
            "colon_ratio": 0.28, "gap_cv": 0.30, "height_cv": 0.18,
            "bb_aspect": 0.85, "coverage": 0.05, "cx_std": 0.12,
            "cy_std": 0.25, "band_cv": 0.30, "short_word_ratio": 0.22,
            "num_ratio": 0.12, "bimodal_ratio": 0.40, "avg_words_per_line": 3.5,
        },
        TABLE_HEAVY: {
            "n_words": 150, "n_columns_dbscan": 4, "left_peaks_kde": 5,
            "colon_ratio": 0.01, "gap_cv": 0.15, "height_cv": 0.12,
            "bb_aspect": 1.50, "coverage": 0.09, "cx_std": 0.30,
            "cy_std": 0.20, "band_cv": 0.10, "short_word_ratio": 0.08,
            "num_ratio": 0.35, "bimodal_ratio": 0.60, "avg_words_per_line": 6.0,
        },
        MIXED: {
            "n_words": 130, "n_columns_dbscan": 2, "left_peaks_kde": 4,
            "colon_ratio": 0.12, "gap_cv": 0.35, "height_cv": 0.25,
            "bb_aspect": 0.90, "coverage": 0.07, "cx_std": 0.18,
            "cy_std": 0.24, "band_cv": 0.22, "short_word_ratio": 0.15,
            "num_ratio": 0.18, "bimodal_ratio": 0.50, "avg_words_per_line": 4.5,
        },
    }

    X, y = [], []
    for label, proto in protos.items():
        samples = jitter(proto, sigma=0.04, n=40)
        X.extend(samples)
        y.extend([label] * 40)

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X, y)
    return clf


def _get_model():
    global _MODEL, _MODEL_LOCK
    if _MODEL is None and not _MODEL_LOCK:
        _MODEL_LOCK = True
        try:
            _MODEL = _build_model()
        finally:
            _MODEL_LOCK = False
    return _MODEL


def _ml_classify(f: Dict[str, float]) -> LayoutResult:
    import numpy as np

    clf = _get_model()
    if clf is None:
        return LayoutResult(SINGLE_COLUMN, 0.50, f, "ML model unavailable — default fallback")

    vec   = np.array([[f.get(k, 0.0) for k in _FEATURE_NAMES]])
    proba = clf.predict_proba(vec)[0]
    idx   = int(proba.argmax())
    label = clf.classes_[idx]
    conf  = float(proba[idx])

    top2  = sorted(zip(clf.classes_, proba), key=lambda x: -x[1])[:2]
    reason = (f"RF: {top2[0][0]} {top2[0][1]:.0%} "
              f"(runner-up: {top2[1][0]} {top2[1][1]:.0%})")
    return LayoutResult(label, conf, f, reason)


# ─── public API ───────────────────────────────────────────────────────────────

def classify_layout(
    words: List[WordBox],
    page_width:  int = 1,
    page_height: int = 1,
) -> LayoutResult:
    """
    Classify the layout type of a document page.

    Parameters
    ----------
    words       : list of WordBox with normalised 0–1 coordinates
    page_width  : pixel width of page (used only if coords are in pixels)
    page_height : pixel height (same)

    Returns
    -------
    LayoutResult with layout_type, confidence, features, reasoning
    """
    # Normalise pixel coords to 0-1 if any coord exceeds 1
    needs_norm = any(w.right > 2 or w.bottom > 2 for w in words)
    if needs_norm and page_width > 1 and page_height > 1:
        words = [
            WordBox(
                text=w.text,
                left=w.left   / page_width,
                top=w.top     / page_height,
                right=w.right / page_width,
                bottom=w.bottom / page_height,
                confidence=w.confidence,
            )
            for w in words
        ]

    feats = _extract_features(words)

    # Try heuristic first (fast, no model warm-up)
    result = _heuristic(feats)
    # n<5 case always returns immediately; other heuristics need >=0.80 to skip ML
    if result is not None and (result.confidence >= 0.80 or feats.get("n_words", 0) < 5):
        return result

    # ML classifier
    ml = _ml_classify(feats)

    # If heuristic was inconclusive but suggests something — blend
    if result is not None and result.confidence > 0.55:
        if result.layout_type == ml.layout_type:
            return LayoutResult(ml.layout_type,
                                min((result.confidence + ml.confidence) / 2 + 0.05, 0.97),
                                feats, f"Heuristic+RF agree: {result.reasoning}")
        # Disagreement — trust RF but lower confidence
        return LayoutResult(ml.layout_type,
                            ml.confidence * 0.90,
                            feats, f"RF override (heuristic said {result.layout_type}): {ml.reasoning}")

    return ml


def words_from_tesseract(tsv_data: dict, page_width: int, page_height: int) -> List[WordBox]:
    """
    Convert pytesseract's image_to_data dict into a list of WordBox
    with normalised 0-1 coordinates.
    """
    words = []
    texts = tsv_data.get("text", [])
    lefts = tsv_data.get("left", [])
    tops  = tsv_data.get("top",  [])
    ws    = tsv_data.get("width", [])
    hs    = tsv_data.get("height", [])
    confs = tsv_data.get("conf", [])

    for i, txt in enumerate(texts):
        if not str(txt).strip():
            continue
        conf_val = float(confs[i]) / 100.0 if confs[i] != -1 else 0.0
        if conf_val < 0.1:
            continue
        l = float(lefts[i])
        t = float(tops[i])
        r = l + float(ws[i])
        b = t + float(hs[i])
        words.append(WordBox(
            text=str(txt),
            left=l   / page_width,
            top=t    / page_height,
            right=r  / page_width,
            bottom=b / page_height,
            confidence=conf_val,
        ))
    return words
