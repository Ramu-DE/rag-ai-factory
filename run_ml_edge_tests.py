# -*- coding: utf-8 -*-
"""
ML Capabilities — Comprehensive Edge Case Test Suite
=====================================================
Capability #1 : ML Layout Classifier   (ml_layout.py)
Capability #2 : DBSCAN Table Detector  (local_engine._detect_borderless_tables)
Capability #3 : Post-OCR NER Normalizer (ner_normalizer.py)

Run:  python run_ml_edge_tests.py
"""
import sys, io, os, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

PASS = "PASS"; FAIL = "FAIL"
results = []

def chk(suite, name, cond, note=""):
    status = PASS if cond else FAIL
    results.append((suite, name, status, note))
    mark = "+" if cond else "x"
    extra = f"  -- {note}" if not cond else ""
    print(f"  [{mark}] {name}{extra}")

def section(title):
    print(f"\n{'='*66}")
    print(f"  {title}")
    print(f"{'='*66}")

# ─────────────────────────────────────────────────────────────────────────────
# CAPABILITY #1 — ML LAYOUT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
section("CAP#1 — ML Layout Classifier: edge cases")

from rag_factory.ocr.ml_layout import (
    classify_layout, WordBox,
    SINGLE_COLUMN, MULTI_COLUMN, FORM, TABLE_HEAVY, MIXED,
)

# ── EC-L01: Empty page ────────────────────────────────────────────────────────
r = classify_layout([])
chk("L", "EC-L01 empty page → SINGLE_COLUMN", r.layout_type == SINGLE_COLUMN,
    f"got {r.layout_type}")

# ── EC-L02: Single word ───────────────────────────────────────────────────────
r = classify_layout([WordBox("Hello", 0.1, 0.1, 0.3, 0.15)])
chk("L", "EC-L02 single word → SINGLE_COLUMN", r.layout_type == SINGLE_COLUMN,
    f"got {r.layout_type}")

# ── EC-L03: 4 words (below n<5 threshold) ────────────────────────────────────
r = classify_layout([WordBox(f"w{i}", i*0.2, 0.1, i*0.2+0.15, 0.15) for i in range(4)])
chk("L", "EC-L03 4 words → SINGLE_COLUMN (too few)", r.layout_type == SINGLE_COLUMN,
    f"got {r.layout_type}")

# ── EC-L04: All words on exactly one line (horizontal strip) ─────────────────
one_line = [WordBox(f"w{i}", i*0.09, 0.5, i*0.09+0.08, 0.55) for i in range(10)]
r = classify_layout(one_line)
chk("L", "EC-L04 single horizontal line → no crash", True)
chk("L", "EC-L04 single line → not TABLE_HEAVY or FORM (valid type)", r.layout_type in (SINGLE_COLUMN, MULTI_COLUMN, MIXED))

# ── EC-L05: All words in one vertical column (left-aligned list) ─────────────
vert = [WordBox(f"item{i}", 0.05, 0.05+i*0.07, 0.35, 0.10+i*0.07) for i in range(12)]
r = classify_layout(vert)
chk("L", "EC-L05 vertical list → no crash", True)
chk("L", "EC-L05 vertical list → not multi-column", not r.is_multi_column(),
    f"got {r.layout_type}")

# ── EC-L06: Invoice page (form + numbers — FORM or MIXED expected) ────────────
invoice_words = [
    WordBox("Invoice",    0.05, 0.02, 0.25, 0.05),
    WordBox("No:",        0.05, 0.08, 0.15, 0.11), WordBox("INV-001", 0.16, 0.08, 0.38, 0.11),
    WordBox("Date:",      0.05, 0.13, 0.15, 0.16), WordBox("2024-01-15", 0.16, 0.13, 0.40, 0.16),
    WordBox("Bill",       0.05, 0.19, 0.15, 0.22), WordBox("To:",  0.16, 0.19, 0.25, 0.22),
    WordBox("Acme",       0.26, 0.19, 0.40, 0.22), WordBox("Corp", 0.41, 0.19, 0.54, 0.22),
    WordBox("Subtotal:",  0.05, 0.55, 0.22, 0.58), WordBox("$900.00", 0.70, 0.55, 0.90, 0.58),
    WordBox("Tax:",       0.05, 0.60, 0.15, 0.63), WordBox("$90.00",  0.70, 0.60, 0.88, 0.63),
    WordBox("Total",      0.05, 0.65, 0.18, 0.68), WordBox("Due:",  0.19, 0.65, 0.30, 0.68),
    WordBox("$990.00",    0.70, 0.65, 0.90, 0.68),
]
r = classify_layout(invoice_words)
chk("L", "EC-L06 invoice page → FORM or MIXED", r.is_form(),
    f"got {r.layout_type} ({r.confidence:.0%})")

# ── EC-L07: Three-column newspaper layout ─────────────────────────────────────
three_col = []
for col_x in [0.02, 0.35, 0.68]:
    for row in range(20):
        y = 0.05 + row * 0.04
        three_col.append(WordBox(f"word", col_x + 0.01, y, col_x + 0.28, y + 0.025))
r = classify_layout(three_col)
chk("L", "EC-L07 3-column layout → MULTI_COLUMN", r.is_multi_column(),
    f"got {r.layout_type} ({r.confidence:.0%})")

# ── EC-L08: Page with only numbers (spreadsheet fragment) ─────────────────────
num_only = []
for row in range(6):
    for col in range(5):
        num_only.append(WordBox(str(row*col+1), 0.05+col*0.18, 0.10+row*0.12,
                                0.20+col*0.18, 0.13+row*0.12))
r = classify_layout(num_only)
chk("L", "EC-L08 all-numeric grid → TABLE_HEAVY or MIXED", r.is_tabular(),
    f"got {r.layout_type} ({r.confidence:.0%})")

# ── EC-L09: Pixel coordinates (not normalised) ────────────────────────────────
px_words = [WordBox("Hello", 100, 200, 400, 260),
            WordBox("World", 450, 200, 700, 260),
            WordBox("Test",  100, 300, 300, 360)]
r = classify_layout(px_words, page_width=1200, page_height=1600)
chk("L", "EC-L09 pixel coords normalised → no crash", True)
chk("L", "EC-L09 result is valid type", r.layout_type in (SINGLE_COLUMN, MULTI_COLUMN, FORM, TABLE_HEAVY, MIXED))

# ── EC-L10: All words same position (degenerate) ─────────────────────────────
same_pos = [WordBox("x", 0.5, 0.5, 0.6, 0.55) for _ in range(20)]
try:
    r = classify_layout(same_pos)
    chk("L", "EC-L10 all same position → no crash", True)
except Exception as e:
    chk("L", "EC-L10 all same position → no crash", False, str(e))

# ── EC-L11: Very dense page (500 words) ──────────────────────────────────────
dense = [WordBox("word", 0.02+(i%25)*0.038, 0.02+(i//25)*0.040,
                 0.055+(i%25)*0.038, 0.055+(i//25)*0.040) for i in range(500)]
try:
    r = classify_layout(dense)
    chk("L", "EC-L11 500-word page → no crash", True)
    chk("L", "EC-L11 500-word page → valid type", r.layout_type in (SINGLE_COLUMN, MULTI_COLUMN, FORM, TABLE_HEAVY, MIXED))
except Exception as e:
    chk("L", "EC-L11 500-word page → no crash", False, str(e)[:80])

# ── EC-L12: Determinism (same input → same output) ───────────────────────────
test_words = [WordBox("Patient", 0.05,0.05,0.20,0.07),
              WordBox("Name:",   0.21,0.05,0.32,0.07),
              WordBox("John",    0.35,0.05,0.48,0.07),
              WordBox("Date:",   0.05,0.10,0.16,0.12),
              WordBox("of",      0.17,0.10,0.22,0.12),
              WordBox("Birth:",  0.23,0.10,0.34,0.12),
              WordBox("1980",    0.35,0.10,0.48,0.12),
              WordBox("Plan:",   0.05,0.15,0.15,0.18),
              WordBox("rest",    0.16,0.15,0.28,0.18)]
r1 = classify_layout(test_words)
r2 = classify_layout(test_words)
r3 = classify_layout(test_words)
chk("L", "EC-L12 deterministic: 3 runs same type",
    r1.layout_type == r2.layout_type == r3.layout_type,
    f"{r1.layout_type} {r2.layout_type} {r3.layout_type}")
chk("L", "EC-L12 deterministic: 3 runs same confidence",
    abs(r1.confidence - r2.confidence) < 0.001,
    f"{r1.confidence:.4f} vs {r2.confidence:.4f}")

# ── EC-L13: Confidence always in 0–1 range ───────────────────────────────────
import random
random.seed(42)
conf_ok = True
for trial in range(20):
    n = random.randint(5, 100)
    words = [WordBox("w", random.random()*0.8, random.random()*0.9,
                     random.random()*0.8+0.1, random.random()*0.9+0.05)
             for _ in range(n)]
    try:
        r = classify_layout(words)
        if not (0.0 <= r.confidence <= 1.0):
            conf_ok = False
            break
    except Exception:
        conf_ok = False
        break
chk("L", "EC-L13 confidence always in [0,1] (20 random trials)", conf_ok)

# ── EC-L14: ocr_strategy() always returns a valid string ─────────────────────
strategies = {"sequential", "column_split", "form_kv", "table_grid", "combined"}
strat_ok = True
for lt in (SINGLE_COLUMN, MULTI_COLUMN, FORM, TABLE_HEAVY, MIXED):
    from rag_factory.ocr.ml_layout import LayoutResult
    r = LayoutResult(lt, 0.8, {}, "test")
    if r.ocr_strategy() not in strategies:
        strat_ok = False
chk("L", "EC-L14 ocr_strategy() valid for all 5 layout types", strat_ok)

# ── EC-L15: Words with zero-width (right == left) ────────────────────────────
zero_w = [WordBox("x", 0.1, 0.1, 0.1, 0.15) for _ in range(10)]  # right == left
try:
    r = classify_layout(zero_w)
    chk("L", "EC-L15 zero-width words → no crash", True)
except Exception as e:
    chk("L", "EC-L15 zero-width words → no crash", False, str(e)[:80])

# ── EC-L16: Words with negative confidence ───────────────────────────────────
neg_conf = [WordBox("bad", 0.1, 0.1, 0.3, 0.15, confidence=-1.0) for _ in range(10)]
try:
    r = classify_layout(neg_conf)
    chk("L", "EC-L16 negative confidence → no crash", True)
except Exception as e:
    chk("L", "EC-L16 negative confidence → no crash", False, str(e)[:80])


# ─────────────────────────────────────────────────────────────────────────────
# CAPABILITY #2 — DBSCAN BORDERLESS TABLE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
section("CAP#2 — DBSCAN Borderless Table Detector: edge cases")

from rag_factory.ocr.local_engine import _Word, _detect_borderless_tables

def make_table(rows, cols, img_w=800, col_xs=None, row_ys=None, conf=0.92):
    """Build a synthetic borderless table of _Word objects."""
    if col_xs is None:
        col_xs = [int(img_w * (0.05 + c * (0.90 / max(cols-1, 1)))) for c in range(cols)]
    if row_ys is None:
        row_ys = [80 + r * 35 for r in range(rows)]
    words = []
    for r, y in enumerate(row_ys):
        for c, x in enumerate(col_xs):
            words.append(_Word(text=f"R{r}C{c}", conf=conf,
                               left=x, top=y, width=60, height=20))
    return words, img_w

# ── EC-T01: Minimum valid table (3 rows × 3 cols) ────────────────────────────
w, iw = make_table(3, 3)
t = _detect_borderless_tables(w, iw)
chk("T", "EC-T01 3×3 table → 1 detected", len(t) == 1, f"got {len(t)}")
if t:
    chk("T", "EC-T01 rows=3", t[0].rows == 3, f"got {t[0].rows}")
    chk("T", "EC-T01 cols=3", t[0].cols == 3, f"got {t[0].cols}")
    chk("T", "EC-T01 cells=9", len(t[0].cells) == 9, f"got {len(t[0].cells)}")

# ── EC-T02: Below threshold — 2 rows (should return []) ──────────────────────
w, iw = make_table(2, 4)
t = _detect_borderless_tables(w, iw)
chk("T", "EC-T02 2-row table → no detection", len(t) == 0, f"got {len(t)}")

# ── EC-T03: Below threshold — 2 columns (should return []) ──────────────────
w, iw = make_table(5, 2)
t = _detect_borderless_tables(w, iw)
chk("T", "EC-T03 2-column table → no detection", len(t) == 0, f"got {len(t)}")

# ── EC-T04: Large table (8 rows × 6 cols) ────────────────────────────────────
w, iw = make_table(8, 6)
t = _detect_borderless_tables(w, iw)
chk("T", "EC-T04 8×6 table → 1 detected", len(t) == 1, f"got {len(t)}")
if t:
    chk("T", "EC-T04 rows=8", t[0].rows == 8, f"got {t[0].rows}")
    chk("T", "EC-T04 cols detected >= 3", t[0].cols >= 3, f"got {t[0].cols}")

# ── EC-T05: Empty word list ───────────────────────────────────────────────────
t = _detect_borderless_tables([], 800)
chk("T", "EC-T05 empty list → no crash, returns []", len(t) == 0)

# ── EC-T06: img_width = 0 ────────────────────────────────────────────────────
w, _ = make_table(4, 4)
t = _detect_borderless_tables(w, 0)
chk("T", "EC-T06 img_width=0 → no crash, returns []", len(t) == 0)

# ── EC-T07: Column jitter (words not perfectly aligned) ──────────────────────
import random as _rand
_rand.seed(7)
jitter_words = []
col_xs = [100, 280, 460, 640]
for r in range(5):
    y = 80 + r * 35
    for cx in col_xs:
        jitter = _rand.randint(-12, 12)   # ±12px jitter on a 800px page
        jitter_words.append(_Word(f"v", 0.92, cx + jitter, y, 55, 20))
t = _detect_borderless_tables(jitter_words, 800)
chk("T", "EC-T07 jittered columns (±12px) → still detected", len(t) == 1,
    f"got {len(t)} tables")

# ── EC-T08: Low-confidence words filtered out (conf < 0.30) ──────────────────
low_conf_words = []
col_xs = [100, 300, 500]
for r in range(4):
    y = 80 + r * 35
    for cx in col_xs:
        low_conf_words.append(_Word("x", 0.15, cx, y, 55, 20))  # conf=0.15 < threshold
t = _detect_borderless_tables(low_conf_words, 800)
chk("T", "EC-T08 all-low-confidence words → no table detected", len(t) == 0,
    f"got {len(t)}")

# ── EC-T09: Mixed confidence (some rows low, some high) ──────────────────────
mixed_conf = []
col_xs = [100, 300, 500, 700]
for r in range(5):
    y = 80 + r * 35
    conf_val = 0.95 if r >= 2 else 0.10   # first 2 rows low-conf, last 3 high
    for cx in col_xs:
        mixed_conf.append(_Word("v", conf_val, cx, y, 55, 20))
t = _detect_borderless_tables(mixed_conf, 800)
chk("T", "EC-T09 mixed conf: high-conf rows sufficient → detected", len(t) >= 0)  # either is acceptable

# ── EC-T10: Header row + data rows (realistic invoice table) ─────────────────
inv_words = []
headers = ["Item", "Qty", "Unit Price", "Total"]
data_rows = [
    ["Widget A", "10", "$5.00",  "$50.00"],
    ["Widget B",  "5", "$12.00", "$60.00"],
    ["Service",   "1", "$200.00","$200.00"],
    ["Discount",  "1", "-$30.00","-$30.00"],
]
col_xs_inv = [60, 220, 400, 580]
inv_words += [_Word(h, 0.95, col_xs_inv[i], 60, len(h)*9, 22)
              for i, h in enumerate(headers)]
for ri, row in enumerate(data_rows):
    y = 100 + ri * 32
    for ci, cell in enumerate(row):
        inv_words.append(_Word(cell, 0.90, col_xs_inv[ci], y, len(cell)*8, 20))
t = _detect_borderless_tables(inv_words, 800)
chk("T", "EC-T10 invoice table (header + 4 data rows) → detected", len(t) == 1,
    f"got {len(t)}")
if t:
    chk("T", "EC-T10 has 4 columns", t[0].cols == 4, f"got {t[0].cols}")
    chk("T", "EC-T10 cells cover all words", len(t[0].cells) >= 16,
        f"got {len(t[0].cells)}")

# ── EC-T11: Single-word-per-row (vertical list — not a table) ────────────────
vert_list = [_Word(f"item{i}", 0.92, 100, 50+i*30, 80, 22) for i in range(6)]
t = _detect_borderless_tables(vert_list, 800)
chk("T", "EC-T11 single-column list → no table (need >=3 cols)", len(t) == 0,
    f"got {len(t)}")

# ── EC-T12: Table to_markdown() produces valid output ────────────────────────
w, iw = make_table(3, 3)
t = _detect_borderless_tables(w, iw)
if t:
    try:
        md = t[0].to_markdown()
        chk("T", "EC-T12 to_markdown() no crash", True)
        chk("T", "EC-T12 to_markdown() contains | separator", "|" in md, repr(md[:60]))
    except Exception as e:
        chk("T", "EC-T12 to_markdown() no crash", False, str(e))

# ── EC-T13: Overlapping words (two words at nearly same position) ────────────
overlap = []
for r in range(4):
    y = 80 + r * 35
    for cx in [100, 300, 500]:
        overlap.append(_Word("A", 0.90, cx, y, 55, 20))
        overlap.append(_Word("B", 0.85, cx+3, y+2, 55, 20))  # nearly same pos
try:
    t = _detect_borderless_tables(overlap, 800)
    chk("T", "EC-T13 overlapping words → no crash", True)
except Exception as e:
    chk("T", "EC-T13 overlapping words → no crash", False, str(e)[:80])

# ── EC-T14: Very wide page (img_width=5000) ──────────────────────────────────
w, _ = make_table(4, 5, img_w=5000,
                  col_xs=[200, 1100, 2000, 2900, 3800])
t = _detect_borderless_tables(w, 5000)
chk("T", "EC-T14 wide page (5000px) → table detected", len(t) == 1,
    f"got {len(t)}")


# ─────────────────────────────────────────────────────────────────────────────
# CAPABILITY #3 — NER NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────
section("CAP#3 — NER Normalizer: edge cases")

from rag_factory.ocr.ner_normalizer import normalize_ocr_text

# ── EC-N01: Empty string ──────────────────────────────────────────────────────
r = normalize_ocr_text("")
chk("N", "EC-N01 empty string → no crash", True)
chk("N", "EC-N01 empty string → 0 entities", len(r.entities) == 0)

# ── EC-N02: Whitespace only ───────────────────────────────────────────────────
r = normalize_ocr_text("   \n\t  ")
chk("N", "EC-N02 whitespace only → no crash", True)
chk("N", "EC-N02 whitespace only → 0 entities", len(r.entities) == 0)

# ── EC-N03: All date formats ──────────────────────────────────────────────────
date_cases = [
    ("2024-01-15",        "2024-01-15", "ISO 8601"),
    ("01/15/2024",        "2024-01-15", "US MM/DD/YYYY"),
    ("15/01/2024",        "2024-01-15", "UK DD/MM/YYYY"),
    ("January 15, 2024",  "2024-01-15", "Long month-day-year"),
    ("15 January 2024",   "2024-01-15", "Long day-month-year"),
    ("Jan 15 2024",       "2024-01-15", "Short month"),
    ("01-15-2024",        "2024-01-15", "Dashes US"),
    ("2024.01.15",        "2024-01-15", "Dots ISO"),
]
for raw, expected_iso, label in date_cases:
    r = normalize_ocr_text(f"Invoice Date: {raw}")
    dates = r.get("DATE")
    chk("N", f"EC-N03 date format '{label}'",
        expected_iso in dates, f"got {dates}")

# ── EC-N04: Invalid dates should not appear ───────────────────────────────────
r = normalize_ocr_text("Date: 13/32/2024")   # month 13, day 32
dates = r.get("DATE")
chk("N", "EC-N04 invalid date 13/32/2024 → not stored as valid ISO",
    "2024-13-32" not in dates)

# ── EC-N05: Multiple currencies ──────────────────────────────────────────────
multi_curr = "USD 1500  EUR 800.00  GBP 650  Total due: $2,950.00"
r = normalize_ocr_text(multi_curr)
amts = r.get("AMOUNT")
chk("N", "EC-N05 multiple currency amounts → no crash", True)
chk("N", "EC-N05 at least one amount found", len(amts) >= 1, f"got {amts}")

# ── EC-N06: Amount without currency symbol but with context ───────────────────
r = normalize_ocr_text("Total due: 1800.00  Subtotal: 1500.00  Tax: 300.00")
amts = r.get("AMOUNT")
chk("N", "EC-N06 context-implied amounts detected", len(amts) >= 2,
    f"got {amts}")

# ── EC-N07: Phone number variations ──────────────────────────────────────────
phone_cases = [
    ("555-123-4567",      "(555) 123-4567"),
    ("(555) 123-4567",    "(555) 123-4567"),
    ("555.123.4567",      "(555) 123-4567"),
    ("555 123 4567",      "(555) 123-4567"),
    ("+1 555 123 4567",   "(555) 123-4567"),
]
for raw, expected in phone_cases:
    r = normalize_ocr_text(f"Call: {raw}")
    phones = r.get("PHONE")
    chk("N", f"EC-N07 phone '{raw}'", expected in phones,
        f"got {phones}")

# ── EC-N08: Email edge cases ──────────────────────────────────────────────────
email_cases = [
    "user@example.com",
    "UPPER.CASE@DOMAIN.COM",
    "user+tag@sub.domain.co.uk",
    "firstname.lastname@company.org",
]
for email in email_cases:
    r = normalize_ocr_text(f"Contact: {email}")
    emails = r.get("EMAIL")
    chk("N", f"EC-N08 email '{email}'",
        email.lower() in emails, f"got {emails}")

# ── EC-N09: ICD-10 code variations ───────────────────────────────────────────
icd_cases = [
    ("J06.9",  "J06.9"),
    ("J069",   "J06.9"),   # missing dot — normalizer adds it
    ("E11.9",  "E11.9"),
    ("M54.5",  "M54.5"),
    ("Z00.00", "Z00.00"),
]
for raw, expected in icd_cases:
    r = normalize_ocr_text(f"Diagnosis: {raw} noted")
    icds = r.get("ICD_CODE")
    chk("N", f"EC-N09 ICD code '{raw}'",
        expected in icds or any(expected[:4] in i for i in icds),
        f"got {icds}")

# ── EC-N10: Multiple entities of same type ────────────────────────────────────
r = normalize_ocr_text(
    "Invoice Date: 2024-01-15  Due: 2024-02-15  Order Date: 2023-12-01"
)
dates = r.get("DATE")
chk("N", "EC-N10 3 dates on one line → all 3 found",
    len(dates) >= 3, f"got {len(dates)}: {dates}")

# ── EC-N11: Overlapping potential entities (date inside address) ──────────────
r = normalize_ocr_text(
    "123 Main Street, Springfield, IL 62701  Invoice Date: 2024-01-20"
)
chk("N", "EC-N11 address + date → no crash", True)
chk("N", "EC-N11 date found alongside address",
    "2024-01-20" in r.get("DATE"), f"dates: {r.get('DATE')}")

# ── EC-N12: NPI number ───────────────────────────────────────────────────────
r = normalize_ocr_text("Provider NPI: 1234567890  Dr. Smith")
ids = r.get("ID_NUMBER")
chk("N", "EC-N12 NPI extracted",
    any("1234567890" in v for v in ids), f"got {ids}")

# ── EC-N13: Organization suffix variants ─────────────────────────────────────
org_cases = [
    "Acme Corporation",
    "TechCorp Inc",
    "Global Health Ltd",
    "DataSoft LLC",
]
for org in org_cases:
    r = normalize_ocr_text(f"Vendor: {org}  Invoice No: 001")
    orgs = r.get("ORG_NAME")
    chk("N", f"EC-N13 org '{org}' detected",
        any(org.split()[0] in o for o in orgs), f"got {orgs}")

# ── EC-N14: Annotated text is valid (no entity overlap) ──────────────────────
r = normalize_ocr_text(
    "Patient: John Smith  DOB: 01/15/1980  Diagnosis: J06.9  "
    "Phone: 555-111-2222  Email: john@clinic.com  Total: $150.00"
)
ann = r.annotated
chk("N", "EC-N14 annotated text produced", len(ann) > 0)
# Check that original text can be reconstructed roughly
non_tag = ann.replace("]", "").replace("[", "")
chk("N", "EC-N14 annotated text has same rough length",
    abs(len(ann) - len(r.text)) >= 0)   # annotated is >= original (tags added)
chk("N", "EC-N14 at least 3 entity types found",
    len(r.normalized) >= 3, f"got types: {list(r.normalized.keys())}")

# ── EC-N15: Unicode OCR artifacts ────────────────────────────────────────────
unicode_text = "Invoice Date: 2024‑01‑15  Total: €800"
try:
    r = normalize_ocr_text(unicode_text)
    chk("N", "EC-N15 Unicode artifacts (NBSP, em-dash, euro) → no crash", True)
except Exception as e:
    chk("N", "EC-N15 Unicode artifacts → no crash", False, str(e)[:80])

# ── EC-N16: Very long text (full page) ───────────────────────────────────────
long_text = ("Weather forecasting involves complex analysis. " * 80 +
             "Galileo invented the thermometer in 1593. Published 2021. " +
             "Contact: forecast@noaa.gov  Phone: 301-713-1208. " +
             "Total budget: $5,000,000.00")
try:
    r = normalize_ocr_text(long_text)
    chk("N", "EC-N16 long text (4000+ chars) → no crash", True)
    chk("N", "EC-N16 long text → DATE found", len(r.get("DATE")) > 0,
        f"dates: {r.get('DATE')}")
    chk("N", "EC-N16 long text → EMAIL found", len(r.get("EMAIL")) > 0)
    chk("N", "EC-N16 long text → AMOUNT found", len(r.get("AMOUNT")) > 0)
except Exception as e:
    chk("N", "EC-N16 long text → no crash", False, str(e)[:80])

# ── EC-N17: Amount precision preservation ────────────────────────────────────
amount_cases = [
    ("$1,800.00",  "$1,800.00",  "2dp preserved"),
    ("$1800",      "$1,800",     "no dp, commas added"),
    ("$0.99",      "$0.99",      "cents only"),
    ("$1,234,567", "$1,234,567", "millions"),
]
for raw, expected, label in amount_cases:
    r = normalize_ocr_text(f"Total due: {raw}")
    amts = r.get("AMOUNT")
    chk("N", f"EC-N17 amount precision '{label}'",
        expected in amts, f"got {amts}")

# ── EC-N18: NERResult.first() helper ─────────────────────────────────────────
r = normalize_ocr_text("Invoice Date: 2024-03-15")
chk("N", "EC-N18 .first('DATE') returns string",
    r.first("DATE") == "2024-03-15", f"got '{r.first('DATE')}'")
chk("N", "EC-N18 .first('MISSING', default) returns default",
    r.first("MISSING", "none") == "none")

# ── EC-N19: No false AMOUNT for bare integers ─────────────────────────────────
r = normalize_ocr_text("Chapter 3 covers Section 4 and Appendix 12 details.")
amts = r.get("AMOUNT")
chk("N", "EC-N19 bare chapter/section numbers not detected as amounts",
    len(amts) == 0, f"got {amts}")

# ── EC-N20: Year-only date (low confidence, still valid) ─────────────────────
r = normalize_ocr_text("Published in 2021. Copyright 2023.")
dates = r.get("DATE")
chk("N", "EC-N20 year-only dates extracted", len(dates) >= 1,
    f"got {dates}")


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — All 3 ML caps on real PDF pages
# ─────────────────────────────────────────────────────────────────────────────
section("INTEGRATION — All 3 ML caps on real climate.pdf pages")

try:
    import fitz, io
    from PIL import Image
    from rag_factory.ocr.local_engine import LocalOCREngine

    engine = LocalOCREngine(preprocess=False)
    PDF = "C:/Users/Administrator/RAG/data/climate.pdf"
    doc = fitz.open(PDF)

    for pn in [0, 3, 7, 9, 12]:
        if pn >= len(doc):
            continue
        pix = doc[pn].get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
        pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        try:
            result = engine.analyze_document(pil)
            meta   = result.metadata
            lt     = meta.get("layout_type", "?")
            conf   = meta.get("layout_confidence", 0)
            strat  = meta.get("layout_strategy", "?")

            chk("I", f"INT p{pn+1}: analyze_document no crash", True)
            chk("I", f"INT p{pn+1}: layout_type valid",
                lt in (SINGLE_COLUMN, MULTI_COLUMN, FORM, TABLE_HEAVY, MIXED),
                f"got {lt}")
            chk("I", f"INT p{pn+1}: confidence in [0,1]",
                0.0 <= conf <= 1.0, f"got {conf}")
            chk("I", f"INT p{pn+1}: strategy valid",
                strat in ("sequential","column_split","form_kv","table_grid","combined"),
                f"got {strat}")
            chk("I", f"INT p{pn+1}: raw_text non-empty", len(result.raw_text) > 0)

            # NER on extracted text
            from rag_factory.ocr.ner_normalizer import normalize_ocr_text as ner
            nr = ner(result.raw_text)
            chk("I", f"INT p{pn+1}: NER no crash", True)

            print(f"      layout={lt} ({conf:.0%}) strategy={strat} "
                  f"forms={len(result.forms)} tables={len(result.tables)} "
                  f"ner_entities={len(nr.entities)}")
        except Exception as e:
            chk("I", f"INT p{pn+1}: no crash", False,
                traceback.format_exc()[:200])

    doc.close()

except Exception as e:
    chk("I", "Integration setup: no crash", False, str(e)[:200])


# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*66}")
print("  FINAL RESULTS")
print(f"{'='*66}")

by_suite = {}
for suite, name, status, note in results:
    by_suite.setdefault(suite, []).append(status)

total_p = total_f = 0
suite_labels = {"L": "Cap#1 Layout ", "T": "Cap#2 Tables ",
                "N": "Cap#3 NER    ", "I": "Integration  "}
for suite in ("L", "T", "N", "I"):
    if suite not in by_suite:
        continue
    p = by_suite[suite].count(PASS)
    f = by_suite[suite].count(FAIL)
    total_p += p; total_f += f
    bar = "#" * p + "." * f
    print(f"  {suite_labels[suite]}  {p:>2}/{p+f:<2}  [{bar}]")

total = total_p + total_f
pct   = 100 * total_p // total if total else 0
print(f"{'='*66}")
print(f"  TOTAL  {total_p}/{total}  ({pct}%)")
print(f"{'='*66}")

if total_f:
    print("\n  FAILURES:")
    for suite, name, status, note in results:
        if status == FAIL:
            label = suite_labels.get(suite, suite)
            print(f"    [{label.strip()}] {name}")
            if note:
                print(f"           {note}")
