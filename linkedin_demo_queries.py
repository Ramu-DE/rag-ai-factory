# -*- coding: utf-8 -*-
"""
LinkedIn Demo Queries — ML Capabilities on Real PDFs
======================================================
Runs 5 demos against the 3 PDFs in C:/Users/Administrator/RAG/data/
using the Layout Classifier, NER Normalizer, AutoSplitter, and AutoCropper.

Run:
    python linkedin_demo_queries.py

PDFs used:
    climate.pdf                           — 13-page academic geography paper
    medicaid.pdf                          — 10-page healthcare data viz guide
    PATIENT INFORMATION SYSTEMS - Copy.pdf — 14-page medical informatics paper
"""
import fitz
from rag_factory.ocr.ml_layout   import WordBox, classify_layout
from rag_factory.ocr.ner_normalizer import NERNormalizer
from rag_factory.ocr.split       import AutoSplitter

DATA = "C:/Users/Administrator/RAG/data"
PDFS = {
    "climate" : f"{DATA}/climate.pdf",
    "medicaid": f"{DATA}/medicaid.pdf",
    "patient" : f"{DATA}/PATIENT INFORMATION SYSTEMS - Copy.pdf",
}

ner      = NERNormalizer()
splitter = AutoSplitter()

SEP  = "=" * 68
THIN = "-" * 68

def wordboxes(page):
    pw, ph = page.rect.width, page.rect.height
    return [
        WordBox(text=w[4], left=w[0]/pw, top=w[1]/ph,
                right=w[2]/pw, bottom=w[3]/ph)
        for w in page.get_text("words")
    ]

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 1 — Layout Classifier: sweep every page across all 3 PDFs
# Shows:  classifier firing heuristic vs RandomForest, confidence per page
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 1 — Layout Classifier: full-document page sweep")
print(f"{SEP}")
print(f"{'PDF':<10} {'Page':>4}  {'Layout':<15} {'Conf':>5}  {'Words':>5}  Reasoning")
print(THIN)

for name, path in PDFS.items():
    doc = fitz.open(path)
    for pno in range(len(doc)):
        wbs = wordboxes(doc[pno])
        r   = classify_layout(wbs, 1, 1)
        print(f"{name:<10} {pno:>4}  {r.layout_type:<15} {r.confidence:>4.0%}  "
              f"{len(wbs):>5}  {r.reasoning}")
    doc.close()

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2 — NER Normalizer: entity extraction from patient PDF (page 1)
# Shows:  PERSON_NAME, EMAIL, DATE extracted from a research paper cover page
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 2 — NER Normalizer: author/contact extraction (patient PDF p.1)")
print(f"{SEP}")

doc  = fitz.open(PDFS["patient"])
text = doc[1].get_text("text")
doc.close()

print("Input text (first 400 chars):")
print(text[:400].strip())
print(THIN)

result = ner.run(text)
print(f"\nEntities found: {len(result.entities)}")
print(f"{'Type':<15} {'Raw':<40} {'Normalized'}")
print(THIN)
for e in result.entities:
    print(f"{e.entity_type:<15} {repr(e.raw):<40} {repr(e.normalized)}")

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3 — NER Normalizer: ICD code + date normalisation (patient PDF p.3)
# Shows:  ICD_CODE P10, DATE → YYYY, AMOUNT normalisation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 3 — NER Normalizer: ICD code + date normalisation (patient PDF p.3)")
print(f"{SEP}")

doc  = fitz.open(PDFS["patient"])
text = doc[3].get_text("text")
doc.close()

result = ner.run(text)
interesting = [e for e in result.entities
               if e.entity_type in ("ICD_CODE", "DATE", "AMOUNT")]
print(f"{'Type':<15} {'Raw':<35} {'Normalized'}")
print(THIN)
for e in interesting:
    print(f"{e.entity_type:<15} {repr(e.raw):<35} {repr(e.normalized)}")

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4 — NER Normalizer: reference bibliography scan (medicaid PDF p.9)
# Shows:  person names + publication dates extracted from a references page
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 4 — NER Normalizer: bibliography scan (medicaid PDF p.9)")
print(f"{SEP}")

doc  = fitz.open(PDFS["medicaid"])
text = doc[9].get_text("text")
doc.close()

print("Input text (first 500 chars):")
print(text[:500].strip())
print(THIN)

result = ner.run(text)
print(f"\n{'Type':<15} {'Raw':<40} {'Normalized'}")
print(THIN)
for e in result.entities:
    print(f"{e.entity_type:<15} {repr(e.raw):<40} {repr(e.normalized)}")

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 5 — AutoSplitter: detect document boundaries in patient PDF
# Shows:  boundary page indices, segment count, page ranges, layout types
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 5 — AutoSplitter: document boundary detection (patient PDF, 14 pages)")
print(f"{SEP}")

result = splitter.split_pdf(PDFS["patient"], classify=False)

print(f"Total pages   : {result.total_pages}")
print(f"Segments found: {len(result.segments)}")
print(f"Boundaries    : {result.boundaries}")
print(f"Method        : {result.method}")
print(THIN)
print(f"{'Seg':>3}  {'Pages':>10}  {'Count':>5}  {'Doc Type':<12}  {'Conf':>5}  Layout Types")
print(THIN)
for seg in result.segments:
    pages = f"p{seg.page_start}–p{seg.page_end}"
    layouts = ", ".join(sorted(set(seg.layout_types)))
    print(f"{seg.segment_idx:>3}  {pages:>10}  {seg.page_count:>5}  "
          f"{seg.doc_type:<12}  {seg.confidence:>5.0%}  {layouts}")

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 6 — Layout + NER pipeline: climate PDF pages with most content
# Shows:  end-to-end: classify layout then extract entities from same page
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 6 — Layout + NER pipeline: climate PDF pages 3, 7, 11")
print(f"{SEP}")

doc = fitz.open(PDFS["climate"])
for pno in [3, 7, 11]:
    page = doc[pno]
    text = page.get_text("text")
    wbs  = wordboxes(page)

    layout = classify_layout(wbs, 1, 1)
    ner_r  = ner.run(text[:3000])

    entity_summary = {}
    for e in ner_r.entities:
        entity_summary.setdefault(e.entity_type, []).append(e.normalized)

    print(f"\n  climate.pdf  page {pno}")
    print(f"  Layout  : {layout.layout_type}  (confidence {layout.confidence:.0%})")
    print(f"  Reasoning: {layout.reasoning}")
    print(f"  Words   : {len(wbs)}")
    print(f"  Entities: {len(ner_r.entities)}")
    for etype, vals in entity_summary.items():
        preview = ", ".join(repr(v) for v in vals[:4])
        more    = f" +{len(vals)-4} more" if len(vals) > 4 else ""
        print(f"    {etype:<15}: {preview}{more}")
doc.close()

print(f"\n{SEP}")
print("All demos complete.")
print(SEP)
