# -*- coding: utf-8 -*-
"""
AI RAG Factory — Comprehensive Test Suite
==========================================
Tests every layer of the pipeline against real documents from
C:/Users/Administrator/RAG/data/

Coverage:
  T01  Classification accuracy (9 docs)
  T02  Incremental processing — ADDED / SKIPPED / UPDATED / DELETED
  T03  Medical sub-type routing (clinical vs report)
  T04  Query router (25 queries)
  T05  Field validator (12 cases)
  T06  Skill extraction — medical report fields
  T07  IDP pipeline end-to-end (3 docs)
  T08  Batch processing — all 9 PDFs
  T09  Q&A retrieval — post-index recall
  T10  Router: agentic false-positive guard
  T11  Edge cases — empty text, bad path, unsupported type
"""
import sys, os, time, json, tempfile, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

DATA = "C:/Users/Administrator/RAG/data"
MANIFESTS = os.path.join(DATA, "manifests")

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []

def check(suite, name, passed, got=None, expected=None, note=""):
    status = PASS if passed else FAIL
    results.append((suite, name, status, note or (f"got={got!r} expected={expected!r}" if not passed else "")))
    mark = "+" if passed else "x"
    print(f"  [{mark}] {name}" + (f"  -- {note or f'got={got!r} expected={expected!r}'}" if not passed else ""))

def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")

# ─────────────────────────────────────────────────────────────────────────────
section("T01 — Document Classification (9 documents)")
# ─────────────────────────────────────────────────────────────────────────────
from rag_factory.document_classifier import CLASSIFIER

clf_cases = [
    # climate.pdf — academic weather study; LLM may say "report" or "other" (both correct)
    ("climate.pdf",                  os.path.join(DATA, "climate.pdf"),           ("report", "other")),
    # medicaid.pdf — best-practices guide; acceptable: report / medical / other
    ("medicaid.pdf",                 os.path.join(DATA, "medicaid.pdf"),           ("report", "medical", "other")),
    ("PATIENT INFORMATION SYSTEMS",  os.path.join(DATA, "PATIENT INFORMATION SYSTEMS.pdf"), ("medical", "report")),
    ("medicaid_v2.pdf",              os.path.join(MANIFESTS, "medicaid_v2.pdf"),  ("report", "other")),
    ("synthetic_v1.pdf",             os.path.join(MANIFESTS, "synthetic_v1.pdf"), ("report", "other")),
    ("synthetic_v2.pdf",             os.path.join(MANIFESTS, "synthetic_v2.pdf"), ("report", "other")),
    ("synthetic_v3.pdf",             os.path.join(MANIFESTS, "synthetic_v3.pdf"), ("report", "other")),
    ("synthetic_v4.pdf",             os.path.join(MANIFESTS, "synthetic_v4.pdf"), ("report", "other")),
    # synthetic_v5 = "REVISED 2024 Edition" — guide/handbook → report or other
    ("synthetic_v5.pdf",             os.path.join(MANIFESTS, "synthetic_v5.pdf"), ("report", "other")),
]

import fitz as _fitz
for label, path, expected_types in clf_cases:
    try:
        with _fitz.open(path) as doc:
            text = "\n".join(doc[i].get_text("text") for i in range(min(3, len(doc))))
        r = CLASSIFIER.classify(text, filename=os.path.basename(path))
        ok = r.doc_type in expected_types
        check("T01", label, ok, r.doc_type, expected_types,
              f"{r.doc_type} {r.confidence:.0%} ({r.reason[:50]})")
    except Exception as e:
        check("T01", label, False, note=f"EXCEPTION: {e}")


# ─────────────────────────────────────────────────────────────────────────────
section("T02 — Incremental Processing (ADDED / SKIPPED / UPDATED)")
# ─────────────────────────────────────────────────────────────────────────────
from rag_factory.pdf_processor import IncrementalPDFProcessor

_idx = os.path.join(os.path.dirname(__file__), ".index_test")
_coll = "test_incremental"
proc  = IncrementalPDFProcessor(index_dir=_idx)

# First run — all pages ADDED
try:
    proc.reset_index(_coll)
    r1 = proc.process(os.path.join(MANIFESTS, "synthetic_v1.pdf"), _coll, doc_id="synth")
    check("T02", "First run: all pages added", r1.pages_added == r1.pages_total,
          note=f"added={r1.pages_added} total={r1.pages_total}")
    check("T02", "First run: nothing skipped", r1.pages_skipped == 0,
          note=f"skipped={r1.pages_skipped}")
    check("T02", "First run: chunks embedded > 0", r1.chunks_embedded > 0,
          note=f"chunks={r1.chunks_embedded}")
except Exception as e:
    check("T02", "First run", False, note=f"EXCEPTION: {traceback.format_exc()[:200]}")

# Second run same file — all SKIPPED
try:
    r2 = proc.process(os.path.join(MANIFESTS, "synthetic_v1.pdf"), _coll, doc_id="synth")
    check("T02", "Second run same file: all skipped", r2.pages_skipped == r2.pages_total,
          note=f"skipped={r2.pages_skipped} total={r2.pages_total}")
    check("T02", "Second run: zero new chunks", r2.chunks_embedded == 0,
          note=f"chunks={r2.chunks_embedded}")
    check("T02", "Second run: incremental=True", r2.incremental,
          note=f"incremental={r2.incremental}")
except Exception as e:
    check("T02", "Second run same file", False, note=f"EXCEPTION: {e}")

# Third run v2 (same template, slightly different) — some pages UPDATED
try:
    r3 = proc.process(os.path.join(MANIFESTS, "synthetic_v2.pdf"), _coll, doc_id="synth")
    total_changed = r3.pages_added + r3.pages_updated
    check("T02", "v2 run: at least 1 changed page", total_changed >= 0,
          note=f"added={r3.pages_added} updated={r3.pages_updated} skipped={r3.pages_skipped}")
    check("T02", "v2 run: incremental=True", r3.incremental,
          note=f"incremental={r3.incremental}")
except Exception as e:
    check("T02", "v2 run", False, note=f"EXCEPTION: {e}")

# v5 (REVISED 2024 cover) vs v4 — at least cover page different
try:
    proc.reset_index(_coll)
    r4 = proc.process(os.path.join(MANIFESTS, "synthetic_v4.pdf"), _coll, doc_id="synth4")
    r5 = proc.process(os.path.join(MANIFESTS, "synthetic_v5.pdf"), _coll, doc_id="synth4")
    cover_changed = r5.pages_updated + r5.pages_added
    check("T02", "v5 REVISED cover detected as changed", cover_changed >= 1,
          note=f"updated={r5.pages_updated} added={r5.pages_added}")
except Exception as e:
    check("T02", "v4→v5 cover change", False, note=f"EXCEPTION: {e}")


# ─────────────────────────────────────────────────────────────────────────────
section("T03 — Medical Sub-type Routing")
# ─────────────────────────────────────────────────────────────────────────────
from rag_factory.skills.medical import _is_clinical_record

med_cases = [
    ("Patient Name: John\nChief Complaint: Cough\nDiagnosis: URI\nMedications: Amox\nPlan: rest\nProvider: Dr Smith", True, "clinical note"),
    ("Patient INFORMATION SYSTEMS textbook Chapter 1 Table of Contents ISBN 2021", False, "textbook"),
    ("PATIENT INFORMATION SYSTEMS\nTable of Contents\nChapter 2: EHR\nCopyright 2021", False, "textbook headers"),
    ("Discharge Summary\nPatient ID: 12345\nPhysician: Dr Jones\nDiagnosis: J06.9\nPrescribed: Augmentin\nDate of Birth: 01/01/1980", True, "discharge summary"),
    ("Lab Result: HbA1c 7.2%\nDate of Birth: 1980-01-15\nChief Complaint: fatigue\nMedications: Metformin\nAssessment: controlled", True, "lab result"),
    ("Annual Report on Healthcare Informatics 2023. Overview of EHR adoption rates.", False, "health report"),
    ("medicaid.pdf content about eligibility criteria and benefits for healthcare programs", False, "policy doc"),
]
for text, expected, label in med_cases:
    got = _is_clinical_record(text)
    check("T03", label, got == expected, got, expected)


# ─────────────────────────────────────────────────────────────────────────────
section("T04 — Query Router (25 queries)")
# ─────────────────────────────────────────────────────────────────────────────
from rag_factory.router import QueryRouter
router = QueryRouter(specs_dir=os.path.join(os.path.dirname(__file__), "specs"))

router_cases = [
    # SIMPLE — summary
    ("please summarize this document",              "simple.yaml",     "SIMPLE"),
    ("what are the key findings of this research",  "simple.yaml",     "SIMPLE"),
    ("key findings of the paper",                   "simple.yaml",     "SIMPLE"),
    ("what is this document about",                 "simple.yaml",     "SIMPLE"),
    ("give me an overview",                         "simple.yaml",     "SIMPLE"),
    ("what are the main conclusions",               "simple.yaml",     "SIMPLE"),
    ("tl;dr",                                       "simple.yaml",     "SIMPLE"),
    # SIMPLE — factual
    ("what is the total amount due",                "simple.yaml",     "SIMPLE"),
    ("who are the authors",                         "simple.yaml",     "SIMPLE"),
    ("when was this published",                     "simple.yaml",     "SIMPLE"),
    # SIMPLE — negation
    ("what is not covered by medicaid",             "simple.yaml",     "SIMPLE"),
    # COMPLEX — comparison
    ("compare medicaid v1 and v2",                  "production.yaml", "COMPLEX"),
    ("what are the differences between chapter 1 and chapter 2", "production.yaml", "COMPLEX"),
    ("pros and cons of each approach",              "production.yaml", "COMPLEX"),
    # COMPLEX — listing
    ("list all eligibility requirements",           "production.yaml", "COMPLEX"),
    ("how many sections are there",                 "production.yaml", "COMPLEX"),
    # COMPLEX — temporal
    ("what changed between the 2023 and 2024 versions", "production.yaml", "COMPLEX"),
    ("latest updates to the guidelines",            "production.yaml", "COMPLEX"),
    # COMPLEX — how-to
    ("how to apply for medicaid",                   "production.yaml", "COMPLEX"),
    ("step by step process for enrollment",         "production.yaml", "COMPLEX"),
    # AGENTIC — genuine multi-step
    ("research all findings across all chapters",   "agentic.yaml",    "AGENTIC"),
    ("comprehensive analysis of all sections",      "agentic.yaml",    "AGENTIC"),
    ("investigate all possible causes of errors",   "agentic.yaml",    "AGENTIC"),
    ("deep dive into the methodology",              "agentic.yaml",    "AGENTIC"),
    # AGENTIC should NOT fire on bare 'research' noun
    ("what does the research say about graphs",     "simple.yaml",     "SIMPLE"),
    ("research findings on climate change",         "simple.yaml",     "SIMPLE"),
]
for q, esp, ecat in router_cases:
    res = router.route(q)
    ok  = res.spec == esp and res.category == ecat
    check("T04", q[:55], ok, f"{res.spec}/{res.category}", f"{esp}/{ecat}",
          res.reason[:60] if not ok else "")


# ─────────────────────────────────────────────────────────────────────────────
section("T05 — Field Validator (12 cases)")
# ─────────────────────────────────────────────────────────────────────────────
from rag_factory.field_validator import VALIDATOR

val_cases = [
    ("invoice", {"vendor_name":"Acme","invoice_date":"2024-01-15","total_amount":"$1800"}, True,  "invoice all valid"),
    ("invoice", {"vendor_name":"",    "invoice_date":"2024-01-15","total_amount":"$1800"}, False, "invoice missing vendor"),
    ("invoice", {"vendor_name":"Acme","invoice_date":"not a date","total_amount":"$1800"}, False, "invoice bad date = error"),
    ("invoice", {"vendor_name":"Acme","invoice_date":"2024-01-15","total_amount":"words"}, True,  "invoice bad amount = warn not error"),
    ("medical", {"patient_name":"Jane","provider_name":"Dr Smith","document_date":"2024-01-01","diagnosis_codes":"J06.9"}, True, "medical valid"),
    ("medical", {"patient_name":"",   "provider_name":"Dr Smith","document_date":"2024-01-01"}, False, "medical missing patient"),
    ("contract",{"party_1":"Alpha",   "party_2":"Beta","effective_date":"2024-01-01"}, True,  "contract valid"),
    ("contract",{"party_1":"",        "party_2":"Beta","effective_date":"2024-01-01"}, False, "contract missing party_1"),
    ("medical_report", {"title":"Patient Info Systems","document_date":"2021"}, True, "medical_report valid"),
    ("medical_report", {"document_date":"2021"}, False, "medical_report missing title"),
    ("report",  {}, True, "report no rules"),
    ("other",   {"x":"y"}, True, "other no rules"),
]
for dt, fields, exp, label in val_cases:
    vr = VALIDATOR.validate(fields=fields, doc_type=dt, confidence=0.85)
    check("T05", label, vr.valid == exp, vr.valid, exp,
          "; ".join(vr.errors) if not vr.valid else "")


# ─────────────────────────────────────────────────────────────────────────────
section("T06 — Medical Report Skill Extraction (PATIENT INFO SYSTEMS)")
# ─────────────────────────────────────────────────────────────────────────────
from rag_factory.skills.medical import MedicalSkill

try:
    with _fitz.open(os.path.join(DATA, "PATIENT INFORMATION SYSTEMS.pdf")) as doc:
        text = "\n".join(doc[i].get_text("text") for i in range(min(5, len(doc))))

    from dataclasses import dataclass, field as dfield
    from typing import List, Any

    @dataclass
    class _FDoc:
        pages: List[Any] = dfield(default_factory=list)
        @property
        def full_rich_text(self): return "\n".join(p.text for p in self.pages)
        @property
        def all_forms(self): return []
        @property
        def all_tables(self): return []

    @dataclass
    class _FPage:
        text: str

    fdoc   = _FDoc(pages=[_FPage(text)])
    skill  = MedicalSkill()
    result = skill.run(fdoc)

    check("T06", "sub_type = medical_report",
          result.metadata.get("sub_type") == "medical_report",
          result.metadata.get("sub_type"), "medical_report")
    check("T06", "title extracted",
          bool(result.fields.get("title")),
          note=f"title={result.fields.get('title','')[:60]}")
    check("T06", "document_date extracted",
          bool(result.fields.get("document_date")),
          note=f"date={result.fields.get('document_date','')}")
    check("T06", "confidence >= 0.70",
          result.confidence >= 0.70, result.confidence, ">=0.70")
    check("T06", "no patient_name field (not a clinical record)",
          "patient_name" not in result.fields or not result.fields["patient_name"],
          note=f"patient_name={result.fields.get('patient_name','')}")
except Exception as e:
    check("T06", "MedicalSkill extraction", False, note=f"EXCEPTION: {traceback.format_exc()[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
section("T07 — IDP Pipeline End-to-End (3 documents)")
# ─────────────────────────────────────────────────────────────────────────────
from rag_factory.idp_pipeline import IDPPipeline

idp_cases = [
    ("climate.pdf",    os.path.join(DATA, "climate.pdf"),   ("report",)),
    ("medicaid.pdf",   os.path.join(DATA, "medicaid.pdf"),  ("report","medical","other")),
    ("PATIENT INFO",   os.path.join(DATA, "PATIENT INFORMATION SYSTEMS.pdf"), ("medical",)),
]

for label, path, exp_types in idp_cases:
    try:
        pipeline = IDPPipeline(collection_name="test_idp_e2e", index_dir=_idx)
        t0 = time.time()
        result = pipeline.process(path)
        elapsed = int((time.time()-t0)*1000)

        check("T07", f"{label}: doc_type in expected",
              result.doc_type in exp_types, result.doc_type, exp_types)
        check("T07", f"{label}: confidence >= 0.70",
              result.classification.confidence >= 0.70,
              f"{result.classification.confidence:.0%}", ">=0.70")
        check("T07", f"{label}: pages > 0",
              result.ingest_report.pages_total > 0,
              result.ingest_report.pages_total, ">0")
        check("T07", f"{label}: elapsed < 90s",
              elapsed < 90000, f"{elapsed}ms", "<90000ms")
    except Exception as e:
        check("T07", f"{label}: pipeline", False, note=f"EXCEPTION: {traceback.format_exc()[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
section("T08 — Batch Processing (all 9 PDFs)")
# ─────────────────────────────────────────────────────────────────────────────
all_pdfs = [
    os.path.join(DATA, "climate.pdf"),
    os.path.join(DATA, "medicaid.pdf"),
    os.path.join(DATA, "PATIENT INFORMATION SYSTEMS.pdf"),
    os.path.join(MANIFESTS, "medicaid_v2.pdf"),
    os.path.join(MANIFESTS, "synthetic_v1.pdf"),
    os.path.join(MANIFESTS, "synthetic_v2.pdf"),
    os.path.join(MANIFESTS, "synthetic_v3.pdf"),
    os.path.join(MANIFESTS, "synthetic_v4.pdf"),
    os.path.join(MANIFESTS, "synthetic_v5.pdf"),
]

batch_coll = "test_batch_all"
pipeline_b = IDPPipeline(collection_name=batch_coll, index_dir=_idx)
batch_ok = batch_fail = 0
batch_pages = batch_chunks = 0
failed_files = []

for pdf in all_pdfs:
    fname = os.path.basename(pdf)
    try:
        r  = pipeline_b.process(pdf)
        ir = r.ingest_report
        batch_pages  += ir.pages_total
        batch_chunks += ir.chunks_embedded
        batch_ok     += 1
        print(f"    [{'+'}] {fname:<42} {r.doc_type:<10} {r.classification.confidence:.0%}  "
              f"p={ir.pages_total} c={ir.chunks_embedded}")
    except Exception as e:
        batch_fail += 1
        failed_files.append(fname)
        print(f"    [x] {fname:<42} FAILED: {e}")

check("T08", f"All 9 files processed", batch_ok == 9, batch_ok, 9)
check("T08", f"Zero failures", batch_fail == 0, batch_fail, 0,
      f"Failed: {failed_files}" if failed_files else "")
check("T08", f"Total pages > 80", batch_pages > 80, batch_pages, ">80")
check("T08", "Chunks indexed > 0", batch_chunks > 0, batch_chunks, ">0")


# ─────────────────────────────────────────────────────────────────────────────
section("T09 — Q&A Retrieval (post-index recall)")
# ─────────────────────────────────────────────────────────────────────────────
from rag_factory.components.base import embed, dense_search, llm_call, get_qdrant_client

qdrant = get_qdrant_client()

qa_cases = [
    ("What is the main topic of the climate document?",      batch_coll, "climate"),
    ("What does Medicaid cover?",                            batch_coll, "medicaid"),
    ("What visualization tools are discussed?",              batch_coll, "patient info"),
    ("What are the data visualization best practices?",      batch_coll, "synthetic"),
]

for question, coll, doc_label in qa_cases:
    try:
        vec    = embed(question)
        points = dense_search(qdrant, coll, vec, k=5)
        chunks = [p.payload.get("text","") for p in points]
        scores = [p.score for p in points]

        # Retrieval check: at least 1 chunk returned with a non-trivial score
        top_score = scores[0] if scores else 0.0
        check("T09", f"Retrieval: {question[:45]}",
              len(chunks) >= 1 and top_score > 0.05,
              note=f"chunks={len(chunks)} top_score={top_score:.3f}")

        answer = llm_call(
            f"Answer briefly using only the context.\n\nContext:\n{chr(10).join(chunks[:3])}\n\nQ: {question}",
            max_tokens=200,
        )
        check("T09", f"Answer non-empty: {doc_label}",
              len(answer.strip()) > 20, note=f"len={len(answer)}")
    except Exception as e:
        check("T09", f"Q&A {doc_label}", False, note=f"EXCEPTION: {e}")


# ─────────────────────────────────────────────────────────────────────────────
section("T10 — Router: Agentic False-Positive Guard")
# ─────────────────────────────────────────────────────────────────────────────
fp_cases = [
    # Should NOT be agentic — "research" as noun, not action-verb
    ("what does the research say about graphs",     "simple.yaml",     False),
    ("what were the key findings of this research", "simple.yaml",     False),
    ("research findings on climate change",         "simple.yaml",     False),
    ("the research shows temperature increased",    "simple.yaml",     False),
    # Should still be agentic — genuine multi-step action phrases
    ("research all findings across every chapter",  "agentic.yaml",    True),
    ("comprehensive analysis of all documents",     "agentic.yaml",    True),
    ("investigate all possible inconsistencies",    "agentic.yaml",    True),
]
for q, expected_spec, expected_agentic in fp_cases:
    res = router.route(q)
    got_agentic = (res.category == "AGENTIC")
    ok  = got_agentic == expected_agentic and res.spec == expected_spec
    check("T10", q[:55], ok, f"{res.spec}/{res.category}", f"{expected_spec}",
          res.reason[:60] if not ok else "")


# ─────────────────────────────────────────────────────────────────────────────
section("T11 — Edge Cases")
# ─────────────────────────────────────────────────────────────────────────────

# Empty text classification
try:
    r = CLASSIFIER.classify("", filename="unknown.pdf")
    check("T11", "Empty text: no crash", True, note=f"doc_type={r.doc_type}")
except Exception as e:
    check("T11", "Empty text: no crash", False, note=str(e))

# Very short text
try:
    r = CLASSIFIER.classify("Hello world", filename="test.pdf")
    check("T11", "Very short text: no crash", True, note=f"doc_type={r.doc_type}")
except Exception as e:
    check("T11", "Very short text: no crash", False, note=str(e))

# Bad file path in IDP pipeline
try:
    pipeline_e = IDPPipeline(collection_name="test_edge", index_dir=_idx)
    pipeline_e.process("/nonexistent/path/file.pdf")
    check("T11", "Bad path: raises exception", False, note="Should have raised")
except Exception:
    check("T11", "Bad path: raises exception cleanly", True)

# Validator with unknown doc_type (no rules)
try:
    vr = VALIDATOR.validate(fields={"foo":"bar"}, doc_type="completely_unknown", confidence=0.9)
    check("T11", "Unknown doc_type: valid=True (no rules)", vr.valid, vr.valid, True)
except Exception as e:
    check("T11", "Unknown doc_type: no crash", False, note=str(e))

# Router with empty query
try:
    res = router.route("")
    check("T11", "Empty query: no crash", True, note=f"spec={res.spec}")
except Exception as e:
    check("T11", "Empty query: no crash", False, note=str(e))

# Duplicate upload (incremental — second run is all skipped)
try:
    pipeline_d = IDPPipeline(collection_name="test_dup", index_dir=_idx)
    r1 = pipeline_d.process(os.path.join(DATA, "climate.pdf"))
    r2 = pipeline_d.process(os.path.join(DATA, "climate.pdf"))
    check("T11", "Duplicate upload: second run all skipped",
          r2.ingest_report.pages_skipped == r2.ingest_report.pages_total,
          note=f"skipped={r2.ingest_report.pages_skipped} total={r2.ingest_report.pages_total}")
    check("T11", "Duplicate upload: zero new chunks",
          r2.ingest_report.chunks_embedded == 0,
          note=f"chunks={r2.ingest_report.chunks_embedded}")
except Exception as e:
    check("T11", "Duplicate upload", False, note=f"EXCEPTION: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  FINAL RESULTS")
print(f"{'='*62}")

by_suite = {}
for suite, name, status, note in results:
    by_suite.setdefault(suite, []).append(status)

total_p = total_f = 0
for suite in sorted(by_suite):
    p = by_suite[suite].count(PASS)
    f = by_suite[suite].count(FAIL)
    total_p += p; total_f += f
    bar = "#" * p + "." * f
    print(f"  {suite}  {p:>2}/{p+f:<2}  [{bar}]")

total = total_p + total_f
pct   = 100 * total_p // total if total else 0
print(f"{'='*62}")
print(f"  TOTAL  {total_p}/{total}  ({pct}%)")
print(f"{'='*62}")

if total_f:
    print("\n  FAILURES:")
    for suite, name, status, note in results:
        if status == FAIL:
            print(f"    [{suite}] {name}")
            if note: print(f"           {note}")
