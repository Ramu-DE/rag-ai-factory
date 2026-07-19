# -*- coding: utf-8 -*-
"""Build NB39: AI RAG Factory - IDP Pipeline Demo"""
import nbformat, os

NB_PATH = os.path.join(os.path.dirname(__file__), "notebooks", "NB39_IDP_Demo.ipynb")

def md(s):   return nbformat.v4.new_markdown_cell(s)
def code(s): return nbformat.v4.new_code_cell(s)

cells = []

cells.append(md("""\
# NB39 — AI RAG Factory: Intelligent Document Processing (IDP) Demo

> **AI Factory | NB39 | IDP Layer**

End-to-end demonstration of the IDP pipeline — classify, extract, validate, and index documents for Q&A.

| Step | What it shows |
|---|---|
| 1 | Document Classifier — heuristic + LLM routing |
| 2 | Document Skills — invoice, contract, medical, ID, custom YAML |
| 3 | Field Validator — required fields, format rules, cross-field checks |
| 4 | Incremental PDF Processor — per-page SHA-256 hash, skip unchanged |
| 5 | Full IDP Pipeline — single call end-to-end |
| 6 | Batch processing — multiple docs in one call |
| 7 | Export — CSV + JSON |
| 8 | Q&A over indexed documents — Ask tab equivalent |
"""))

# ── 1: Setup ──────────────────────────────────────────────────────────────────
cells.append(md("## Setup"))
cells.append(code("""\
import subprocess, sys
for p in ["pymupdf","boto3","qdrant-client","pydantic>=2.0","python-dotenv",
          "rank-bm25","numpy","rich","pandas"]:
    subprocess.run([sys.executable,"-m","pip","install",p,"-q"], check=False)
print("deps ready")
"""))

cells.append(code("""\
import os, sys, json
sys.path.insert(0, os.path.abspath(".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.abspath(".."), ".env"), override=True)

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich         import box

console = Console()
print("Environment ready")
"""))

# ── 2: Document Classifier ────────────────────────────────────────────────────
cells.append(md("""\
## Step 1 — Document Classifier

Two-layer classification:
- **Layer 1**: keyword heuristics (fast, free, no LLM call)
- **Layer 2**: LLM classifier only when heuristics are ambiguous or below threshold (>=4 matches, 2x ratio)
"""))
cells.append(code("""\
from rag_factory.document_classifier import CLASSIFIER

test_docs = [
    ("invoice_q1.pdf",
     "INVOICE #INV-2024-001 Bill To: John Smith Vendor: Acme Corp "
     "Invoice Date: 2024-01-15 Due Date: 2024-02-15 Payment Terms: Net 30 "
     "Item: Consulting 10hrs x $150 = $1,500 Subtotal: $1,500 Tax: $300 Total Due: $1,800 "
     "Remit to: accounts@acme.com Purchase Order PO-2024-99 VAT number GB123456789"),
    ("services_agreement.pdf",
     "SERVICES AGREEMENT This Agreement is entered into by and between Party A and Party B "
     "hereinafter referred to as the Parties. Effective Date: January 1 2024. "
     "Governing Law: State of New York. The parties agree to the following terms and conditions. "
     "Intellectual property created under this agreement shall be work-for-hire. "
     "Either party may terminate with 30 days notice. Liability cap: $50,000. Confidential."),
    ("patient_record.pdf",
     "CLINICAL NOTES Patient: Jane Doe DOB: 1985-03-12 Provider: Dr. Smith NPI: 1234567890 "
     "Chief Complaint: Persistent cough for 3 weeks. Diagnosis: J06.9 ICD-10 URI. "
     "Medications: Amoxicillin 500mg TID x 7 days. Assessment: Viral upper respiratory infection. "
     "Plan: Rest, fluids, follow up in 2 weeks if symptoms persist. Rx: Cough suppressant PRN."),
    ("climate_report.pdf",
     "Climate Change Report: Executive Summary. This report presents findings on global temperature "
     "trends and carbon emission trajectories. Recommendations for mitigation strategies are outlined "
     "in Appendix A. Table of contents lists 12 chapters covering renewable energy, carbon capture, "
     "and policy frameworks. Annual data shows 1.1 degree Celsius rise since pre-industrial levels."),
    ("passport_scan.pdf",
     "PASSPORT United States of America Surname: SMITH Given Names: JOHN ROBERT "
     "Nationality: USA Date of Birth: 15 MAR 1980 Sex: M Place of Birth: NEW YORK "
     "Date of Issue: 10 JAN 2020 Date of Expiry: 09 JAN 2030 "
     "MRZ: P<USASMITH<<JOHN<ROBERT<<<<<<<<<<<<< 12345678901USA8003159M3001099<<<<<<06"),
]

t = Table(title="Document Classifier Results", box=box.ROUNDED, show_lines=True)
t.add_column("Filename",        style="bold cyan", width=24)
t.add_column("Doc Type",        width=14)
t.add_column("Confidence",      width=12)
t.add_column("Method",          width=12)
t.add_column("Extr. Mode",      width=14)
t.add_column("Reason",          width=50)

for fname, text in test_docs:
    r = CLASSIFIER.classify(text, filename=fname)
    t.add_row(
        fname,
        r.doc_type,
        f"{r.confidence:.0%}",
        "heuristic" if r.heuristic else "LLM",
        r.extraction_mode,
        r.reason[:48],
    )
console.print(t)
"""))

# ── 3: Skills ─────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 2 — Document Skills

Each skill extracts structured fields from an extracted document.
Skills use Textract where available, fall back to LLM extraction.
"""))
cells.append(code("""\
from rag_factory.skills import InvoiceSkill, ContractSkill, MedicalSkill, list_skills
from rag_factory.skills.base import ExtractionResult

t2 = Table(title="Registered Document Skills", box=box.SIMPLE)
t2.add_column("Doc Type",    style="bold", width=18)
t2.add_column("Capability",               width=70)
for dt, desc in list_skills().items():
    t2.add_row(dt, desc)
console.print(t2)
"""))

cells.append(code("""\
# Test Invoice Skill with synthetic extracted doc
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class _FakeDoc:
    pages:          List[Any] = field(default_factory=list)
    expense_fields: List[Any] = field(default_factory=list)
    id_fields:      Dict[str,str] = field(default_factory=dict)

    @property
    def full_text(self):
        return "\\n".join(p.text for p in self.pages)

    @property
    def full_rich_text(self):
        return self.full_text

    @property
    def all_tables(self):
        return []

    @property
    def all_forms(self):
        return []

@dataclass
class _FakePage:
    text: str

invoice_doc = _FakeDoc(pages=[_FakePage(
    "INVOICE #INV-2024-001\\n"
    "Vendor: Acme Corp, 100 Main St, New York, NY 10001\\n"
    "Bill To: XYZ Ltd\\n"
    "Invoice Date: 2024-01-15\\nDue Date: 2024-02-15\\nPayment Terms: Net 30\\n"
    "Consulting Services: 10 hours x $200 = $2,000\\n"
    "Subtotal: $2,000  Tax (15%): $300  Total Due: $2,300\\n"
    "Please remit to: accounts@acme.com  Purchase Order: PO-2024-007"
)])

result = InvoiceSkill().run(invoice_doc)
console.print(Panel(
    "\\n".join(f"  {k:<22}: {v}" for k,v in result.fields.items()),
    title="InvoiceSkill extraction",
    border_style="green",
))
print(f"Confidence: {result.confidence:.0%}  |  Warnings: {result.warnings or 'none'}")
"""))

# ── 4: Custom Skill YAML ──────────────────────────────────────────────────────
cells.append(md("""\
## Step 3 — Custom Skill via YAML

No code needed. Define fields in YAML, run the skill on any document.
Three example YAMLs ship with the factory in `skills/`:
- `purchase_order.yaml`
- `insurance_claim.yaml`
- `employee_onboarding.yaml`
"""))
cells.append(code("""\
from rag_factory.skills.custom import CustomSkill
import os

yaml_path = os.path.abspath(os.path.join("..", "skills", "purchase_order.yaml"))
po_skill  = CustomSkill.from_yaml(yaml_path)
print(f"Loaded skill : {po_skill._skill_name}")
print(f"Doc type     : {po_skill.doc_type}")
print(f"Required     : {po_skill.required_fields}")
print(f"Fields       : {[f['name'] for f in po_skill._field_defs]}")
"""))

# ── 5: Field Validator ────────────────────────────────────────────────────────
cells.append(md("""\
## Step 4 — Field Validator

Rule engine covering:
- `required` — field must be present and non-empty
- `regex` — value must match expected format (date, amount, ICD-10 code)
- `numeric_range` — value must be within bounds
- `cross_field` — if field A present, field B must also be present
"""))
cells.append(code("""\
from rag_factory.field_validator import VALIDATOR

# Good invoice
vr_good = VALIDATOR.validate(
    fields={"vendor_name":"Acme Corp","invoice_date":"2024-01-15","total_amount":"$2,300.00"},
    doc_type="invoice", confidence=0.92
)
print(f"Good invoice   -> valid={vr_good.valid}  errors={vr_good.errors}  warnings={vr_good.warnings}")

# Bad invoice — missing required, bad date
vr_bad = VALIDATOR.validate(
    fields={"vendor_name":"","invoice_date":"tomorrow","total_amount":"two thousand"},
    doc_type="invoice", confidence=0.35
)
print(f"Bad invoice    -> valid={vr_bad.valid}")
for e in vr_bad.errors:   print(f"  ERROR  : {e}")
for w in vr_bad.warnings: print(f"  WARN   : {w}")

# Medical record
vr_med = VALIDATOR.validate(
    fields={"patient_name":"Jane Doe","provider_name":"Dr Smith","document_date":"2024-03-10",
            "diagnosis_codes":"J06.9","medications":"Amoxicillin 500mg"},
    doc_type="medical", confidence=0.85
)
print(f"\\nMedical record -> valid={vr_med.valid}  errors={vr_med.errors}")
"""))

# ── 6: Incremental PDF Processor ─────────────────────────────────────────────
cells.append(md("""\
## Step 5 — Incremental PDF Processor

Per-page SHA-256 hashing:
- **ADDED**: new page, embed it
- **UPDATED**: content changed, delete old chunks + re-embed
- **UNCHANGED**: skip (zero embedding cost)
- **DELETED**: page removed from PDF, delete old chunks

Industry impact: 100-page PDF with 3 changed pages → **97% cost saving**.
"""))
cells.append(code("""\
from rag_factory.pdf_processor import IncrementalPDFProcessor, _page_hash, _chunk_page

# Demonstrate hash stability
h1 = _page_hash("same text here")
h2 = _page_hash("same text here")
h3 = _page_hash("slightly different text")
print(f"Same content hash match  : {h1 == h2}")   # True
print(f"Different content no match: {h1 == h3}")  # False
print(f"Hash (first 20 chars)    : {h1[:20]}...")

# Demonstrate chunker
chunks = _chunk_page("A" * 1200, chunk_size=400, overlap=50)
print(f"\\n1200-char page -> {len(chunks)} chunks of ~400 chars with 50-char overlap")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: len={len(c)}, starts='{c[:20]}...'")

# Show index stats for any previously processed collection
proc  = IncrementalPDFProcessor(index_dir=os.path.join(os.path.abspath(".."), ".index"))
stats = proc.get_index_stats("idp_documents")
print(f"\\nIndex stats for 'idp_documents': {stats}")
"""))

# ── 7: Full IDP Pipeline ──────────────────────────────────────────────────────
cells.append(md("""\
## Step 6 — Full IDP Pipeline (single call)

`IDPPipeline.process(file_path)`:
1. Extract (PyMuPDF for digital, Textract for scans)
2. Classify (heuristic → LLM)
3. Re-extract with correct Textract mode if needed (expense/id)
4. Run skill (invoice/contract/medical/id)
5. Validate fields
6. Incremental ingest into Qdrant

All in one call. Zero boilerplate.
"""))
cells.append(code("""\
# Create a test PDF in memory for demonstration
import fitz, tempfile, os

# Build a synthetic invoice PDF
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), "INVOICE #INV-NB39-001", fontsize=16)
page.insert_text((50, 90),  "Vendor: Demo Corp, 1 Factory Lane, London EC1A 1BB", fontsize=11)
page.insert_text((50, 110), "Bill To: Research Institute Ltd", fontsize=11)
page.insert_text((50, 140), "Invoice Date: 2024-06-01   Due Date: 2024-07-01   Terms: Net 30", fontsize=11)
page.insert_text((50, 170), "Item: AI Research Services  Qty: 20hrs  Rate: $250  Amount: $5,000", fontsize=11)
page.insert_text((50, 190), "Item: Data Annotation      Qty: 50hrs  Rate: $80   Amount: $4,000", fontsize=11)
page.insert_text((50, 220), "Subtotal: $9,000   Tax (10%): $900   Total Due: $9,900", fontsize=11)
page.insert_text((50, 250), "Payment: accounts@democorp.ai   PO: PO-2024-NB39", fontsize=11)

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    doc.save(tmp.name)
    test_pdf = tmp.name
doc.close()

print(f"Test PDF created: {test_pdf}")
"""))

cells.append(code("""\
from rag_factory.idp_pipeline import IDPPipeline

pipeline = IDPPipeline(
    collection_name="nb39_idp_demo",
    index_dir=os.path.join(os.path.abspath(".."), ".index"),
    chunk_size=400,
)
result = pipeline.process(test_pdf)
os.unlink(test_pdf)

print(f"File       : {result.file_name}")
print(f"Doc type   : {result.doc_type}  ({result.classification.confidence:.0%} confidence)")
print(f"Method     : {result.method}  (scanned={result.is_scanned})")
print(f"Reason     : {result.classification.reason}")
print()

ir = result.ingest_report
print(f"Pages      : {ir.pages_total} total | {ir.pages_added} added | {ir.pages_skipped} skipped")
print(f"Chunks     : {ir.chunks_embedded} embedded in {ir.elapsed_ms}ms")
print()

if result.extraction:
    print("Extracted fields:")
    for k, v in result.extraction.fields.items():
        print(f"  {k:<22}: {v}")
    print(f"Confidence : {result.extraction.confidence:.0%}")

if result.validation:
    print(f"\\nValidation : valid={result.validation.valid}")
    for e in result.validation.errors:   print(f"  ERROR  : {e}")
    for w in result.validation.warnings: print(f"  WARN   : {w}")
"""))

# ── 8: Batch + Export ─────────────────────────────────────────────────────────
cells.append(md("""\
## Step 7 — Batch Processing + Export

Process multiple documents, then export results as CSV or JSON.
"""))
cells.append(code("""\
import pandas as pd, json

# Create 3 test PDFs (invoice, contract, climate report)
test_files = []

def _make_pdf(text: str, suffix: str = ".pdf") -> str:
    import fitz, tempfile
    doc  = fitz.open()
    page = doc.new_page()
    # split into lines to avoid overflow
    y = 50
    for line in text.split("\\n"):
        page.insert_text((50, y), line, fontsize=10)
        y += 16
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        doc.save(tmp.name)
    doc.close()
    return tmp.name

test_files.append(_make_pdf(
    "INVOICE #INV-BATCH-001\\nVendor: Batch Supplies Co\\nInvoice Date: 2024-05-10\\n"
    "Due Date: 2024-06-10\\nPayment Terms: Net 30\\nTotal Due: $3,500\\nTax: $350\\n"
    "Subtotal: $3,150\\nPO: PO-BATCH-001\\nRemit to: ap@batchsupplies.com"
))
test_files.append(_make_pdf(
    "SERVICE AGREEMENT\\nThis Agreement between Alpha Corp and Beta Ltd.\\n"
    "Effective Date: 2024-01-01\\nGoverning Law: California\\n"
    "Either party may terminate with 60 days written notice.\\n"
    "Liability cap: $100,000. All IP shall be work-for-hire.\\n"
    "Confidential information shall not be disclosed to third parties.\\n"
    "Indemnification clause applies to all direct damages."
))
test_files.append(_make_pdf(
    "CLIMATE SCIENCE QUARTERLY REPORT\\nExecutive Summary:\\n"
    "Global temperatures have risen 1.2C since pre-industrial levels.\\n"
    "Findings show accelerating ice sheet loss in Arctic regions.\\n"
    "Recommendations include immediate carbon capture investment.\\n"
    "Appendix A: Country emissions data table. Appendix B: Model parameters."
))

print(f"Created {len(test_files)} test PDFs")
"""))

cells.append(code("""\
# Process all three files
batch_results = []
pipeline2 = IDPPipeline(
    collection_name="nb39_batch_demo",
    index_dir=os.path.join(os.path.abspath(".."), ".index"),
)

for fp in test_files:
    r  = pipeline2.process(fp)
    ir = r.ingest_report
    batch_results.append({
        "file":        r.file_name,
        "doc_type":    r.doc_type,
        "confidence":  f"{r.classification.confidence:.0%}",
        "pages":       ir.pages_total,
        "chunks":      ir.chunks_embedded,
        "elapsed_ms":  r.elapsed_ms,
        "valid":       r.validation.valid if r.validation else "n/a",
        "errors":      "; ".join(r.validation.errors) if r.validation else "",
        "fields":      json.dumps(r.extraction.fields) if r.extraction else "{}",
    })
    os.unlink(fp)

df = pd.DataFrame(batch_results)

t3 = Table(title="Batch Processing Results", box=box.ROUNDED, show_lines=True)
for col in ["file","doc_type","confidence","pages","chunks","elapsed_ms","valid"]:
    t3.add_column(col, width=14)
for _, row in df.iterrows():
    t3.add_row(*[str(row[c]) for c in ["file","doc_type","confidence","pages","chunks","elapsed_ms","valid"]])
console.print(t3)

# Export CSV
csv_path  = "/tmp/nb39_batch_results.csv"
json_path = "/tmp/nb39_batch_results.json"
df.to_csv(csv_path, index=False)
with open(json_path, "w") as f:
    json.dump(batch_results, f, indent=2)
print(f"\\nCSV  exported : {csv_path}")
print(f"JSON exported : {json_path}")
"""))

# ── 9: Q&A ────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 8 — Q&A Over Indexed Documents

After IDP indexing, use the same RAG pipeline to answer questions.
The router picks the optimal pattern based on query intent.
"""))
cells.append(code("""\
from rag_factory.router import QueryRouter
from rag_factory.components.base import embed, dense_search, llm_call, get_qdrant_client
from rag_factory.guards import AmbiguityGuard, RetrievalGuard, SystemGuard, GenerationGuard

router = QueryRouter(specs_dir=os.path.join(os.path.abspath(".."), "specs"))
qdrant = get_qdrant_client()

queries = [
    ("What is the total amount due on the invoice?",          "nb39_idp_demo"),
    ("What are the termination terms in the agreement?",       "nb39_batch_demo"),
    ("What does the climate report recommend?",                "nb39_batch_demo"),
]

for query, collection in queries:
    routing  = router.route(query)
    a_result = AmbiguityGuard().run({"query": query, "retrieved_chunks": [], "tenant_id": "nb39"})
    eff_q    = a_result.get("query", query)

    vec    = embed(eff_q)
    try:
        points = dense_search(qdrant, collection, vec, k=5)
        chunks = [p.payload for p in points]
    except Exception:
        chunks = []

    context = "\\n\\n".join(c.get("text","") for c in chunks)
    answer  = llm_call(
        f"Answer using only the context.\\n\\nContext:\\n{context}\\n\\nQuestion: {query}",
        max_tokens=300,
    )

    console.print(Panel(
        f"[bold]Q:[/] {query}\\n"
        f"[dim]Spec: {routing.spec} | Category: {routing.category} | Chunks: {len(chunks)}[/]\\n\\n"
        f"[bold]A:[/] {answer[:400]}",
        border_style="cyan",
    ))
"""))

cells.append(md("""\
## Summary

| Component | What it does |
|---|---|
| `DocumentClassifier` | Heuristic (≥4 keywords, 2x ratio) + LLM fallback — classifies in <10ms |
| `InvoiceSkill` | Textract AnalyzeExpense + LLM gap-fill |
| `ContractSkill` | LLM: parties, dates, clauses, obligations |
| `MedicalSkill` | LLM + Textract forms: ICD-10, medications |
| `IDCardSkill` | Textract AnalyzeID |
| `CustomSkill` | YAML-defined fields — no code needed |
| `FieldValidator` | Required / regex / cross-field rules per doc_type |
| `IncrementalPDFProcessor` | Per-page SHA-256 hash — skip unchanged pages |
| `IDPPipeline` | Single call wiring all above + Qdrant ingest |
| `QueryRouter` | Heuristic + LLM routes query to optimal RAG spec |

**Next**: run `streamlit run app.py` and use the **IDP (Smart Extract)** + **Batch Upload** tabs.
"""))

nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"},
}
os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Written : {NB_PATH}")
print(f"Cells   : {len(cells)}")
