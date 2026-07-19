# -*- coding: utf-8 -*-
"""AI RAG Factory — accuracy + routing test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

# ── ROUTER ────────────────────────────────────────────────────────────────────
from rag_factory.router import QueryRouter
router = QueryRouter(specs_dir=os.path.join(os.path.dirname(__file__), "specs"))

router_tests = [
    # Summary / overview → simple
    ("please summarize",                           "simple.yaml",     "SIMPLE",  "summarize bare"),
    ("give me a summary of this document",         "simple.yaml",     "SIMPLE",  "give me a summary"),
    ("what are the key points",                    "simple.yaml",     "SIMPLE",  "key points"),
    ("tl;dr",                                      "simple.yaml",     "SIMPLE",  "tl;dr"),
    ("overview of the report",                     "simple.yaml",     "SIMPLE",  "overview"),
    ("what is this document about",                "simple.yaml",     "SIMPLE",  "what is about"),
    ("what is the main finding",                   "simple.yaml",     "SIMPLE",  "main finding"),
    ("provide a summary",                          "simple.yaml",     "SIMPLE",  "provide a summary"),
    # Factual → simple
    ("what is the total invoice amount",           "simple.yaml",     "SIMPLE",  "factual amount"),
    ("who is the vendor",                          "simple.yaml",     "SIMPLE",  "factual vendor"),
    ("when was the contract signed",               "simple.yaml",     "SIMPLE",  "factual date"),
    # Negation → simple
    ("what is not covered by the policy",          "simple.yaml",     "SIMPLE",  "negation not"),
    # Comparison → production
    ("compare medicare and medicaid",              "production.yaml", "COMPLEX", "comparison"),
    ("what are the differences between A and B",   "production.yaml", "COMPLEX", "difference"),
    ("pros and cons of each approach",             "production.yaml", "COMPLEX", "pros and cons"),
    ("contrast chapter 1 vs chapter 2",            "production.yaml", "COMPLEX", "vs contrast"),
    # Listing → production
    ("list all recommendations",                   "production.yaml", "COMPLEX", "list all"),
    ("how many chapters are there",                "production.yaml", "COMPLEX", "how many"),
    ("enumerate all findings",                     "production.yaml", "COMPLEX", "enumerate"),
    # Temporal → production
    ("what changed since 2022",                    "production.yaml", "COMPLEX", "temporal since"),
    ("latest findings on climate",                 "production.yaml", "COMPLEX", "temporal latest"),
    ("history of the regulation",                  "production.yaml", "COMPLEX", "history of"),
    # How-to → production
    ("how to apply for medicaid",                  "production.yaml", "COMPLEX", "how to"),
    ("walk me through the process",                "production.yaml", "COMPLEX", "walk me through"),
    ("step by step guide to setup",                "production.yaml", "COMPLEX", "step by step"),
    # Agentic → agentic
    ("research all findings across chapters",      "agentic.yaml",    "AGENTIC", "research"),
    ("comprehensive deep dive analysis",           "agentic.yaml",    "AGENTIC", "deep dive"),
    ("investigate all possible causes",            "agentic.yaml",    "AGENTIC", "investigate"),
]

rp = rf = 0
router_fails = []
for q, es, ec, label in router_tests:
    r = router.route(q)
    if r.spec == es and r.category == ec:
        rp += 1
    else:
        rf += 1
        router_fails.append((label, q, es, ec, r.spec, r.category, r.reason))


# ── CLASSIFIER ────────────────────────────────────────────────────────────────
from rag_factory.document_classifier import CLASSIFIER

clf_tests = [
    ("invoice_001.pdf",
     "INVOICE #INV-2024-001 Vendor: Acme Corp Invoice Date: 2024-01-15 Due Date: 2024-02-15 "
     "Payment Terms: Net 30 Subtotal: $1,500 Tax: $300 Total Due: $1,800 PO: PO-99 VAT: GB123456789 "
     "Remit to: accounts@acme.com Bill To: XYZ Ltd",
     "invoice", "invoice keywords"),

    ("services_agreement.pdf",
     "SERVICES AGREEMENT This Agreement is entered into by and between Party A and Party B "
     "Effective Date: January 1 2024 Governing Law: State of New York. "
     "Either party may terminate with 30 days notice. Liability cap: $50,000. "
     "Intellectual property shall be work-for-hire. Confidential. Indemnification clause.",
     "contract", "contract keywords"),

    ("patient_record.pdf",
     "CLINICAL NOTES Patient: Jane Doe DOB: 1985-03-12 Provider: Dr. Smith NPI: 1234567890 "
     "Diagnosis: J06.9 ICD-10 URI Medications: Amoxicillin 500mg TID x 7 days "
     "Chief Complaint: Persistent cough. Plan: Rest fluids follow up. Rx: PRN",
     "medical", "medical record"),

    ("climate_report.pdf",
     "Climate Change Report Executive Summary. This report presents findings on global temperature "
     "trends and carbon emission trajectories. Recommendations for mitigation strategies. "
     "Table of contents: 12 chapters on renewable energy, carbon capture, policy frameworks. "
     "Annual data shows 1.1 degree Celsius rise since pre-industrial levels.",
     "report", "climate report not medical"),

    ("passport_scan.pdf",
     "PASSPORT United States of America Surname: SMITH Given Names: JOHN ROBERT "
     "Nationality: USA Date of Birth: 15 MAR 1980 Sex: M "
     "Date of Issue: 10 JAN 2020 Date of Expiry: 09 JAN 2030 "
     "MRZ: P<USASMITH<<JOHN<ROBERT<<<<< 12345678901USA8003159M3001099<<<<<<06",
     "id_document", "passport MRZ"),

    ("drivers_license.pdf",
     "DRIVERS LICENSE State: California DL: D1234567 "
     "Name: JOHN DOE DOB: 01/15/1990 Exp: 01/15/2027 "
     "Address: 123 Main St Sacramento CA 95814 Class: C Restrictions: None",
     "id_document", "drivers license"),

    ("annual_report.pdf",
     "ANNUAL REPORT 2023 Fiscal Year Ended December 31 2023 "
     "Table of Contents Executive Summary Operations Finance Risk Appendix "
     "Revenue: $50M EBITDA: $12M Net Income: $8M EPS: $2.10 "
     "Board of Directors recommends dividend of $0.50 per share.",
     "report", "annual financial report"),

    ("random_text.pdf",
     "The quick brown fox jumps over the lazy dog. Some random unrelated text here. "
     "Nothing specific about any document type. General writing sample.",
     "other", "ambiguous random text"),

    ("purchase_order.pdf",
     "PURCHASE ORDER PO-2024-007 Vendor: Office Supplies Co Buyer: Tech Corp "
     "Order Date: 2024-03-01 Delivery Date: 2024-03-15 "
     "Item: Laptop x5 $1,200 each = $6,000. Subtotal: $6,000 Tax: $600 Total: $6,600 "
     "Approved by: John Manager Delivery to: 123 Main St",
     ("invoice", "form"), "purchase order (invoice or form both acceptable)"),
]

cp = cf = 0
clf_fails = []
for fname, text, exp, label in clf_tests:
    r = CLASSIFIER.classify(text, filename=fname)
    allowed = exp if isinstance(exp, tuple) else (exp,)
    if r.doc_type in allowed:
        cp += 1
    else:
        cf += 1
        clf_fails.append((label, fname, allowed, r.doc_type, r.confidence, r.reason))


# ── VALIDATOR ─────────────────────────────────────────────────────────────────
from rag_factory.field_validator import VALIDATOR

val_tests = [
    # Invoice
    ("invoice", {"vendor_name": "Acme Corp",  "invoice_date": "2024-01-15", "total_amount": "$1,800.00"}, True,  "invoice all valid"),
    ("invoice", {"vendor_name": "",            "invoice_date": "2024-01-15", "total_amount": "$1,800.00"}, False, "invoice missing vendor"),
    ("invoice", {"vendor_name": "Acme Corp",   "invoice_date": "next tuesday","total_amount": "$1,800.00"}, False,"invoice bad date = error"),
    ("invoice", {"vendor_name": "Acme Corp",   "invoice_date": "2024-01-15", "total_amount": "two thousand"}, True, "invoice bad amount = warn only"),
    ("invoice", {"vendor_name": "Acme Corp",   "invoice_date": "2024-01-15", "total_amount": "1800"},        True,  "invoice digits only amount"),
    ("invoice", {"vendor_name": "Acme Corp",   "invoice_date": "01/15/2024", "total_amount": "$1,800"},     True,  "invoice US date format"),
    # Medical
    ("medical", {"patient_name": "Jane Doe", "provider_name": "Dr Smith", "document_date": "2024-03-10", "diagnosis_codes": "J06.9", "medications": "Amoxicillin"}, True, "medical all valid"),
    ("medical", {"patient_name": "",          "provider_name": "Dr Smith", "document_date": "2024-03-10"}, False, "medical missing patient"),
    ("medical", {"patient_name": "Jane Doe",  "provider_name": "Dr Smith", "document_date": "not a date"},  False, "medical bad date"),
    # Contract
    ("contract", {"party_1": "Alpha Corp", "party_2": "Beta Ltd", "effective_date": "2024-01-01"}, True,  "contract all valid"),
    ("contract", {"party_1": "",            "party_2": "Beta Ltd", "effective_date": "2024-01-01"}, False, "contract missing party_1"),
    ("contract", {"party_1": "Alpha Corp",  "party_2": "",         "effective_date": "2024-01-01"}, False, "contract missing party_2"),
    # Catch-all types
    ("other",  {"anything": "goes here"}, True, "other always valid"),
    ("report", {},                         True, "report always valid"),
    ("form",   {"field": "value"},         True, "form always valid"),
    # Low confidence warning (still valid — no errors)
    ("invoice", {"vendor_name": "Acme", "invoice_date": "2024-01-15", "total_amount": "$100"}, True, "invoice low conf still valid"),
]

vp = vf = 0
val_fails = []
for dt, fields, exp, label in val_tests:
    conf = 0.85
    vr = VALIDATOR.validate(fields=fields, doc_type=dt, confidence=conf)
    if vr.valid == exp:
        vp += 1
    else:
        vf += 1
        val_fails.append((label, dt, exp, vr.valid, vr.errors, vr.warnings))


# ── PRINT REPORT ──────────────────────────────────────────────────────────────
total_p = rp + cp + vp
total_t = len(router_tests) + len(clf_tests) + len(val_tests)
pct = 100 * total_p // total_t

print("=" * 62)
print("  AI RAG Factory — Test Report")
print("=" * 62)
print(f"  Router     : {rp:>2}/{len(router_tests):>2}  ({100*rp//len(router_tests)}%)")
print(f"  Classifier : {cp:>2}/{len(clf_tests):>2}  ({100*cp//len(clf_tests)}%)")
print(f"  Validator  : {vp:>2}/{len(val_tests):>2}  ({100*vp//len(val_tests)}%)")
print(f"  -----------------------------------------")
print(f"  TOTAL      : {total_p:>2}/{total_t}  ({pct}%)")
print("=" * 62)

if router_fails:
    print("\nRouter FAILURES:")
    for label, q, es, ec, gs, gc, reason in router_fails:
        print(f"  [{label}]")
        print(f"    query    : {q!r}")
        print(f"    expected : {es} / {ec}")
        print(f"    got      : {gs} / {gc} -- {reason[:70]}")

if clf_fails:
    print("\nClassifier FAILURES:")
    for label, fname, exp, got, conf, reason in clf_fails:
        print(f"  [{label}] {fname}")
        print(f"    expected : {exp}")
        print(f"    got      : {got} ({conf:.0%}) -- {reason[:70]}")

if val_fails:
    print("\nValidator FAILURES:")
    for label, dt, exp, got, errs, warns in val_fails:
        print(f"  [{label}] doc_type={dt}  expected valid={exp}  got valid={got}")
        for e in errs:  print(f"    ERROR : {e}")
        for w in warns: print(f"    WARN  : {w}")

if total_p == total_t:
    print("\n  ALL TESTS PASSED")
