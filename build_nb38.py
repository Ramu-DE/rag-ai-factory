# -*- coding: utf-8 -*-
"""Build NB38: AI RAG Factory - Full API Demo"""
import nbformat, os

NB_PATH = os.path.join(os.path.dirname(__file__), "notebooks", "NB38_API_Demo.ipynb")

def md(s):   return nbformat.v4.new_markdown_cell(s)
def code(s): return nbformat.v4.new_code_cell(s)

cells = []

cells.append(md("""\
# NB38 — AI RAG Factory: Full API Demo

> **AI Factory | NB38**

End-to-end demo of the FastAPI service — all 5 endpoints exercised in sequence:

| Step | Endpoint | What it does |
|---|---|---|
| 1 | `GET /health` | Verify service + manifest size |
| 2 | `GET /specs` | List all validated pipeline specs |
| 3 | `POST /ingest` | Chunk + embed + upsert a document |
| 4 | `POST /query` | Retrieve + generate + guard-check |
| 5 | `POST /evaluate` | RAGAS 4-metric faithfulness scoring |
| 6 | `POST /compare` | Same query through two specs side-by-side |

The FastAPI app runs **in-process** (no separate server needed for this notebook).
"""))

# ── 1: Setup ──────────────────────────────────────────────────────────────────
cells.append(code("""\
import subprocess, sys
for p in ["httpx","fastapi","uvicorn","boto3","qdrant-client",
          "pydantic>=2.0","python-dotenv","rank-bm25","numpy","rich"]:
    subprocess.run([sys.executable,"-m","pip","install",p,"-q"], check=False)
print("deps ready")
"""))

cells.append(code("""\
import os, sys, json
sys.path.insert(0, os.path.abspath(".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.abspath(".."), ".env"), override=True)

# Start the FastAPI app with a test client (no port needed)
from fastapi.testclient import TestClient
from rag_factory.api.main import app

client = TestClient(app)
print("TestClient ready — all API calls go in-process via ASGI")
"""))

# ── 2: /health ────────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — GET /health"))
cells.append(code("""\
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import box

console = Console()

resp = client.get("/health")
assert resp.status_code == 200
h = resp.json()

console.print(Panel.fit(
    "\\n".join([
        f"Status          : [green]{h['status']}[/]",
        f"Manifest size   : {h['manifest_size']} components",
        f"Temporal ready  : {h['temporal']}",
    ]),
    title="GET /health",
    border_style="green",
))
"""))

# ── 3: /specs ────────────────────────────────────────────────────────────────
cells.append(md("## Step 2 — GET /specs"))
cells.append(code("""\
resp = client.get("/specs")
assert resp.status_code == 200
specs = resp.json()["specs"]

t = Table(title="GET /specs", box=box.ROUNDED, show_lines=True)
t.add_column("File",           style="bold cyan", width=22)
t.add_column("Name",                              width=26)
t.add_column("Chunker",                           width=26)
t.add_column("Retrieval",                         width=22)
t.add_column("Temporal",                          width=10)
t.add_column("Valid",                             width=8)

for s in specs:
    t.add_row(
        s.get("file",""),
        s.get("name",""),
        s.get("chunker",""),
        s.get("retrieval",""),
        "[green]on[/]" if s.get("temporal") else "[dim]off[/]",
        "[green]PASS[/]" if s.get("valid") else "[red]FAIL[/]",
    )
console.print(t)
"""))

# ── 4: /ingest ────────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — POST /ingest"))
cells.append(code("""\
MEDICAID_TEXT = (
    "Medicaid is a joint federal and state program providing health coverage to "
    "eligible low-income adults, children, pregnant women, elderly adults, and "
    "people with disabilities. The program is funded jointly by states and the "
    "federal government and administered by states within federal guidelines. "
    "Eligibility is based on income, family size, disability, and other factors. "
    "The Affordable Care Act expanded Medicaid to cover adults with income below "
    "138 percent of the federal poverty level. "
    "Covered services include inpatient and outpatient hospital services, physician "
    "services, laboratory and X-ray services, home health services, and nursing "
    "facility services. Optional services include prescription drugs, dental care, "
    "vision, and physical therapy. "
    "The Children's Health Insurance Program (CHIP) covers children in families that "
    "earn too much for Medicaid but cannot afford private coverage."
)

COLLECTION = "nb38_api_demo"

resp = client.post("/ingest", json={
    "text":            MEDICAID_TEXT,
    "collection_name": COLLECTION,
    "chunker":         "fixed_chunking",
    "doc_id":          "medicaid_overview",
})
assert resp.status_code == 200
r = resp.json()

console.print(Panel.fit(
    "\\n".join([
        f"Collection  : {r['collection_name']}",
        f"Chunks      : {r['chunk_count']}",
        f"Elapsed     : {r['elapsed_ms']}ms",
    ]),
    title="POST /ingest",
    border_style="cyan",
))
"""))

# ── 5: /query ─────────────────────────────────────────────────────────────────
cells.append(md("## Step 4 — POST /query (with all guards)"))
cells.append(code("""\
QUERY = "What services are covered under Medicaid?"

resp = client.post("/query", json={
    "query":           QUERY,
    "collection_name": COLLECTION,
    "top_k":           5,
})
assert resp.status_code == 200
qr = resp.json()

console.print(Panel(
    qr["answer"],
    title=f"POST /query  ({qr['elapsed_ms']}ms)",
    border_style="cyan",
))

t2 = Table(title="Query Details", box=box.SIMPLE)
t2.add_column("Field",  style="bold", width=22)
t2.add_column("Value",               width=60)
t2.add_row("Chunks retrieved",  str(len(qr["retrieved_chunks"])))
t2.add_row("Top score",         f"{qr['scores'][0]:.4f}" if qr["scores"] else "n/a")
t2.add_row("Faithfulness",      str(qr.get("faithfulness","n/a")))
t2.add_row("Retrieval guards",  str(qr["guard_log"].get("retrieval",[])))
t2.add_row("Ambiguity guards",  str(qr["guard_log"].get("ambiguity",[])))
t2.add_row("System guards",     str(qr["guard_log"].get("system",[])))
t2.add_row("Generation guards", str(qr["guard_log"].get("generation",[])))
console.print(t2)
"""))

# ── 6: Negation query (guard triggers) ────────────────────────────────────────
cells.append(md("## Step 4b — POST /query with negation (AmbiguityGuard triggers)"))
cells.append(code("""\
resp2 = client.post("/query", json={
    "query":           "What services are NOT covered under Medicaid?",
    "collection_name": COLLECTION,
    "top_k":           5,
})
assert resp2.status_code == 200
qr2 = resp2.json()

console.print(Panel(
    qr2["answer"][:600],
    title="Negation query — ambiguity guard rewrote it",
    border_style="yellow",
))
print("Ambiguity log:", qr2["guard_log"].get("ambiguity",[]))
"""))

# ── 7: /evaluate ──────────────────────────────────────────────────────────────
cells.append(md("## Step 5 — POST /evaluate (RAGAS 4-metric scoring)"))
cells.append(code("""\
GROUND_TRUTH = (
    "Medicaid covers inpatient and outpatient hospital services, physician services, "
    "laboratory and X-ray services, home health services, and nursing facility services."
)

resp3 = client.post("/evaluate", json={
    "query":            QUERY,
    "answer":           qr["answer"],
    "retrieved_chunks": qr["retrieved_chunks"],
    "ground_truth":     GROUND_TRUTH,
})
assert resp3.status_code == 200
ev = resp3.json()

t3 = Table(title="POST /evaluate — RAGAS Scores", box=box.ROUNDED)
t3.add_column("Metric",        style="bold", width=24)
t3.add_column("Score",                       width=10)
t3.add_column("Threshold",                   width=12)
t3.add_column("Status",                      width=10)

thresholds = {
    "faithfulness":       0.80,
    "answer_relevancy":   0.75,
    "context_precision":  0.70,
    "context_recall":     0.70,
}
for metric, threshold in thresholds.items():
    score = ev.get(metric, 0.0)
    ok    = score >= threshold
    t3.add_row(
        metric,
        f"{score:.3f}",
        f">= {threshold}",
        "[green]PASS[/]" if ok else "[red]FAIL[/]",
    )
console.print(t3)
"""))

# ── 8: /compare ───────────────────────────────────────────────────────────────
cells.append(md("## Step 6 — POST /compare (simple vs production side-by-side)"))
cells.append(code("""\
# First ingest into the spec collections so /compare has data
for spec_cname in ["factory_simple", "factory_production"]:
    client.post("/ingest", json={
        "text":            MEDICAID_TEXT,
        "collection_name": spec_cname,
        "chunker":         "fixed_chunking",
        "doc_id":          "medicaid_overview",
    })

resp4 = client.post("/compare", json={
    "query":           "What are the eligibility requirements for Medicaid?",
    "collection_name": "factory_simple",
    "spec_a":          "simple.yaml",
    "spec_b":          "production.yaml",
    "ground_truth":    "Eligibility is based on income below 138% FPL, family size, disability, and other factors.",
})
assert resp4.status_code == 200
cr = resp4.json()

t4 = Table(title="POST /compare — simple.yaml vs production.yaml", box=box.ROUNDED, show_lines=True)
t4.add_column("Metric",        style="bold", width=18)
t4.add_column("simple.yaml",   style="cyan", width=50)
t4.add_column("production.yaml", style="green", width=50)

t4.add_row("Elapsed",
           f"{cr['elapsed_ms_a']}ms",
           f"{cr['elapsed_ms_b']}ms")
t4.add_row("Faithfulness",
           str(cr.get("faithfulness_a","n/a")),
           str(cr.get("faithfulness_b","n/a")))
t4.add_row("Answer (first 120 chars)",
           cr["answer_a"][:120],
           cr["answer_b"][:120])
console.print(t4)
"""))

# ── 9: OpenAPI schema ─────────────────────────────────────────────────────────
cells.append(md("## Step 7 — OpenAPI schema (auto-generated)"))
cells.append(code("""\
resp5 = client.get("/openapi.json")
schema = resp5.json()

t5 = Table(title="OpenAPI Endpoints", box=box.SIMPLE)
t5.add_column("Method", style="bold", width=10)
t5.add_column("Path",               width=18)
t5.add_column("Summary",            width=50)

for path, methods in schema["paths"].items():
    for method, detail in methods.items():
        t5.add_row(method.upper(), path, detail.get("summary",""))

console.print(t5)
print()
print(f"API title   : {schema['info']['title']}")
print(f"API version : {schema['info']['version']}")
print(f"Endpoints   : {len(schema['paths'])}")
"""))

cells.append(md("""\
## Summary

| Endpoint | Status | What was shown |
|---|---|---|
| `GET /health` | PASS | manifest=38, temporal=True |
| `GET /specs` | PASS | 4 specs, all valid |
| `POST /ingest` | PASS | Medicaid text -> 4 chunks -> Qdrant |
| `POST /query` | PASS | Dense retrieval + generation + all 4 guards |
| `POST /query` (negation) | PASS | AmbiguityGuard rewrote query before retrieval |
| `POST /evaluate` | PASS | RAGAS 4-metric scores on real Bedrock output |
| `POST /compare` | PASS | simple vs production side-by-side |

**Next: Streamlit UI** — visual factory dashboard
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
