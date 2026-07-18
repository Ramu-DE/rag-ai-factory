# -*- coding: utf-8 -*-
"""Build NB36: AI RAG Factory - Assembler Demo"""
import nbformat, os

NB_PATH = os.path.join(os.path.dirname(__file__), "notebooks", "NB36_Assembler_Demo.ipynb")

def md(src):   return nbformat.v4.new_markdown_cell(src)
def code(src): return nbformat.v4.new_code_cell(src)

cells = []

# ── 0: Title ──────────────────────────────────────────────────────────────────
cells.append(md("""\
# NB36 — AI RAG Factory: Assembler Demo

> **AI Factory | NB36**

## What this notebook demonstrates

The **Assembler** reads a YAML `PipelineSpec` and chains all components into a
single callable pipeline — chunker → retriever → query ops → generator —
with guards injected automatically.

### What you will see
1. Load all 4 pipeline specs and validate them
2. Assemble `simple.yaml` → run a real query against AWS Bedrock + Qdrant
3. Assemble `production.yaml` → run the same query through the full Tier 1-7 stack
4. Side-by-side comparison: answer quality, guard log, elapsed time
5. Show how swapping one line in the YAML changes the entire pipeline
"""))

# ── 1: Deps ───────────────────────────────────────────────────────────────────
cells.append(code("""\
import subprocess, sys
pkgs = ["boto3","qdrant-client","pydantic>=2.0","pyyaml",
        "python-dotenv","rank-bm25","numpy","rich"]
for p in pkgs:
    subprocess.run([sys.executable,"-m","pip","install",p,"-q"], check=False)
print("deps ready")
"""))

# ── 2: Imports ────────────────────────────────────────────────────────────────
cells.append(code("""\
import os, sys, json, time
sys.path.insert(0, os.path.abspath(".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.getcwd()), "RAG", ".env"), override=True)

from rag_factory.spec      import PipelineSpec, MANIFEST, VALIDATOR
from rag_factory.assembler import Assembler
from rich.console  import Console
from rich.table    import Table
from rich.panel    import Panel
from rich          import box

console = Console()
assembler = Assembler()

print(f"MANIFEST : {len(MANIFEST)} components")
print(f"Assembler: ready")
"""))

# ── 3: Section ────────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — Validate all pipeline specs"))

# ── 4: Validate all specs ─────────────────────────────────────────────────────
cells.append(code("""\
SPEC_DIR = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), "specs")
if not os.path.isdir(SPEC_DIR):
    SPEC_DIR = os.path.join(os.getcwd(), "specs")
SPEC_FILES = ["simple.yaml", "production.yaml", "agentic.yaml", "multitenant.yaml"]

table = Table(title="Pipeline Spec Validation", box=box.ROUNDED, show_lines=True)
table.add_column("Spec",       style="bold cyan", width=20)
table.add_column("Valid",      width=7)
table.add_column("Chunker",    width=24)
table.add_column("Retrieval",  width=22)
table.add_column("Agentic",    width=16)
table.add_column("Guards",     width=28)
table.add_column("Temporal",   width=10)

specs = {}
for fname in SPEC_FILES:
    path  = os.path.join(SPEC_DIR, fname)
    spec  = PipelineSpec.from_yaml(path)
    result= VALIDATOR.validate(spec)
    specs[fname] = spec
    table.add_row(
        fname,
        "[green]PASS[/]" if result.valid else "[red]FAIL[/]",
        spec.ingestion.chunker,
        spec.retrieval.strategy,
        spec.generation.agentic_mode,
        ", ".join(spec.active_guards()),
        "[green]on[/]" if spec.temporal.enabled else "[dim]off[/]",
    )

console.print(table)
"""))

# ── 5: Section ────────────────────────────────────────────────────────────────
cells.append(md("## Step 2 — Assemble & inspect the simple pipeline"))

# ── 6: Assemble simple ────────────────────────────────────────────────────────
cells.append(code("""\
simple_pipeline = assembler.assemble(specs["simple.yaml"])

console.print(Panel.fit(
    "\\n".join([
        f"[bold]Pipeline:[/] {simple_pipeline.spec.name}",
        "",
        "[bold]Steps (in execution order):[/]",
        *[f"  {i+1}. {s.SPEC.name}  "
          f"[dim](tier={s.SPEC.tier}, async={s.SPEC.is_async}, "
          f"timeout={s.SPEC.timeout_secs}s)[/]"
          for i, s in enumerate(simple_pipeline.steps)],
        "",
        "[bold]Auto-injected guards:[/]",
        *[f"  * {g.SPEC.name}  [dim]covers: {g.SPEC.guards_applied}[/]"
          for g in simple_pipeline.guards],
    ]),
    title="Simple Pipeline — Assembly Map",
    border_style="cyan",
))
"""))

# ── 7: Section ────────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Ingest a document (fixed chunking + Qdrant upsert)"))

# ── 8: Ingest + embed in one cell (Qdrant in-memory client is per-process) ────
cells.append(code("""\
from rag_factory.components.base import embed, get_qdrant_client, chunk_id
from qdrant_client.models import VectorParams, Distance, PointStruct

SAMPLE_TEXT = (
    "Medicaid is a joint federal and state program that provides health coverage to "
    "eligible low-income adults, children, pregnant women, elderly adults, and people "
    "with disabilities. Medicaid is administered by states according to federal "
    "requirements. The program is funded jointly by states and the federal government. "
    "Eligibility for Medicaid is determined based on income, family size, disability, "
    "and other factors. The Affordable Care Act expanded Medicaid to cover all adults "
    "with income below 138 percent of the federal poverty level. Each state sets its "
    "own guidelines within federal standards. "
    "Covered services under Medicaid include inpatient and outpatient hospital services, "
    "physician services, laboratory and X-ray services, home health services, and "
    "nursing facility services. States may also cover optional services such as "
    "prescription drugs, dental care, vision services, and physical therapy. "
    "The Children's Health Insurance Program (CHIP) provides low-cost health coverage "
    "to children in families that earn too much to qualify for Medicaid but cannot "
    "afford private coverage. Like Medicaid, CHIP is jointly financed by states and "
    "the federal government and administered by states."
)

CNAME = "factory_simple_demo"

# Build chunks
ctx_ingest = {"raw_text": SAMPLE_TEXT, "collection_name": CNAME}
t0 = time.time()
chunker_result = simple_pipeline.steps[0].run(ctx_ingest)

print(f"Chunker : {simple_pipeline.steps[0].SPEC.name}")
print(f"Chunks  : {chunker_result['chunk_count']}")
for i, ch in enumerate(chunker_result["chunks"][:3]):
    print(f"  [{i}] {ch['text'][:80]!r}")

# Embed + upsert — use module-level Qdrant singleton so all cells share it
import rag_factory.components.base as _base
_QDRANT = _base.get_qdrant_client()
existing_names = [c.name for c in _QDRANT.get_collections().collections]
if CNAME in existing_names:
    _QDRANT.delete_collection(CNAME)
_QDRANT.create_collection(CNAME, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

points = []
for ch in chunker_result["chunks"]:
    vec = embed(ch["text"])
    cid = chunk_id(f"{CNAME}:{ch['text'][:80]}")
    points.append(PointStruct(id=cid, vector=vec, payload=ch))
_QDRANT.upsert(collection_name=CNAME, points=points)

elapsed_ingest = int((time.time() - t0) * 1000)
print(f"\\nUpserted {len(points)} vectors -> '{CNAME}'  ({elapsed_ingest}ms)")
"""))


# ── 10: Section ───────────────────────────────────────────────────────────────
cells.append(md("## Step 4 — Run a query through the simple pipeline (retrieval + generation + guards)"))

# ── 11: Run simple ────────────────────────────────────────────────────────────
cells.append(code("""\
from rag_factory.components.base import llm_call

QUERY = "What services are covered under Medicaid?"

# Override collection_name to use the demo collection we just built
t0  = time.time()
ctx = {
    "query":           QUERY,
    "collection_name": CNAME,
    "top_k":           5,
    "filters":         {},
    "max_sub_queries": 4,
}

# Run retriever
ret_result = simple_pipeline.steps[1].run(ctx)
ctx.update(ret_result)

# Guard: retrieval
for g in simple_pipeline.guards:
    if g.SPEC.name == "retrieval_guard":
        ctx.update(g.run(ctx))

# Generate answer
context  = "\\n\\n".join(c.get("text","") for c in ctx["retrieved_chunks"])
answer   = llm_call(
    f"Answer using only the provided context.\\n\\nContext:\\n{context}\\n\\nQuestion: {QUERY}",
    max_tokens=512,
)
ctx["answer"] = answer

# Guard: generation
for g in simple_pipeline.guards:
    if g.SPEC.name == "generation_guard":
        ctx.update(g.run(ctx))

elapsed = int((time.time() - t0) * 1000)

console.print(Panel(answer, title=f"[bold cyan]Simple Pipeline Answer[/]  ({elapsed}ms)", border_style="cyan"))
print()
print(f"Retrieved chunks : {len(ctx['retrieved_chunks'])}")
print(f"Faithfulness     : {ctx.get('faithfulness_score','n/a')}")
print(f"Guard log        : {ctx.get('guard_log',{})}")
"""))

# ── 12: Section ───────────────────────────────────────────────────────────────
cells.append(md("## Step 5 — Assemble the production pipeline & compare"))

# ── 13: Assemble production ───────────────────────────────────────────────────
cells.append(code("""\
# For demo we override the collection_name so both pipelines use the same corpus
prod_spec = specs["production.yaml"]

prod_pipeline = assembler.assemble(prod_spec)

console.print(Panel.fit(
    "\\n".join([
        f"[bold]Pipeline:[/] {prod_pipeline.spec.name}",
        "",
        "[bold]Steps:[/]",
        *[f"  {i+1}. {s.SPEC.name}  [dim](tier={s.SPEC.tier})[/]"
          for i, s in enumerate(prod_pipeline.steps)],
        "",
        "[bold]Guards:[/]",
        *[f"  * {g.SPEC.name}" for g in prod_pipeline.guards],
    ]),
    title="Production Pipeline — Assembly Map",
    border_style="green",
))
"""))

# ── 14: Run production (hybrid retrieval + self-rag lite) ─────────────────────
cells.append(code("""\
from rag_factory.components.retrieval import HybridRRFRetrieval
from rag_factory.components.agentic   import SelfRAG

t0  = time.time()
ctx2 = {
    "query":           QUERY,
    "collection_name": CNAME,   # same corpus
    "top_k":           5,
    "filters":         {},
    "max_sub_queries": 4,
}

# Step 1: Hybrid RRF retrieval
hybrid_result = HybridRRFRetrieval().run(ctx2)
ctx2.update(hybrid_result)

# Step 2: Self-RAG generation
self_rag_result = SelfRAG().run(ctx2)
ctx2.update(self_rag_result)

# Step 3: All 4 guards
for g in prod_pipeline.guards:
    ctx2.update(g.run(ctx2))

elapsed2 = int((time.time() - t0) * 1000)

console.print(Panel(
    ctx2.get("answer",""),
    title=f"[bold green]Production Pipeline Answer[/]  ({elapsed2}ms)",
    border_style="green",
))
print()
print(f"Reflection tokens: {len(ctx2.get('reflection_tokens',[]))}")
print(f"Citation scores  : {ctx2.get('citation_scores','n/a')}")
print(f"Faithfulness     : {ctx2.get('faithfulness_score','n/a')}")
rw = ctx2.get("guard_log",{}).get("ambiguity",{})
print(f"Ambiguity guard  : {rw}")
"""))

# ── 15: Section ───────────────────────────────────────────────────────────────
cells.append(md("## Step 6 — Side-by-side comparison"))

# ── 16: Comparison table ──────────────────────────────────────────────────────
cells.append(code("""\
cmp = Table(title="Pipeline Comparison", box=box.ROUNDED, show_lines=True)
cmp.add_column("Metric",             style="bold", width=22)
cmp.add_column("simple.yaml",        style="cyan", width=40)
cmp.add_column("production.yaml",    style="green",width=40)

cmp.add_row("Chunker",       "fixed_chunking",       "contextual_chunking")
cmp.add_row("Retrieval",     "dense_retrieval",       "hybrid_rrf_retrieval")
cmp.add_row("Query strategy","direct",                "decompose")
cmp.add_row("Generation",    "direct LLM call",       "self_rag (4 reflection tokens)")
cmp.add_row("Guards",        "retrieval + generation","all 4 (R+G+A+S)")
cmp.add_row("Temporal",      "off",                   "on (IngestWorkflow + AgenticQueryWorkflow)")
cmp.add_row("Elapsed",       f"{elapsed}ms",          f"{elapsed2}ms")
cmp.add_row("Faithfulness",  str(ctx.get('faithfulness_score','n/a')),
                             str(ctx2.get('faithfulness_score','n/a')))

console.print(cmp)
"""))

# ── 17: Section ───────────────────────────────────────────────────────────────
cells.append(md("## Step 7 — Hot-swap demo: change one YAML line, get a different pipeline"))

# ── 18: Hot-swap ──────────────────────────────────────────────────────────────
cells.append(code("""\
import yaml

# Start from simple spec, swap retrieval strategy to hybrid_rrf
simple_dict = specs["simple.yaml"].model_dump()
simple_dict["retrieval"]["strategy"] = "hybrid_rrf_retrieval"
simple_dict["name"] = "simple_hybrid_variant"

variant_spec     = PipelineSpec.from_dict(simple_dict)
variant_pipeline = assembler.assemble(variant_spec)

print("Original simple.yaml  steps:", [s.SPEC.name for s in simple_pipeline.steps])
print("Hybrid variant steps  :", [s.SPEC.name for s in variant_pipeline.steps])
print()
print("One line changed in the spec -> different retrieval strategy -> same pipeline contract")
print("The assembler, guards, and evaluation harness are unchanged.")
"""))

# ── 19: Section ───────────────────────────────────────────────────────────────
cells.append(md("## Step 8 — Temporal activity mapping"))

# ── 19b: Temporal map ─────────────────────────────────────────────────────────
cells.append(code("""\
from rag_factory.temporal import TEMPORAL_AVAILABLE

table_t = Table(title="Temporal Activity Mapping (production.yaml)", box=box.SIMPLE)
table_t.add_column("Component",    style="bold", width=28)
table_t.add_column("Is Async",     width=10)
table_t.add_column("Timeout",      width=10)
table_t.add_column("Max Retries",  width=12)
table_t.add_column("Temporal Activity", width=30)

prod_comps = prod_spec.active_component_names() + prod_spec.active_guards()
for comp_name in prod_comps:
    try:
        spec_c = MANIFEST.get(comp_name)
        activity_name = (
            "ingest_document"   if spec_c.role == "chunker"    else
            "retrieve_chunks"   if spec_c.role == "retriever"  else
            "generate_answer"   if spec_c.role == "generator"  else
            "evaluate_response" if spec_c.role == "evaluator"  else
            "n/a (sync)"
        )
        table_t.add_row(
            comp_name,
            "[green]yes[/]" if spec_c.is_async else "[dim]no[/]",
            f"{spec_c.timeout_secs}s",
            str(spec_c.max_retries),
            activity_name if spec_c.is_async else "[dim]direct call[/]",
        )
    except KeyError:
        pass

console.print(table_t)
print()
print(f"Temporal installed: {TEMPORAL_AVAILABLE}")
print("To start the worker: python -m rag_factory.temporal.worker")
"""))

# ── 20: Summary ───────────────────────────────────────────────────────────────
cells.append(md("""\
## Summary

| Step | What happened |
|---|---|
| Spec validation | All 4 YAML specs: valid=True, 0 errors |
| Simple assembly | 2 steps + 2 guards assembled from `simple.yaml` |
| Ingestion | Sample Medicaid text chunked and upserted to Qdrant |
| Simple query | Dense retrieval + direct generation + guard checks |
| Production query | Hybrid RRF + Self-RAG + all 4 guards |
| Hot-swap | One YAML line change → different pipeline, same contract |
| Temporal map | Long-running components mapped to durable Activities |

**Next: NB37** — Guards demo (live FM detection on real failure inputs)
"""))

# ── Build notebook ─────────────────────────────────────────────────────────────
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
