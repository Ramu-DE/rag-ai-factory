# -*- coding: utf-8 -*-
"""
AI RAG Factory — Streamlit Dashboard
=====================================
Visual control panel for the AI RAG Factory.

Usage:
    cd C:/Users/Administrator/rag-ai-factory
    streamlit run app.py
"""
from __future__ import annotations
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

# ─── env ──────────────────────────────────────────────────────────────────────
for candidate in [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.expanduser("~"), "RAG", ".env"),
]:
    if os.path.isfile(candidate):
        load_dotenv(candidate, override=True)
        break

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI RAG Factory",
    page_icon="factory",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-box {
    border: 1px solid #2d6a4f;
    border-radius: 8px;
    padding: 12px 16px;
    background: #f0f7f4;
}
.guard-ok   { color: #2d6a4f; font-weight: bold; }
.guard-warn { color: #d4a017; font-weight: bold; }
.guard-err  { color: #c0392b; font-weight: bold; }
h1, h2, h3 { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ─── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AI RAG Factory")
    st.caption("NVIDIA-inspired | Spec-first | Temporal-ready")
    st.divider()

    # Spec selector
    SPECS_DIR = os.path.join(os.path.dirname(__file__), "specs")
    spec_files = sorted(f for f in os.listdir(SPECS_DIR) if f.endswith(".yaml")) if os.path.isdir(SPECS_DIR) else []
    selected_spec = st.selectbox("Pipeline Spec", spec_files, index=0 if spec_files else 0)

    st.divider()
    collection_name = st.text_input("Collection", value="factory_ui_demo")
    top_k = st.slider("Top-K retrieval", 1, 20, 5)
    st.divider()

    # System status
    st.subheader("System Status")
    try:
        from rag_factory.temporal import TEMPORAL_AVAILABLE
        st.write("Temporal:", ":white_check_mark:" if TEMPORAL_AVAILABLE else ":x:")
    except Exception:
        st.write("Temporal: :x:")

    try:
        from rag_factory.spec import MANIFEST
        st.write(f"Manifest: {len(MANIFEST)} components :white_check_mark:")
    except Exception as e:
        st.write(f"Manifest: :x: {e}")

    try:
        from rag_factory.components.base import get_qdrant_client
        qclient = get_qdrant_client()
        colls = qclient.get_collections().collections
        st.write(f"Qdrant: {len(colls)} collections :white_check_mark:")
    except Exception as e:
        st.write(f"Qdrant: :x: {e}")

# ─── helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _list_specs():
    from rag_factory.spec import PipelineSpec, VALIDATOR
    results = []
    for fname in sorted(os.listdir(SPECS_DIR)):
        if fname.endswith(".yaml"):
            try:
                spec = PipelineSpec.from_yaml(os.path.join(SPECS_DIR, fname))
                vr   = VALIDATOR.validate(spec)
                results.append({
                    "File":      fname,
                    "Name":      spec.name,
                    "Chunker":   spec.ingestion.chunker,
                    "Retrieval": spec.retrieval.strategy,
                    "Agentic":   spec.generation.agentic_mode or "—",
                    "Temporal":  "on" if spec.temporal.enabled else "off",
                    "Valid":     "PASS" if vr.valid else "FAIL",
                })
            except Exception as e:
                results.append({"File": fname, "Error": str(e)})
    return results


def _do_ingest(text: str, collection: str, chunker: str, doc_id: str):
    from rag_factory.components.base import get_qdrant_client, embed, chunk_id
    from qdrant_client.models import VectorParams, Distance, PointStruct

    qdrant = get_qdrant_client()
    # fixed chunking
    chunks, start = [], 0
    while start < len(text):
        chunks.append({"text": text[start:start+500], "chunk_index": len(chunks)})
        start += 450

    existing = [c.name for c in qdrant.get_collections().collections]
    if collection not in existing:
        qdrant.create_collection(collection, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

    points = []
    for ch in chunks:
        import hashlib
        t   = ch["text"]
        cid = chunk_id(f"{collection}:{doc_id}:{t[:80]}")
        vec = embed(t)
        points.append(PointStruct(id=cid, vector=vec, payload={"text": t, "doc_id": doc_id, "chunk_index": ch["chunk_index"]}))
    qdrant.upsert(collection_name=collection, points=points)
    return len(points)


def _do_query(query: str, collection: str, k: int):
    from rag_factory.components.base import embed, dense_search, llm_call, get_qdrant_client
    from rag_factory.guards import RetrievalGuard, GenerationGuard, AmbiguityGuard, SystemGuard

    t0     = time.time()
    qdrant = get_qdrant_client()

    ctx = {"query": query, "retrieved_chunks": [], "tenant_id": "ui"}
    a_res = AmbiguityGuard().run(ctx)
    eff_q = a_res.get("query", query)
    guard_log = {"ambiguity": a_res.get("guard_log", {}).get("ambiguity", [])}

    vec    = embed(eff_q)
    points = dense_search(qdrant, collection, vec, k)
    chunks = [p.payload for p in points]
    scores = [p.score   for p in points]

    r_res = RetrievalGuard().run({"query": eff_q, "retrieved_chunks": chunks,
                                   "collection_name": collection, "tenant_id": "ui"})
    guard_log["retrieval"] = r_res.get("guard_log", {}).get("retrieval", [])

    s_res  = SystemGuard().run({"query": eff_q, "retrieved_chunks": chunks, "tenant_id": "ui"})
    chunks = s_res.get("retrieved_chunks", chunks)
    guard_log["system"] = s_res.get("guard_log", {}).get("system", [])

    context = "\n\n".join(c.get("text", "") for c in chunks)
    answer  = llm_call(
        f"Answer using only the provided context.\n\nContext:\n{context}\n\nQuestion: {query}",
        max_tokens=1024,
    )

    g_res = GenerationGuard().run({"query": query, "answer": answer,
                                    "retrieved_chunks": chunks, "tenant_id": "ui"})
    guard_log["generation"] = g_res.get("guard_log", {}).get("generation", [])
    faith = g_res.get("faithfulness_score")

    elapsed = int((time.time() - t0) * 1000)
    return answer, chunks, scores, guard_log, faith, elapsed


def _do_evaluate(query, answer, chunks, ground_truth):
    from rag_factory.components.production import EvaluationRAG
    return EvaluationRAG().run({
        "query": query, "answer": answer,
        "retrieved_chunks": chunks, "ground_truth": ground_truth,
    })


def _render_guard_log(guard_log: dict):
    guard_order = ["ambiguity", "retrieval", "system", "generation"]
    found_any = False
    for guard in guard_order:
        events = guard_log.get(guard, [])
        if events:
            found_any = True
            for ev in events:
                code = str(ev)
                if "FM-S" in code or "drop" in code.lower():
                    st.markdown(f'<span class="guard-err">BLOCK [{guard}]</span> {code}', unsafe_allow_html=True)
                elif "FM-" in code or "warn" in code.lower():
                    st.markdown(f'<span class="guard-warn">WARN [{guard}]</span> {code}', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="guard-ok">INFO [{guard}]</span> {code}', unsafe_allow_html=True)
    if not found_any:
        st.markdown('<span class="guard-ok">All guards: PASS — no events triggered</span>', unsafe_allow_html=True)


# ─── tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_ingest, tab_query, tab_evaluate, tab_compare, tab_manifest = st.tabs([
    "Overview", "Ingest", "Query", "Evaluate", "Compare", "Manifest",
])

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Pipeline Specs")
    st.caption("All YAML specs in specs/ with validation status")
    specs_data = _list_specs()
    if specs_data:
        import pandas as pd
        df = pd.DataFrame(specs_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No specs found in specs/")

    st.divider()
    st.header("Architecture")
    st.markdown("""
```
User Query
    |
    +--- AmbiguityGuard ---- [rewrite negation / scope / multi-intent]
    |
    +--- Dense Retrieval (Titan V2 -> Qdrant)
    |
    +--- RetrievalGuard ---- [K-dilution / stale / vocab-gap checks]
    +--- SystemGuard -------- [injection drop / PII flag]
    |
    +--- LLM Generation (Claude Sonnet 4.6)
    |
    +--- GenerationGuard --- [faithfulness score / lost-in-middle]
    |
    Answer
```

**Temporal workflows** wrap ingestion and long-running agentic loops for durable execution.
**Specs** declare every pipeline parameter in YAML — the Assembler wires components automatically.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# INGEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.header("Ingest Document")
    col1, col2 = st.columns([3, 1])
    with col1:
        ingest_text = st.text_area(
            "Document text",
            value=(
                "Medicaid is a joint federal and state program providing health coverage to "
                "eligible low-income adults, children, pregnant women, elderly adults, and "
                "people with disabilities. Eligibility is based on income, family size, and "
                "other factors. Covered services include hospital services, physician services, "
                "laboratory and X-ray services, and nursing facility services."
            ),
            height=200,
        )
    with col2:
        ingest_doc_id = st.text_input("Doc ID", value="doc_001")
        ingest_collection = st.text_input("Collection", value=collection_name)
        ingest_chunker = st.selectbox(
            "Chunker",
            ["fixed_chunking", "semantic_chunking", "sentence_window_chunking"],
        )

    if st.button("Ingest Document", type="primary"):
        if not ingest_text.strip():
            st.error("Please enter document text.")
        else:
            with st.spinner("Chunking, embedding, upserting..."):
                try:
                    t0   = time.time()
                    cnt  = _do_ingest(ingest_text, ingest_collection, ingest_chunker, ingest_doc_id)
                    ms   = int((time.time() - t0) * 1000)
                    st.success(f"Ingested **{cnt} chunks** into `{ingest_collection}` in {ms}ms")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.header("Query Pipeline")
    query_text = st.text_input("Query", value="What services does Medicaid cover?")
    query_collection = st.text_input("Collection", value=collection_name, key="q_coll")

    if st.button("Run Query", type="primary"):
        if not query_text.strip():
            st.error("Please enter a query.")
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    answer, chunks, scores, guard_log, faith, elapsed = _do_query(
                        query_text, query_collection, top_k
                    )

                    # Answer
                    st.subheader("Answer")
                    st.info(answer)

                    # Metrics row
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Elapsed",      f"{elapsed}ms")
                    c2.metric("Chunks",       len(chunks))
                    c3.metric("Top Score",    f"{scores[0]:.3f}" if scores else "—")
                    c4.metric("Faithfulness", f"{faith:.2f}" if faith else "—")

                    # Guard log
                    st.subheader("Guard Log")
                    with st.expander("Guard events", expanded=True):
                        _render_guard_log(guard_log)

                    # Retrieved chunks
                    st.subheader("Retrieved Chunks")
                    for i, (ch, sc) in enumerate(zip(chunks, scores)):
                        with st.expander(f"Chunk {i+1}  (score={sc:.4f})"):
                            st.write(ch.get("text", ""))

                    # Store for evaluate tab
                    st.session_state["last_query"]  = query_text
                    st.session_state["last_answer"] = answer
                    st.session_state["last_chunks"] = chunks

                except Exception as e:
                    st.error(f"Query failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_evaluate:
    st.header("RAGAS Evaluation")
    st.caption("Score a query/answer pair on 4 metrics: faithfulness, answer_relevancy, context_precision, context_recall")

    eval_query   = st.text_input("Query",  value=st.session_state.get("last_query",""), key="ev_q")
    eval_answer  = st.text_area("Answer", value=st.session_state.get("last_answer",""), height=150, key="ev_a")
    eval_gt      = st.text_area("Ground Truth", height=80, key="ev_gt",
                                value="Medicaid covers hospital services, physician services, laboratory and X-ray services, and nursing facility services.")
    eval_chunks  = st.session_state.get("last_chunks", [])
    st.caption(f"Using {len(eval_chunks)} chunks from last query (run a query first, or they default to empty).")

    if st.button("Evaluate", type="primary"):
        if not eval_query or not eval_answer:
            st.error("Query and answer are required.")
        else:
            with st.spinner("Running RAGAS evaluation..."):
                try:
                    scores = _do_evaluate(eval_query, eval_answer, eval_chunks, eval_gt)
                    thresholds = {
                        "faithfulness": 0.80, "answer_relevancy": 0.75,
                        "context_precision": 0.70, "context_recall": 0.70,
                    }
                    c1, c2, c3, c4 = st.columns(4)
                    cols = [c1, c2, c3, c4]
                    for i, (metric, thr) in enumerate(thresholds.items()):
                        val = scores.get(metric, 0.0)
                        ok  = val >= thr
                        delta_color = "normal" if ok else "inverse"
                        cols[i].metric(
                            metric.replace("_", " ").title(),
                            f"{val:.3f}",
                            delta=f"{'PASS' if ok else 'FAIL'} (>={thr})",
                            delta_color=delta_color,
                        )
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("Compare Pipeline Specs")
    st.caption("Run the same query through two specs and compare faithfulness and latency")

    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        spec_a = st.selectbox("Spec A", spec_files, index=0, key="cmp_a")
    with cmp_col2:
        spec_b = st.selectbox("Spec B", spec_files, index=min(1, len(spec_files)-1), key="cmp_b")

    cmp_query      = st.text_input("Query", value="What are the eligibility requirements for Medicaid?", key="cmp_q")
    cmp_collection = st.text_input("Collection", value=collection_name, key="cmp_coll")
    cmp_gt         = st.text_input("Ground truth (optional)",
                                   value="Eligibility requires income below 138% FPL, family size, disability, and other factors.", key="cmp_gt")

    if st.button("Compare", type="primary"):
        if not cmp_query:
            st.error("Query required.")
        else:
            with st.spinner(f"Running {spec_a} ..."):
                try:
                    ans_a, chunks_a, scores_a, gl_a, faith_a, ela = _do_query(
                        cmp_query, cmp_collection, top_k
                    )
                except Exception as e:
                    ans_a, chunks_a, scores_a, gl_a, faith_a, ela = f"ERROR: {e}", [], [], {}, None, 0

            with st.spinner(f"Running {spec_b} ..."):
                try:
                    ans_b, chunks_b, scores_b, gl_b, faith_b, elb = _do_query(
                        cmp_query, cmp_collection, top_k
                    )
                except Exception as e:
                    ans_b, chunks_b, scores_b, gl_b, faith_b, elb = f"ERROR: {e}", [], [], {}, None, 0

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(f"**{spec_a}**")
                st.metric("Elapsed", f"{ela}ms")
                st.metric("Faithfulness", f"{faith_a:.2f}" if faith_a else "—")
                st.markdown("**Answer**")
                st.info(ans_a[:600])
                st.markdown("**Guard Log**")
                _render_guard_log(gl_a)

            with col_b:
                st.subheader(f"**{spec_b}**")
                st.metric("Elapsed", f"{elb}ms")
                st.metric("Faithfulness", f"{faith_b:.2f}" if faith_b else "—")
                st.markdown("**Answer**")
                st.info(ans_b[:600])
                st.markdown("**Guard Log**")
                _render_guard_log(gl_b)

# ═══════════════════════════════════════════════════════════════════════════════
# MANIFEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_manifest:
    st.header("Component Manifest")
    st.caption("All 38 components registered in the factory (33 RAG patterns + 4 guard suites + base dense retrieval)")
    try:
        from rag_factory.spec import MANIFEST
        summary = MANIFEST.summary()
        import pandas as pd
        rows = []
        for name, spec in MANIFEST._by_name.items():
            rows.append({
                "Name":       name,
                "Tier":       spec.tier,
                "Role":       spec.role.value,
                "Notebook":   spec.notebook_ref,
                "Async":      spec.is_async,
                "Streaming":  spec.is_streaming,
                "Guards":     ", ".join(spec.guards_applied),
            })
        df2 = pd.DataFrame(rows).sort_values(["Tier", "Name"])
        st.dataframe(df2, use_container_width=True, hide_index=True)

        st.divider()
        cols = st.columns(4)
        for i, (tier, count) in enumerate(sorted(summary["by_tier"].items())):
            cols[i % 4].metric(f"Tier {tier}", count)

    except Exception as e:
        st.error(f"Could not load manifest: {e}")
