# -*- coding: utf-8 -*-
"""
AI RAG Factory — Streamlit Dashboard
=====================================
Usage:  streamlit run app.py
"""
from __future__ import annotations
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

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
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.route-card {
    background: #eaf4ee;
    border-left: 4px solid #2d6a4f;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: monospace;
}
.route-card.complex  { border-color: #1e3a5f; background: #eaf0f7; }
.route-card.agentic  { border-color: #7b2d8b; background: #f5eaf7; }
.route-card.mt       { border-color: #b8660e; background: #fdf3e7; }
.inc-badge           { background:#2d6a4f; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
.full-badge          { background:#1e3a5f; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
.guard-ok   { color:#2d6a4f; font-weight:bold; }
.guard-warn { color:#d4a017; font-weight:bold; }
.guard-err  { color:#c0392b; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ─── sidebar ──────────────────────────────────────────────────────────────────
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".index")

with st.sidebar:
    st.title("AI RAG Factory")
    st.caption("Spec-first | Incremental PDF | Smart routing")
    st.divider()

    SPECS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")
    spec_files = sorted(f for f in os.listdir(SPECS_DIR) if f.endswith(".yaml")) if os.path.isdir(SPECS_DIR) else []

    manual_spec = st.selectbox("Override spec (optional)", ["auto-route"] + spec_files)
    top_k       = st.slider("Top-K retrieval", 1, 20, 5)
    tenant_id   = st.text_input("Tenant ID (optional)", value="", placeholder="leave blank for shared")

    st.divider()
    st.subheader("System")
    try:
        from rag_factory.temporal import TEMPORAL_AVAILABLE
        st.write("Temporal :", ":white_check_mark:" if TEMPORAL_AVAILABLE else ":x: not installed")
    except Exception:
        st.write("Temporal : :x:")

    try:
        from rag_factory.spec import MANIFEST
        st.write(f"Manifest : {len(MANIFEST)} components :white_check_mark:")
    except Exception as e:
        st.write(f"Manifest : :x: {e}")

    try:
        from rag_factory.components.base import get_qdrant_client
        _qc = get_qdrant_client()
        _n  = len(_qc.get_collections().collections)
        st.write(f"Qdrant   : {_n} collections :white_check_mark:")
    except Exception as e:
        st.write(f"Qdrant   : :x: {e}")

    try:
        import fitz
        st.write("PyMuPDF  : :white_check_mark:")
    except ImportError:
        try:
            import pdfminer
            st.write("pdfminer : :white_check_mark: (PyMuPDF preferred)")
        except ImportError:
            st.write("PDF lib  : :x: install pymupdf")

# ─── helpers ──────────────────────────────────────────────────────────────────
def _render_routing(result) -> None:
    cat_class = {"SIMPLE": "", "COMPLEX": "complex", "AGENTIC": "agentic", "MULTITENANT": "mt"}
    cls = cat_class.get(result.category, "")
    badge = "heuristic (no LLM)" if result.heuristic else "LLM classified"
    st.markdown(
        f'<div class="route-card {cls}">'
        f'<b>Spec selected :</b> <code>{result.spec}</code> &nbsp;|&nbsp; '
        f'<b>Category :</b> {result.category} &nbsp;|&nbsp; '
        f'<b>Confidence :</b> {result.confidence:.0%} &nbsp;|&nbsp; '
        f'<b>Method :</b> {badge}<br>'
        f'<b>Reason :</b> {result.reason}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_guard_log(guard_log: dict) -> None:
    found = False
    for guard in ["ambiguity", "retrieval", "system", "generation"]:
        for ev in guard_log.get(guard, []):
            found = True
            code  = str(ev)
            if "FM-S" in code or "drop" in code.lower():
                st.markdown(f'<span class="guard-err">BLOCK [{guard}]</span> {code}', unsafe_allow_html=True)
            elif "FM-" in code or "warn" in code.lower():
                st.markdown(f'<span class="guard-warn">WARN [{guard}]</span> {code}', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="guard-ok">INFO [{guard}]</span> {code}', unsafe_allow_html=True)
    if not found:
        st.markdown('<span class="guard-ok">All guards PASS — no events</span>', unsafe_allow_html=True)


def _do_query(query: str, collection: str, k: int):
    from rag_factory.components.base import embed, dense_search, llm_call, get_qdrant_client
    from rag_factory.guards import RetrievalGuard, GenerationGuard, AmbiguityGuard, SystemGuard

    t0 = time.time()
    qdrant = get_qdrant_client()

    ctx = {"query": query, "retrieved_chunks": [], "tenant_id": tenant_id or "default"}
    a_res = AmbiguityGuard().run(ctx)
    eff_q = a_res.get("query", query)
    guard_log: dict = {"ambiguity": a_res.get("guard_log", {}).get("ambiguity", [])}

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qfilter = None
    if tenant_id:
        qfilter = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])

    vec    = embed(eff_q)
    points = dense_search(qdrant, collection, vec, k, filters=qfilter)
    chunks = [p.payload for p in points]
    scores = [p.score   for p in points]

    r_res = RetrievalGuard().run({"query": eff_q, "retrieved_chunks": chunks,
                                   "collection_name": collection, "tenant_id": tenant_id or "default"})
    guard_log["retrieval"] = r_res.get("guard_log", {}).get("retrieval", [])

    s_res  = SystemGuard().run({"query": eff_q, "retrieved_chunks": chunks, "tenant_id": tenant_id or "default"})
    chunks = s_res.get("retrieved_chunks", chunks)
    guard_log["system"] = s_res.get("guard_log", {}).get("system", [])

    context = "\n\n".join(c.get("text", "") for c in chunks)
    answer  = llm_call(
        f"Answer using only the provided context.\n\nContext:\n{context}\n\nQuestion: {query}",
        max_tokens=1024,
    )

    g_res = GenerationGuard().run({"query": query, "answer": answer,
                                    "retrieved_chunks": chunks, "tenant_id": tenant_id or "default"})
    guard_log["generation"] = g_res.get("guard_log", {}).get("generation", [])
    faith = g_res.get("faithfulness_score")

    return answer, chunks, scores, guard_log, faith, int((time.time() - t0) * 1000)


@st.cache_data(ttl=60, show_spinner=False)
def _list_specs():
    from rag_factory.spec import PipelineSpec, VALIDATOR
    rows = []
    for fname in sorted(os.listdir(SPECS_DIR)):
        if not fname.endswith(".yaml"):
            continue
        try:
            spec = PipelineSpec.from_yaml(os.path.join(SPECS_DIR, fname))
            vr   = VALIDATOR.validate(spec)
            rows.append({
                "File":      fname,
                "Name":      spec.name,
                "Chunker":   spec.ingestion.chunker,
                "Retrieval": spec.retrieval.strategy,
                "Agentic":   spec.generation.agentic_mode or "—",
                "Temporal":  "on" if spec.temporal.enabled else "off",
                "Valid":     "PASS" if vr.valid else "FAIL",
            })
        except Exception as e:
            rows.append({"File": fname, "Error": str(e)})
    return rows


# ─── tabs ─────────────────────────────────────────────────────────────────────
tab_idp, tab_batch, tab_pdf, tab_ask, tab_eval, tab_compare, tab_overview, tab_manifest = st.tabs([
    "IDP (Smart Extract)", "Batch Upload", "PDF Upload + RAG", "Ask (Smart Route)", "Evaluate", "Compare Specs", "Overview", "Manifest",
])


# ═══════════════════════════════════════════════════════════════════════════════
# IDP — Intelligent Document Processing
# ═══════════════════════════════════════════════════════════════════════════════
with tab_idp:
    st.header("Intelligent Document Processing")
    st.caption(
        "Upload any document — invoice, contract, medical record, ID, or custom. "
        "The pipeline auto-classifies it, runs the right extraction skill, validates fields, "
        "and indexes it for Q&A — all in one call."
    )

    col_idp1, col_idp2 = st.columns([2, 1])
    with col_idp1:
        idp_file = st.file_uploader(
            "Drop any document (PDF, PNG, JPG, TIFF)",
            type=["pdf","png","jpg","jpeg","tiff","tif"],
            key="idp_upload",
            label_visibility="collapsed",
        )
    with col_idp2:
        idp_collection   = st.text_input("Collection", value="idp_documents", key="idp_coll")
        doc_type_override = st.selectbox(
            "Doc type override",
            ["auto-detect", "invoice", "contract", "medical", "id_document", "report", "form", "other"],
        )
        idp_force = st.checkbox("Force full re-index", key="idp_force")

    # Skill registry table
    with st.expander("Available document skills"):
        from rag_factory.skills.registry import list_skills
        import pandas as pd
        skill_rows = [{"Doc Type": dt, "Capability": desc} for dt, desc in list_skills().items()]
        st.dataframe(pd.DataFrame(skill_rows), use_container_width=True, hide_index=True)

    if idp_file is not None:
        st.info(f"**{idp_file.name}** — {idp_file.size / 1024:.1f} KB")

        if st.button("Process Document", type="primary", key="idp_btn"):
            with st.spinner("Classifying → extracting → validating → indexing..."):
                try:
                    import tempfile, os
                    suffix = os.path.splitext(idp_file.name)[1] or ".pdf"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(idp_file.getbuffer())
                        tmp_path = tmp.name

                    from rag_factory.idp_pipeline import IDPPipeline
                    pipeline = IDPPipeline(
                        collection_name=idp_collection,
                        index_dir=INDEX_DIR,
                        force_reindex=idp_force,
                        tenant_id=tenant_id or None,
                    )
                    result = pipeline.process(
                        file_path=tmp_path,
                        doc_type_override=None if doc_type_override == "auto-detect" else doc_type_override,
                        original_name=idp_file.name,
                    )
                    os.unlink(tmp_path)

                    # ── classification banner ─────────────────────────────
                    cat_colors = {
                        "invoice": "#2d6a4f", "contract": "#1e3a5f",
                        "medical": "#7b2d8b", "id_document": "#b8660e",
                    }
                    color = cat_colors.get(result.doc_type, "#555")
                    scan_label = "SCANNED (Textract OCR)" if result.is_scanned else "DIGITAL PDF (PyMuPDF)"
                    st.markdown(
                        f'<div style="background:{color}15;border-left:4px solid {color};'
                        f'border-radius:6px;padding:10px 14px;font-family:monospace;">'
                        f'<b>Doc type :</b> {result.doc_type.upper()} &nbsp;|&nbsp; '
                        f'<b>Confidence :</b> {result.classification.confidence:.0%} &nbsp;|&nbsp; '
                        f'<b>Method :</b> {scan_label}<br>'
                        f'<b>Reason :</b> {result.classification.reason}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── ingest metrics ────────────────────────────────────
                    st.subheader("Incremental Ingest")
                    ir = result.ingest_report
                    badge = "INCREMENTAL" if ir.incremental else "FULL INDEX"

                    # Get total chunks in collection (includes previously indexed chunks)
                    try:
                        from rag_factory.components.base import get_qdrant_client
                        _qc = get_qdrant_client()
                        _cinfo = _qc.get_collection(ir.collection_name)
                        total_chunks_in_coll = _cinfo.points_count
                    except Exception:
                        total_chunks_in_coll = ir.chunks_embedded

                    c1,c2,c3,c4,c5,c6 = st.columns(6)
                    c1.metric("Pages",           ir.pages_total)
                    c2.metric("Added",           ir.pages_added)
                    c3.metric("Updated",         ir.pages_updated)
                    c4.metric("Skipped",         ir.pages_skipped,
                              help="Unchanged pages — zero embedding cost")
                    c5.metric("New chunks",      ir.chunks_embedded,
                              help="Chunks embedded this run")
                    c6.metric("Total in index",  total_chunks_in_coll,
                              help="All chunks available for Q&A")

                    if ir.incremental and ir.chunks_embedded == 0 and ir.pages_skipped == ir.pages_total:
                        st.info(
                            f"All {ir.pages_skipped} pages unchanged since last upload — "
                            f"no re-embedding needed. {total_chunks_in_coll} chunks already in the index and ready for Q&A."
                        )
                    else:
                        st.caption(f"[{badge}] {ir.elapsed_ms}ms | collection: {ir.collection_name}")

                    # ── extracted fields ──────────────────────────────────
                    if result.extraction:
                        st.subheader("Extracted Fields")
                        fields_df = pd.DataFrame(
                            [{"Field": k, "Value": v} for k, v in result.extraction.fields.items()]
                        )
                        st.dataframe(fields_df, use_container_width=True, hide_index=True)

                        if result.extraction.tables:
                            st.subheader("Extracted Tables / Line Items")
                            for i, row in enumerate(result.extraction.tables[:20]):
                                with st.expander(f"Row {i+1}"):
                                    st.json(row)

                        conf = result.extraction.confidence
                        conf_color = "normal" if conf >= 0.75 else "inverse"
                        st.metric("Extraction confidence", f"{conf:.0%}",
                                  delta="HIGH" if conf >= 0.75 else "LOW — review recommended",
                                  delta_color=conf_color)

                    # ── validation (only for skill doc types) ─────────────
                    if result.validation:
                        st.subheader("Field Validation")
                        vr = result.validation
                        if vr.valid and not vr.warnings:
                            st.success("All required fields present and valid.")
                        else:
                            for e in vr.errors:
                                st.error(e)
                            for w in vr.warnings:
                                st.warning(w)
                    elif result.doc_type not in ("invoice","contract","medical","id_document"):
                        st.info(
                            f"Doc type **{result.doc_type}** — no structured field extraction "
                            f"(no skill required). Document is indexed and fully queryable via Ask tab."
                        )

                    # Store for Ask tab — also force-update the selectbox key so
                    # the dropdown reflects this collection immediately on tab switch
                    st.session_state["active_collection"] = idp_collection
                    st.session_state["active_pdf"]        = idp_file.name
                    st.session_state["ask_coll_select"]   = idp_collection
                    st.session_state["_ask_coll_source"]  = idp_collection

                    st.success(f"Document indexed into `{idp_collection}` — switch to **Ask** tab to query it.")

                except Exception as e:
                    st.error(f"IDP pipeline failed: {e}")
                    import traceback; st.code(traceback.format_exc())

    # ── architecture explainer ────────────────────────────────────────────────
    with st.expander("IDP Pipeline architecture"):
        st.code("""
Upload (PDF / PNG / JPG / TIFF)
    |
    +── DocumentClassifier (heuristic → LLM)
    |       invoice / contract / medical / id_document / report / form
    |
    +── Extractor (auto-selects)
    |       Digital PDF  → PyMuPDF (free, fast, structure-aware)
    |       Scanned PDF  → Textract DetectText / AnalyzeDocument
    |       Invoice PDF  → Textract AnalyzeExpense
    |       ID document  → Textract AnalyzeID
    |       Image file   → Textract always
    |
    +── Document Skill
    |       InvoiceSkill   → vendor, date, total, line_items
    |       ContractSkill  → parties, dates, clauses, obligations
    |       MedicalSkill   → patient, provider, ICD-10, medications
    |       IDCardSkill    → name, DOB, ID#, expiry
    |       CustomSkill    → YAML-defined fields (no code needed)
    |
    +── FieldValidator
    |       required fields, date format, amount format,
    |       cross-field rules, confidence threshold
    |
    +── IncrementalPDFProcessor
            per-page SHA-256 hash → skip unchanged
            delete stale Qdrant chunks for changed pages
            embed + upsert only changed pages
            → ready for Q&A via Ask tab
""", language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Batch Document Upload")
    st.caption(
        "Upload multiple files at once — each is classified, extracted, validated, "
        "and incrementally indexed. Download results as CSV or JSON."
    )

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        batch_files = st.file_uploader(
            "Drop multiple PDFs or images",
            type=["pdf","png","jpg","jpeg","tiff","tif"],
            accept_multiple_files=True,
            key="batch_upload",
            label_visibility="collapsed",
        )
    with col_b2:
        batch_collection = st.text_input("Collection", value="idp_batch", key="bcoll")
        batch_chunk_size = st.slider("Chunk size", 200, 800, 400, key="bchunk")
        batch_force      = st.checkbox("Force full re-index", key="bforce")
        batch_export_fmt = st.radio("Export format", ["CSV", "JSON"], horizontal=True)

    if batch_files:
        st.info(f"{len(batch_files)} file(s) selected: {', '.join(f.name for f in batch_files)}")

    if batch_files and st.button("Process All Files", type="primary", key="batch_btn"):
        import tempfile, os, json as _json
        import pandas as pd
        from rag_factory.idp_pipeline import IDPPipeline

        progress_bar = st.progress(0, text="Starting...")
        status_area  = st.empty()
        results_data = []
        total        = len(batch_files)
        succeeded = failed = total_pages = total_chunks = 0

        pipeline = IDPPipeline(
            collection_name=batch_collection,
            index_dir=INDEX_DIR,
            chunk_size=batch_chunk_size,
            force_reindex=batch_force,
            tenant_id=tenant_id or None,
        )

        for i, upload in enumerate(batch_files):
            progress_bar.progress((i) / total, text=f"Processing {upload.name} ({i+1}/{total})...")
            suffix = os.path.splitext(upload.name)[1] or ".pdf"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getbuffer())
                tmp_path = tmp.name

            try:
                result = pipeline.process(tmp_path, original_name=upload.name)
                ir     = result.ingest_report

                # Total chunks already in collection (includes previously indexed)
                try:
                    from rag_factory.components.base import get_qdrant_client
                    _qc2 = get_qdrant_client()
                    _ci2 = _qc2.get_collection(batch_collection)
                    total_in_index = _ci2.points_count
                except Exception:
                    total_in_index = ir.chunks_embedded

                sub_type = (result.extraction.metadata or {}).get("sub_type", "") \
                           if result.extraction else ""
                row = {
                    "file_name":             upload.name,
                    "doc_type":              result.doc_type,
                    "sub_type":              sub_type,
                    "confidence":            round(result.classification.confidence, 2),
                    "classification_reason": result.classification.reason,
                    "is_scanned":            result.is_scanned,
                    "method":                result.method,
                    "pages_total":           ir.pages_total,
                    "pages_added":           ir.pages_added,
                    "pages_updated":         ir.pages_updated,
                    "pages_skipped":         ir.pages_skipped,
                    "chunks_this_run":       ir.chunks_embedded,
                    "chunks_total_in_index": total_in_index,
                    "elapsed_ms":            result.elapsed_ms,
                    "incremental":           ir.incremental,
                    "extraction_confidence": round(result.extraction.confidence, 2) if result.extraction else None,
                    "validation_valid":      result.validation.valid if result.validation else None,
                    "validation_errors":     "; ".join(result.validation.errors) if result.validation else "",
                    "validation_warnings":   "; ".join(result.validation.warnings) if result.validation else "",
                    "extracted_fields":      _json.dumps(result.extraction.fields, ensure_ascii=False)
                                             if result.extraction else "{}",
                    "error":                 "",
                }
                # Flatten ALL extracted fields as individual columns for CSV readability
                if result.extraction and result.extraction.fields:
                    for fk, fv in result.extraction.fields.items():
                        row[f"field_{fk}"] = fv
                results_data.append(row)
                total_pages  += ir.pages_total
                total_chunks += ir.chunks_embedded
                succeeded    += 1
                inc_note = (f" | {ir.pages_skipped}/{ir.pages_total}p skipped (incremental)"
                            if ir.incremental and ir.pages_skipped > 0 else "")
                idx_note = f" | {total_in_index} total in index" if total_in_index > 0 else ""
                status_area.success(
                    f"[{i+1}/{total}] {upload.name} — {result.doc_type.upper()}"
                    + (f" [{sub_type}]" if sub_type else "")
                    + f" | {ir.pages_total}p | {ir.chunks_embedded} new chunks"
                    + inc_note + idx_note + f" | {result.elapsed_ms}ms"
                )
            except Exception as exc:
                results_data.append({
                    "file_name": upload.name, "doc_type": "error",
                    "error": str(exc),
                })
                failed += 1
                status_area.error(f"[{i+1}/{total}] {upload.name} — FAILED: {exc}")
            finally:
                os.unlink(tmp_path)

        progress_bar.progress(1.0, text="Complete")

        # ── summary metrics ───────────────────────────────────────────────
        st.divider()
        total_in_coll = 0
        try:
            from rag_factory.components.base import get_qdrant_client as _gqc
            _qcoll = _gqc().get_collection(batch_collection)
            total_in_coll = _qcoll.points_count
        except Exception:
            total_in_coll = total_chunks

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Files",           total)
        c2.metric("Succeeded",       succeeded)
        c3.metric("Failed",          failed, delta=f"-{failed}" if failed else None,
                  delta_color="inverse" if failed else "off")
        c4.metric("Total pages",     total_pages)
        c5.metric("New chunks",      total_chunks,    help="Chunks embedded this run")
        c6.metric("Total in index",  total_in_coll,   help="All chunks available for Q&A")

        # ── results table ─────────────────────────────────────────────────
        if results_data:
            df_batch = pd.DataFrame(results_data)
            st.dataframe(df_batch, use_container_width=True, hide_index=True)

            # ── export ────────────────────────────────────────────────────
            st.subheader("Export Results")
            if batch_export_fmt == "CSV":
                csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name=f"batch_results_{batch_collection}.csv",
                    mime="text/csv",
                    type="primary",
                )
            else:
                # Full JSON with extracted fields + tables
                full_records = []
                for r, upload in zip(results_data, batch_files):
                    full_records.append(r)
                json_bytes = _json.dumps(full_records, indent=2, default=str).encode("utf-8")
                st.download_button(
                    label="Download JSON",
                    data=json_bytes,
                    file_name=f"batch_results_{batch_collection}.json",
                    mime="application/json",
                    type="primary",
                )

            # ── per-file detail accordion ─────────────────────────────────
            st.subheader("Per-file Details")
            for row in results_data:
                icon      = "PASS" if not row.get("error") else "FAIL"
                sub_label = f" [{row['sub_type']}]" if row.get("sub_type") else ""
                with st.expander(f"[{icon}] {row['file_name']} — {row.get('doc_type','?').upper()}{sub_label}"):
                    if row.get("error"):
                        st.error(row["error"])
                    else:
                        col_x, col_y = st.columns(2)
                        with col_x:
                            st.write(f"**Doc type:** {row['doc_type']}" +
                                     (f"  ·  *{row['sub_type']}*" if row.get('sub_type') else ""))
                            st.write(f"**Confidence:** {row['confidence']:.0%}")
                            st.write(f"**Pages:** {row['pages_total']} total, "
                                     f"{row['pages_skipped']} skipped (incremental)")
                            st.write(f"**New chunks:** {row['chunks_this_run']}")
                            st.write(f"**Total in index:** {row.get('chunks_total_in_index', '—')}")
                            st.write(f"**Elapsed:** {row['elapsed_ms']}ms")
                        with col_y:
                            st.write(f"**Method:** {row['method']}")
                            st.write(f"**Incremental:** {row['incremental']}")
                            if row.get("validation_valid") is not None:
                                v_icon = "PASS" if row["validation_valid"] else "FAIL"
                                st.write(f"**Validation:** {v_icon}")
                            if row.get("validation_errors"):
                                st.error(row["validation_errors"])
                            if row.get("validation_warnings"):
                                st.warning(row["validation_warnings"])
                        if row.get("classification_reason"):
                            st.caption(f"Classification: {row['classification_reason']}")
                        if row.get("extracted_fields") and row["extracted_fields"] != "{}":
                            try:
                                fields_dict = _json.loads(row["extracted_fields"])
                                if fields_dict:
                                    st.write("**Extracted fields:**")
                                    st.json(fields_dict)
                            except Exception:
                                pass

        st.session_state["active_collection"] = batch_collection
        st.session_state["ask_coll_select"]  = batch_collection
        st.session_state["_ask_coll_source"] = batch_collection


# ═══════════════════════════════════════════════════════════════════════════════
# PDF UPLOAD — incremental processing
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pdf:
    st.header("Upload & Process PDF")
    st.caption("Per-page SHA-256 hashing — only changed pages are re-embedded on re-upload.")

    col_up, col_opt = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Drop a PDF here", type=["pdf"], label_visibility="collapsed")
    with col_opt:
        pdf_collection = st.text_input("Collection name", value="factory_pdf_demo", key="pdf_coll")
        chunk_size     = st.slider("Chunk size (chars)", 200, 800, 400, step=50)
        force_reindex  = st.checkbox("Force full re-index (ignore hash cache)")

    if uploaded is not None:
        st.info(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")

        # Show existing index stats if available
        from rag_factory.pdf_processor import IncrementalPDFProcessor
        proc = IncrementalPDFProcessor(index_dir=INDEX_DIR, chunk_size=chunk_size)
        stats = proc.get_index_stats(pdf_collection)
        if stats["indexed_pages"] > 0:
            st.markdown(
                f'<span class="inc-badge">INCREMENTAL</span> &nbsp; '
                f'Previous index: {stats["indexed_pages"]} pages, '
                f'{stats["total_chunks"]} chunks, last run: {stats["last_processed"]}',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="full-badge">FULL INDEX</span> &nbsp; No previous index found — will embed all pages.', unsafe_allow_html=True)

        if st.button("Process PDF", type="primary"):
            if force_reindex:
                proc.reset_index(pdf_collection)
                st.toast("Hash index cleared — full re-index will run.")

            with st.spinner("Processing PDF incrementally..."):
                try:
                    # Save upload to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.getbuffer())
                        tmp_path = tmp.name

                    report = proc.process(
                        pdf_path        = tmp_path,
                        collection_name = pdf_collection,
                        doc_id          = uploaded.name.replace(" ", "_"),
                        tenant_id       = tenant_id or None,
                    )
                    os.unlink(tmp_path)

                    # ── processing report ─────────────────────────────────
                    badge_html = '<span class="inc-badge">INCREMENTAL</span>' if report.incremental else '<span class="full-badge">FULL INDEX</span>'
                    st.markdown(badge_html, unsafe_allow_html=True)

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Pages total",    report.pages_total)
                    c2.metric("Added",          report.pages_added,   delta=f"+{report.pages_added}" if report.pages_added else None)
                    c3.metric("Updated",        report.pages_updated, delta=f"~{report.pages_updated}" if report.pages_updated else None)
                    c4.metric("Skipped",        report.pages_skipped, help="Unchanged pages — zero embedding cost")
                    c5.metric("Chunks embedded",report.chunks_embedded)

                    st.success(f"{report.summary()}")

                    if report.pages_deleted:
                        st.warning(f"{report.pages_deleted} page(s) removed from index (PDF shrunk).")
                    if report.pages_skipped == report.pages_total and report.incremental:
                        st.info("No changes detected — full re-index skipped. Upload a modified PDF to trigger updates.")

                    # Store collection for Ask tab — force selectbox to update
                    st.session_state["active_collection"] = pdf_collection
                    st.session_state["active_pdf"]        = uploaded.name
                    st.session_state["ask_coll_select"]   = pdf_collection
                    st.session_state["_ask_coll_source"]  = pdf_collection

                except ImportError as ie:
                    st.error(
                        f"PDF library missing: {ie}\n\n"
                        "Install with:  `pip install pymupdf`  (preferred)  "
                        "or  `pip install pdfminer.six`"
                    )
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    import traceback; st.code(traceback.format_exc())

    # ── incremental explainer ─────────────────────────────────────────────────
    with st.expander("How incremental processing works"):
        st.markdown("""
**Per-page content hashing**

| Page state | Action | Embedding cost |
|---|---|---|
| New page | Chunk → Embed → Upsert | Full embedding |
| Page content changed | Delete old chunks → Re-embed | Full re-embedding for that page only |
| Page unchanged | Skip entirely | **Zero** |
| Page removed from PDF | Delete old chunks from Qdrant | No embedding |

The hash index is stored in `.index/<collection>_index.json`.
On a 100-page PDF where only 3 pages changed → **97% embedding cost saved**.

**Industry use cases this solves:**
- Policy documents with quarterly updates (only changed sections re-indexed)
- Technical manuals with version patches
- Research papers with errata
- Any corpus where wholesale re-indexing is too expensive
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# ASK — smart query routing
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.header("Ask a Question")
    st.caption("The router picks the optimal pipeline spec based on query intent — no manual spec selection needed.")

    # Sync collection from whichever tab last processed a document.
    # Write to ask_coll_select BEFORE the widget renders so Streamlit picks it up.
    active_coll = st.session_state.get("active_collection", "")
    if active_coll and st.session_state.get("_ask_coll_source") != active_coll:
        st.session_state["ask_coll_select"]  = active_coll
        st.session_state["_ask_coll_source"] = active_coll

    active_pdf = st.session_state.get("active_pdf", "")

    # Fetch all available collections from Qdrant for the dropdown
    try:
        from rag_factory.components.base import get_qdrant_client as _gqc_ask
        _available_colls = sorted(
            c.name for c in _gqc_ask().get_collections().collections
        )
    except Exception:
        _available_colls = []

    col_ask1, col_ask2 = st.columns([3, 1])
    with col_ask1:
        query_text = st.text_input(
            "Your question",
            value="",
            placeholder="Ask anything about your uploaded documents...",
        )
    with col_ask2:
        if _available_colls:
            # Determine the correct default: last active collection, else first in list
            _want = st.session_state.get("ask_coll_select", active_coll or "")
            _default_idx = _available_colls.index(_want) \
                           if _want in _available_colls else 0
            ask_collection = st.selectbox(
                "Collection",
                _available_colls,
                index=_default_idx,
                key="ask_coll_select",
            )
        else:
            ask_collection = st.text_input(
                "Collection",
                value=active_coll or "idp_documents",
                key="ask_coll_text",
            )

    if active_pdf and ask_collection:
        st.caption(f"Active document: **{active_pdf}** → `{ask_collection}`")

    # Routing preview (live, before submit)
    if query_text.strip():
        from rag_factory.router import QueryRouter
        _router = QueryRouter(specs_dir=SPECS_DIR)
        _preview = _router.route(query_text, tenant_id=tenant_id or None)
        st.markdown("**Routing preview** (updates as you type)")
        _render_routing(_preview)

    if st.button("Ask", type="primary"):
        if not query_text.strip():
            st.error("Please enter a question.")
        else:
            # Determine spec
            from rag_factory.router import QueryRouter
            router = QueryRouter(specs_dir=SPECS_DIR)

            if manual_spec != "auto-route":
                from rag_factory.router import RoutingResult
                routing = RoutingResult(
                    spec=manual_spec, category="MANUAL", confidence=1.0,
                    reason="Manual override from sidebar.", heuristic=True,
                )
            else:
                routing = router.route(query_text, tenant_id=tenant_id or None)

            st.subheader("Routing Decision")
            _render_routing(routing)

            with st.spinner(f"Running {routing.spec} pipeline..."):
                try:
                    answer, chunks, scores, guard_log, faith, elapsed = _do_query(
                        query_text, ask_collection, top_k
                    )

                    st.subheader("Answer")
                    st.success(answer)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Elapsed",      f"{elapsed}ms")
                    c2.metric("Chunks used",  len(chunks))
                    c3.metric("Top score",    f"{scores[0]:.3f}" if scores else "—")
                    c4.metric("Faithfulness", f"{faith:.2f}" if faith else "—")

                    st.subheader("Guard Log")
                    with st.expander("Guard events", expanded=True):
                        _render_guard_log(guard_log)

                    st.subheader("Retrieved Chunks")
                    for i, (ch, sc) in enumerate(zip(chunks, scores)):
                        label = (
                            f"Chunk {i+1} | score={sc:.4f} | "
                            f"page={ch.get('page_num','?')} | "
                            f"doc={ch.get('pdf_name', ch.get('doc_id','?'))}"
                        )
                        with st.expander(label):
                            st.write(ch.get("text", ""))

                    # Persist for evaluate tab
                    st.session_state["last_query"]   = query_text
                    st.session_state["last_answer"]  = answer
                    st.session_state["last_chunks"]  = chunks
                    st.session_state["last_routing"] = routing.to_dict()

                except Exception as e:
                    st.error(f"Query failed: {e}")
                    import traceback; st.code(traceback.format_exc())

    # Routing logic explainer
    with st.expander("Routing decision table"):
        from rag_factory.router import ROUTING_LOGIC
        import pandas as pd
        df_route = pd.DataFrame(ROUTING_LOGIC, columns=["Query signal", "Spec selected", "Reason"])
        st.dataframe(df_route, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("RAGAS Evaluation")
    ev_q  = st.text_input("Query",  value=st.session_state.get("last_query",""), key="ev_q")
    ev_a  = st.text_area("Answer", value=st.session_state.get("last_answer",""), height=120, key="ev_a")
    ev_gt = st.text_area("Ground Truth", height=70, key="ev_gt",
                          value="Medicaid covers hospital, physician, lab/X-ray, home health, and nursing facility services.")
    ev_chunks = st.session_state.get("last_chunks", [])
    st.caption(f"Using {len(ev_chunks)} chunks from last query.")

    if st.button("Evaluate", type="primary", key="eval_btn"):
        if not ev_q or not ev_a:
            st.error("Query and answer required.")
        else:
            with st.spinner("Running RAGAS 4-metric evaluation..."):
                try:
                    from rag_factory.components.production import EvaluationRAG
                    scores = EvaluationRAG().run({
                        "query": ev_q, "answer": ev_a,
                        "retrieved_chunks": ev_chunks, "ground_truth": ev_gt,
                    })
                    thresholds = {
                        "faithfulness": 0.80, "answer_relevancy": 0.75,
                        "context_precision": 0.70, "context_recall": 0.70,
                    }
                    cols = st.columns(4)
                    for i, (metric, thr) in enumerate(thresholds.items()):
                        val = scores.get(metric, 0.0)
                        ok  = val >= thr
                        cols[i].metric(
                            metric.replace("_", " ").title(), f"{val:.3f}",
                            delta=f"{'PASS' if ok else 'FAIL'} (>={thr})",
                            delta_color="normal" if ok else "inverse",
                        )
                    if st.session_state.get("last_routing"):
                        st.caption(f"Spec used: {st.session_state['last_routing'].get('spec','?')} | "
                                   f"Category: {st.session_state['last_routing'].get('category','?')}")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE SPECS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("Compare Pipeline Specs")
    c1c, c2c = st.columns(2)
    with c1c:
        spec_a = st.selectbox("Spec A", spec_files, index=0, key="cmp_a")
    with c2c:
        spec_b = st.selectbox("Spec B", spec_files, index=min(1, len(spec_files)-1), key="cmp_b")

    cmp_q    = st.text_input("Query", value="What services does Medicaid cover?", key="cmp_q")
    cmp_coll = st.text_input("Collection", value=st.session_state.get("active_collection","factory_pdf_demo"), key="cmp_coll")

    if st.button("Compare", type="primary", key="cmp_btn"):
        if not cmp_q:
            st.error("Query required.")
        else:
            col_a, col_b = st.columns(2)
            results = {}
            for spec_name, col in [(spec_a, col_a), (spec_b, col_b)]:
                with col:
                    with st.spinner(f"Running {spec_name}..."):
                        try:
                            ans, chs, scs, gl, faith, ela = _do_query(cmp_q, cmp_coll, top_k)
                            results[spec_name] = dict(answer=ans, chunks=chs, scores=scs,
                                                       guard_log=gl, faith=faith, elapsed=ela)
                            st.subheader(spec_name)
                            st.metric("Elapsed",      f"{ela}ms")
                            st.metric("Faithfulness", f"{faith:.2f}" if faith else "—")
                            st.info(ans[:500])
                            _render_guard_log(gl)
                        except Exception as e:
                            st.error(f"{spec_name} failed: {e}")

            if len(results) == 2:
                st.divider()
                st.caption("Winner by faithfulness: " + (
                    spec_a if (results[spec_a].get("faith") or 0) >= (results[spec_b].get("faith") or 0)
                    else spec_b
                ))


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Pipeline Specs")
    import pandas as pd
    rows = _list_specs()
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.header("End-to-End Architecture")
    st.code("""
PDF Upload
    |
    +─── IncrementalPDFProcessor
    |         per-page SHA-256 hash
    |         skip UNCHANGED pages (zero embedding cost)
    |         delete UPDATED/DELETED chunk IDs from Qdrant
    |         embed + upsert ADDED/UPDATED pages only
    |         persist hash index (.index/<collection>_index.json)
    |
User Query
    |
    +─── QueryRouter (Layer 1: heuristic regex, Layer 2: LLM)
    |         → picks: simple / production / agentic / multitenant spec
    |
    +─── AmbiguityGuard    [rewrite negation, multi-intent, scope]
    +─── Dense Retrieval   [Titan V2 → Qdrant, tenant filter]
    +─── RetrievalGuard    [K-dilution, stale, vocab-gap]
    +─── SystemGuard       [injection drop, PII flag]
    +─── LLM Generation    [AWS Bedrock LLM]
    +─── GenerationGuard   [faithfulness score, lost-in-middle]
    |
    Answer  +  Guard log  +  RAGAS scores  +  Routing explanation
""", language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# MANIFEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_manifest:
    st.header("Component Manifest (38 components)")
    try:
        from rag_factory.spec import MANIFEST
        rows_m = []
        for name, spec in MANIFEST._by_name.items():
            rows_m.append({
                "Name":      name,
                "Tier":      spec.tier,
                "Role":      spec.role.value,
                "Notebook":  spec.notebook_ref,
                "Async":     spec.is_async,
                "Streaming": spec.is_streaming,
                "Failures":  ", ".join(spec.failure_modes),
            })
        df_m = pd.DataFrame(rows_m).sort_values(["Tier", "Name"])
        st.dataframe(df_m, use_container_width=True, hide_index=True)

        st.divider()
        by_tier = MANIFEST.summary()["by_tier"]
        cols = st.columns(min(len(by_tier), 5))
        for i, (t, c) in enumerate(sorted(by_tier.items())):
            cols[i % len(cols)].metric(f"Tier {t}", c)
    except Exception as e:
        st.error(f"Could not load manifest: {e}")
