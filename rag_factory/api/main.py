# -*- coding: utf-8 -*-
"""
AI RAG Factory — FastAPI service
Endpoints: /health  /specs  /ingest  /query  /evaluate  /compare
           /idp/classify  /idp/process  /idp/batch  /idp/collections/{name}/stats
"""
from __future__ import annotations
import os, time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv(override=True)

from .schemas import (
    IngestRequest, IngestResponse,
    QueryRequest,  QueryResponse,
    EvaluateRequest, EvaluateResponse,
    CompareRequest,  CompareResponse,
    SpecListResponse, HealthResponse,
    IDPClassifyRequest, IDPClassifyResponse,
    IDPProcessRequest, IDPBatchResponse, IDPFileResult,
    IDPCollectionStats,
)
from ..spec       import PipelineSpec, MANIFEST, VALIDATOR
from ..assembler  import Assembler
from ..temporal   import TEMPORAL_AVAILABLE

app = FastAPI(
    title="AI RAG Factory",
    description="NVIDIA-inspired, Temporal-ready RAG pipeline engine over 33 patterns",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_assembler = Assembler()
SPECS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "specs")


def _load_spec(spec_name: str) -> PipelineSpec:
    """Load a spec by filename (e.g. 'simple.yaml') or full path."""
    if os.path.isfile(spec_name):
        return PipelineSpec.from_yaml(spec_name)
    candidate = os.path.join(SPECS_DIR, spec_name)
    if os.path.isfile(candidate):
        return PipelineSpec.from_yaml(candidate)
    raise HTTPException(status_code=404, detail=f"Spec not found: {spec_name}")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        manifest_size=len(MANIFEST),
        temporal=TEMPORAL_AVAILABLE,
    )


# ---------------------------------------------------------------------------
# GET /specs
# ---------------------------------------------------------------------------
@app.get("/specs", response_model=SpecListResponse, tags=["meta"])
def list_specs():
    results = []
    if os.path.isdir(SPECS_DIR):
        for fname in sorted(os.listdir(SPECS_DIR)):
            if fname.endswith(".yaml"):
                try:
                    spec = PipelineSpec.from_yaml(os.path.join(SPECS_DIR, fname))
                    vr   = VALIDATOR.validate(spec)
                    results.append({
                        "file":          fname,
                        "name":          spec.name,
                        "description":   spec.description,
                        "chunker":       spec.ingestion.chunker,
                        "retrieval":     spec.retrieval.strategy,
                        "agentic_mode":  spec.generation.agentic_mode,
                        "temporal":      spec.temporal.enabled,
                        "valid":         vr.valid,
                        "warnings":      vr.warnings,
                    })
                except Exception as e:
                    results.append({"file": fname, "error": str(e)})
    return SpecListResponse(specs=results)


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse, tags=["pipeline"])
def ingest(req: IngestRequest):
    from ..components.base import get_qdrant_client, embed, chunk_id
    from qdrant_client.models import VectorParams, Distance, PointStruct

    t0     = time.time()
    qdrant = get_qdrant_client()
    cname  = req.collection_name

    # Select chunker
    chunker_map = {
        "fixed_chunking":         lambda: _chunker_fixed(req.text),
        "semantic_chunking":      lambda: _chunker_semantic(req.text),
        "sentence_window_chunking": lambda: _chunker_window(req.text),
    }
    chunk_fn = chunker_map.get(req.chunker, chunker_map["fixed_chunking"])
    chunks   = chunk_fn()

    # Ensure collection exists
    existing = [c.name for c in qdrant.get_collections().collections]
    if cname not in existing:
        qdrant.create_collection(
            cname,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    # Embed + upsert
    pages_changed = 0
    points = []
    for ch in chunks:
        import hashlib, uuid as _uuid
        text     = ch.get("text","")
        pg_hash  = hashlib.sha256(text.encode()).hexdigest()
        cid      = chunk_id(f"{cname}:{req.doc_id}:{text[:80]}")
        payload  = {**ch, "doc_id": req.doc_id, "page_hash": pg_hash}
        if req.tenant_id:
            payload["tenant_id"] = req.tenant_id
        vec = embed(text)
        points.append(PointStruct(id=cid, vector=vec, payload=payload))
        pages_changed += 1

    qdrant.upsert(collection_name=cname, points=points)
    elapsed = int((time.time() - t0) * 1000)

    return IngestResponse(
        collection_name=cname,
        chunk_count=len(chunks),
        pages_changed=pages_changed,
        elapsed_ms=elapsed,
    )


def _chunker_fixed(text: str, size: int = 500, overlap: int = 50):
    chunks, start = [], 0
    while start < len(text):
        chunks.append({"text": text[start:start+size], "chunk_index": len(chunks)})
        start += size - overlap
    return chunks

def _chunker_semantic(text: str):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [{"text": s, "chunk_index": i} for i, s in enumerate(sentences) if len(s) > 20]

def _chunker_window(text: str, window: int = 3):
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 10]
    chunks = []
    for i, s in enumerate(sentences):
        lo = max(0, i - window)
        hi = min(len(sentences), i + window + 1)
        chunks.append({"text": s, "window_text": " ".join(sentences[lo:hi]), "chunk_index": i})
    return chunks


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse, tags=["pipeline"])
def query(req: QueryRequest):
    from ..components.base import embed, dense_search, llm_call, get_qdrant_client
    from ..guards import RetrievalGuard, GenerationGuard, AmbiguityGuard, SystemGuard

    t0 = time.time()

    # Ambiguity guard first
    ctx: Dict[str, Any] = {
        "query": req.query, "retrieved_chunks": [],
        "tenant_id": req.tenant_id or "default",
    }
    a_result = AmbiguityGuard().run(ctx)
    effective_query = a_result.get("query", req.query)
    guard_log: Dict[str, Any] = {"ambiguity": a_result.get("guard_log",{}).get("ambiguity",[])}

    # Retrieval
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qdrant    = get_qdrant_client()
    vec       = embed(effective_query)
    qfilter   = None
    if req.tenant_id:
        qfilter = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=req.tenant_id))])
    points    = dense_search(qdrant, req.collection_name, vec, req.top_k, filters=qfilter)
    chunks    = [p.payload for p in points]
    scores    = [p.score   for p in points]

    # Retrieval guard
    r_ctx    = {"query": effective_query, "retrieved_chunks": chunks,
                "collection_name": req.collection_name, "tenant_id": req.tenant_id or "default"}
    r_result = RetrievalGuard().run(r_ctx)
    guard_log["retrieval"] = r_result.get("guard_log",{}).get("retrieval",[])

    # System guard
    s_result  = SystemGuard().run({**r_ctx, "retrieved_chunks": chunks})
    chunks    = s_result.get("retrieved_chunks", chunks)
    guard_log["system"] = s_result.get("guard_log",{}).get("system",[])

    # Generation
    context = "\n\n".join(c.get("text","") for c in chunks)
    answer  = llm_call(
        f"Answer using only the provided context.\n\nContext:\n{context}\n\nQuestion: {req.query}",
        max_tokens=1024,
    )

    # Generation guard
    g_result = GenerationGuard().run({
        "query": req.query, "answer": answer, "retrieved_chunks": chunks,
        "tenant_id": req.tenant_id or "default",
    })
    guard_log["generation"] = g_result.get("guard_log",{}).get("generation",[])
    faithfulness = g_result.get("faithfulness_score")

    elapsed = int((time.time() - t0) * 1000)

    return QueryResponse(
        answer=answer,
        retrieved_chunks=chunks,
        scores=scores,
        guard_log=guard_log,
        faithfulness=faithfulness,
        elapsed_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------------------------
@app.post("/evaluate", response_model=EvaluateResponse, tags=["pipeline"])
def evaluate(req: EvaluateRequest):
    from ..components.production import EvaluationRAG
    result = EvaluationRAG().run({
        "query":            req.query,
        "answer":           req.answer,
        "retrieved_chunks": req.retrieved_chunks,
        "ground_truth":     req.ground_truth,
    })
    return EvaluateResponse(**result)


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------
@app.post("/compare", response_model=CompareResponse, tags=["pipeline"])
def compare(req: CompareRequest):
    from ..components.base import embed, dense_search, llm_call, get_qdrant_client
    from ..guards import GenerationGuard

    def _run_spec(spec_name: str):
        spec   = _load_spec(spec_name)
        t0     = time.time()
        qdrant = get_qdrant_client()
        vec    = embed(req.query)
        k      = spec.retrieval.top_k
        points = dense_search(qdrant, req.collection_name, vec, k)
        chunks = [p.payload for p in points]
        ctx    = "\n\n".join(c.get("text","") for c in chunks)
        answer = llm_call(
            f"Context:\n{ctx}\n\nQuestion: {req.query}", max_tokens=1024
        )
        faith_result = GenerationGuard().run({
            "query": req.query, "answer": answer,
            "retrieved_chunks": chunks, "tenant_id": "default",
        })
        elapsed = int((time.time() - t0) * 1000)
        return answer, faith_result.get("faithfulness_score"), elapsed

    ans_a, faith_a, ela = _run_spec(req.spec_a)
    ans_b, faith_b, elb = _run_spec(req.spec_b)

    return CompareResponse(
        spec_a=req.spec_a, spec_b=req.spec_b,
        answer_a=ans_a,    answer_b=ans_b,
        faithfulness_a=faith_a, faithfulness_b=faith_b,
        elapsed_ms_a=ela,  elapsed_ms_b=elb,
    )


# ---------------------------------------------------------------------------
# POST /idp/classify  — classify document text, no ingestion
# ---------------------------------------------------------------------------
@app.post("/idp/classify", response_model=IDPClassifyResponse, tags=["idp"])
def idp_classify(req: IDPClassifyRequest):
    """
    Classify a document from its extracted text.
    Returns doc_type, confidence, and recommended extraction_mode.
    No embedding or storage performed — pure classification only.
    """
    from ..document_classifier import CLASSIFIER
    result = CLASSIFIER.classify(req.text, filename=req.filename)
    return IDPClassifyResponse(
        doc_type        = result.doc_type,
        confidence      = result.confidence,
        heuristic       = result.heuristic,
        reason          = result.reason,
        extraction_mode = result.extraction_mode,
    )


# ---------------------------------------------------------------------------
# POST /idp/process  — process a single uploaded file end-to-end
# ---------------------------------------------------------------------------
@app.post("/idp/process", response_model=IDPFileResult, tags=["idp"])
async def idp_process(
    file              : UploadFile     = File(...),
    collection_name   : str            = Form("idp_documents"),
    doc_type_override : Optional[str]  = Form(None),
    extraction_mode   : str            = Form("auto"),
    chunk_size        : int            = Form(400),
    force_reindex     : bool           = Form(False),
    tenant_id         : Optional[str]  = Form(None),
):
    """
    Upload a single PDF/image and run the full IDP pipeline:
    extract → classify → skill → validate → incremental ingest.
    """
    import tempfile, os
    from fastapi import UploadFile
    from ..idp_pipeline import IDPPipeline

    contents = await file.read()
    suffix   = os.path.splitext(file.filename or "doc.pdf")[1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pipeline = IDPPipeline(
            collection_name = collection_name,
            index_dir       = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".index"
            ),
            chunk_size      = chunk_size,
            force_reindex   = force_reindex,
            tenant_id       = tenant_id,
        )
        result = pipeline.process(
            file_path         = tmp_path,
            doc_type_override = doc_type_override,
            extraction_mode   = extraction_mode,
        )
    finally:
        os.unlink(tmp_path)

    ir = result.ingest_report
    return IDPFileResult(
        file_name             = result.file_name,
        doc_type              = result.doc_type,
        confidence            = result.classification.confidence,
        is_scanned            = result.is_scanned,
        method                = result.method,
        pages_total           = ir.pages_total,
        pages_added           = ir.pages_added,
        pages_updated         = ir.pages_updated,
        pages_skipped         = ir.pages_skipped,
        chunks_embedded       = ir.chunks_embedded,
        chunks_deleted        = ir.chunks_deleted,
        elapsed_ms            = result.elapsed_ms,
        incremental           = ir.incremental,
        fields                = result.extraction.fields   if result.extraction  else {},
        tables                = result.extraction.tables   if result.extraction  else [],
        extraction_confidence = result.extraction.confidence if result.extraction else None,
        validation_valid      = result.validation.valid    if result.validation  else None,
        validation_errors     = result.validation.errors   if result.validation  else [],
        validation_warnings   = result.validation.warnings if result.validation  else [],
    )


# ---------------------------------------------------------------------------
# POST /idp/batch  — process multiple files, returns per-file results
# ---------------------------------------------------------------------------
@app.post("/idp/batch", response_model=IDPBatchResponse, tags=["idp"])
async def idp_batch(
    files           : List[UploadFile] = File(...),
    collection_name : str              = Form("idp_documents"),
    extraction_mode : str              = Form("auto"),
    chunk_size      : int              = Form(400),
    force_reindex   : bool             = Form(False),
    tenant_id       : Optional[str]    = Form(None),
):
    """
    Process multiple uploaded files in sequence.
    Each file is classified, extracted, validated, and incrementally ingested.
    Returns per-file results plus aggregate stats.
    Continues processing remaining files even if one fails.
    """
    import tempfile, os
    from fastapi import UploadFile
    from ..idp_pipeline import IDPPipeline

    t0      = time.time()
    INDEX_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".index"
    )
    pipeline = IDPPipeline(
        collection_name = collection_name,
        index_dir       = INDEX_DIR,
        chunk_size      = chunk_size,
        force_reindex   = force_reindex,
        tenant_id       = tenant_id,
    )

    file_results : List[IDPFileResult] = []
    total_pages = total_chunks = succeeded = failed = 0

    for upload in files:
        contents = await upload.read()
        suffix   = os.path.splitext(upload.filename or "doc.pdf")[1] or ".pdf"
        fname    = upload.filename or "unknown"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            result = pipeline.process(
                file_path       = tmp_path,
                extraction_mode = extraction_mode,
            )
            ir = result.ingest_report
            file_results.append(IDPFileResult(
                file_name             = result.file_name,
                doc_type              = result.doc_type,
                confidence            = result.classification.confidence,
                is_scanned            = result.is_scanned,
                method                = result.method,
                pages_total           = ir.pages_total,
                pages_added           = ir.pages_added,
                pages_updated         = ir.pages_updated,
                pages_skipped         = ir.pages_skipped,
                chunks_embedded       = ir.chunks_embedded,
                chunks_deleted        = ir.chunks_deleted,
                elapsed_ms            = result.elapsed_ms,
                incremental           = ir.incremental,
                fields                = result.extraction.fields   if result.extraction  else {},
                tables                = result.extraction.tables   if result.extraction  else [],
                extraction_confidence = result.extraction.confidence if result.extraction else None,
                validation_valid      = result.validation.valid    if result.validation  else None,
                validation_errors     = result.validation.errors   if result.validation  else [],
                validation_warnings   = result.validation.warnings if result.validation  else [],
            ))
            total_pages  += ir.pages_total
            total_chunks += ir.chunks_embedded
            succeeded    += 1
        except Exception as exc:
            file_results.append(IDPFileResult(
                file_name=fname, doc_type="error", confidence=0.0,
                is_scanned=False, method="", pages_total=0, pages_added=0,
                pages_updated=0, pages_skipped=0, chunks_embedded=0,
                chunks_deleted=0, elapsed_ms=0, incremental=False,
                error=str(exc),
            ))
            failed += 1
        finally:
            os.unlink(tmp_path)

    return IDPBatchResponse(
        collection_name = collection_name,
        files_submitted = len(files),
        files_succeeded = succeeded,
        files_failed    = failed,
        total_pages     = total_pages,
        total_chunks    = total_chunks,
        elapsed_ms      = int((time.time() - t0) * 1000),
        results         = file_results,
    )


# ---------------------------------------------------------------------------
# GET /idp/collections/{name}/stats  — index stats for a collection
# ---------------------------------------------------------------------------
@app.get("/idp/collections/{collection_name}/stats",
         response_model=IDPCollectionStats, tags=["idp"])
def idp_collection_stats(collection_name: str):
    """
    Return incremental index stats for a collection:
    total chunks in Qdrant, indexed pages, last processed time.
    """
    from ..pdf_processor import IncrementalPDFProcessor
    from ..components.base import get_qdrant_client

    INDEX_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".index"
    )
    proc  = IncrementalPDFProcessor(index_dir=INDEX_DIR)
    stats = proc.get_index_stats(collection_name)

    # Count total Qdrant points
    total_chunks = 0
    doc_types: List[str] = []
    try:
        qdrant = get_qdrant_client()
        cinfo  = qdrant.get_collection(collection_name)
        total_chunks = cinfo.points_count or 0
        # Sample payloads to find doc_types seen
        sample = qdrant.scroll(collection_name=collection_name, limit=100, with_payload=True)
        seen: set = set()
        for pt in (sample[0] if sample else []):
            dt = pt.payload.get("doc_type") or pt.payload.get("doc_id","")
            if dt:
                seen.add(dt)
        doc_types = list(seen)
    except Exception:
        total_chunks = stats.get("total_chunks", 0)

    return IDPCollectionStats(
        collection_name = collection_name,
        total_chunks    = total_chunks,
        indexed_pages   = stats.get("indexed_pages", 0),
        last_processed  = stats.get("last_processed"),
        doc_id          = stats.get("doc_id"),
        doc_types_seen  = doc_types,
    )


