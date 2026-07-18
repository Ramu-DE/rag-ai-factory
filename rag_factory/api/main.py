# -*- coding: utf-8 -*-
"""
AI RAG Factory — FastAPI service
Endpoints: /health  /specs  /ingest  /query  /evaluate  /compare
"""
from __future__ import annotations
import os, time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv(override=True)

from .schemas import (
    IngestRequest, IngestResponse,
    QueryRequest,  QueryResponse,
    EvaluateRequest, EvaluateResponse,
    CompareRequest,  CompareResponse,
    SpecListResponse, HealthResponse,
)
from ..spec       import PipelineSpec, MANIFEST, VALIDATOR
from ..assembler  import Assembler
from ..temporal   import TEMPORAL_AVAILABLE

app = FastAPI(
    title="AI RAG Factory",
    description="NVIDIA-inspired, Temporal-ready RAG pipeline engine over 33 patterns",
    version="0.1.0",
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
