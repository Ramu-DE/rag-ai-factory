# -*- coding: utf-8 -*-
"""
AI RAG Factory — FastAPI service
Endpoints: /health  /specs  /ingest  /ingest/finalized  /query  /evaluate  /compare
           /idp/classify  /idp/process  /idp/batch  /idp/collections/{name}/stats

Service mode (env SERVICE_TOKEN non-empty): the app runs as a private sidecar
behind the AI Control Plane. Every route except GET /health requires
X-Service-Token, X-Tenant-Id and X-Persona-Id; the header tenant overrides any
tenant in the request and is stamped on / filtered for in Qdrant. CORS is
limited to ALLOWED_ORIGINS and the interactive docs are disabled.
"""
from __future__ import annotations
import hashlib, hmac, json, logging, os, re, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import (
    Body, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile,
)
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
    FinalizedIngestResponse,
    DecisionQueryRequest,
)
from ..spec       import PipelineSpec, MANIFEST, VALIDATOR
from ..assembler  import Assembler
from ..temporal   import TEMPORAL_AVAILABLE

_log = logging.getLogger("rag_factory.api")

# ---------------------------------------------------------------------------
# Service mode: shared-token auth + asserted tenant / persona
# ---------------------------------------------------------------------------
TENANT_RE  = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
PERSONA_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _service_token() -> str:
    return os.getenv("SERVICE_TOKEN", "").strip()


SERVICE_MODE = bool(_service_token())   # fixed at startup: CORS + docs


@dataclass(frozen=True)
class ServiceContext:
    service_mode: bool
    tenant_id:    Optional[str] = None
    persona_id:   Optional[str] = None


def service_context(
    request:         Request,
    x_service_token: Optional[str] = Header(None),
    x_tenant_id:     Optional[str] = Header(None),
    x_persona_id:    Optional[str] = Header(None),
) -> ServiceContext:
    """
    App-wide dependency. With SERVICE_TOKEN unset it is a no-op (legacy
    behaviour). With it set, every route except GET /health requires a valid
    X-Service-Token (401) and well-formed X-Tenant-Id / X-Persona-Id (400).
    """
    token = _service_token()
    if not token:
        return ServiceContext(service_mode=False)
    if request.method == "GET" and request.scope.get("endpoint") is health:
        return ServiceContext(service_mode=True)
    if not x_service_token or not hmac.compare_digest(
        x_service_token.encode("utf-8"), token.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Service-Token")
    if not x_tenant_id or not TENANT_RE.fullmatch(x_tenant_id):
        raise HTTPException(status_code=400,
                            detail="Missing or invalid X-Tenant-Id (^[a-z0-9][a-z0-9-]{0,62}$)")
    if not x_persona_id or not PERSONA_RE.fullmatch(x_persona_id):
        raise HTTPException(status_code=400,
                            detail="Missing or invalid X-Persona-Id (^[A-Za-z0-9_-]{1,64}$)")
    _log.info("service request %s %s tenant=%s persona=%s",
              request.method, request.url.path, x_tenant_id, x_persona_id)
    return ServiceContext(service_mode=True, tenant_id=x_tenant_id, persona_id=x_persona_id)


def _effective_tenant(ctx: ServiceContext, requested: Optional[str]) -> Optional[str]:
    """In service mode the asserted header tenant always wins."""
    return ctx.tenant_id if ctx.service_mode else requested


def _tenant_filter(tenant_id: Optional[str]):
    if not tenant_id:
        return None
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    return Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])


app = FastAPI(
    title="AI RAG Factory",
    description="NVIDIA-inspired, Temporal-ready RAG pipeline engine over 33 patterns",
    version="0.2.0",
    dependencies=[Depends(service_context)],
    # The docs/openapi routes bypass dependencies, so hide them in service mode.
    docs_url=None if SERVICE_MODE else "/docs",
    redoc_url=None if SERVICE_MODE else "/redoc",
    openapi_url=None if SERVICE_MODE else "/openapi.json",
)
if SERVICE_MODE:
    _origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if _origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_origins, allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Service-Token", "X-Tenant-Id", "X-Persona-Id"],
        )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

_assembler = Assembler()
SPECS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "specs")


def _load_spec(spec_name: str) -> PipelineSpec:
    """Load a spec by filename (e.g. 'simple.yaml') or full path.
    In service mode only bare filenames inside specs/ are accepted."""
    if SERVICE_MODE or _service_token():
        if os.path.basename(spec_name) != spec_name or spec_name in ("", ".", ".."):
            raise HTTPException(status_code=400, detail="spec must be a file name in specs/")
    elif os.path.isfile(spec_name):
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
def ingest(req: IngestRequest, ctx: ServiceContext = Depends(service_context)):
    from ..components.base import get_qdrant_client, embed, chunk_id
    from qdrant_client.models import VectorParams, Distance, PointStruct

    t0     = time.time()
    qdrant = get_qdrant_client()
    cname  = req.collection_name
    tenant = _effective_tenant(ctx, req.tenant_id)

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
        scope    = f"{tenant}/{req.doc_id}" if tenant else req.doc_id
        cid      = chunk_id(f"{cname}:{scope}:{text[:80]}")
        payload  = {**ch, "doc_id": req.doc_id, "page_hash": pg_hash}
        if tenant:
            payload["tenant_id"] = tenant
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


_FILTER_INDEX_FIELDS = ("tenant_id", "doc_id", "finalized_document_id", "doc_type", "chunk_kind")
_INDEXED: set = set()


def _ensure_filter_indexes(qdrant, cname: str) -> None:
    """Keyword payload indexes for the fields every finalized query filters on.

    Qdrant Cloud clusters in strict mode refuse filters on unindexed fields,
    and on large collections unindexed filters are slow. Idempotent.
    """
    if cname in _INDEXED:
        return
    from qdrant_client.models import PayloadSchemaType
    for field in _FILTER_INDEX_FIELDS:
        try:
            qdrant.create_payload_index(cname, field_name=field, field_schema=PayloadSchemaType.KEYWORD)
        except Exception as exc:   # already exists, or the in-memory client
            _log.debug("payload index %s.%s: %s", cname, field, exc)
    _INDEXED.add(cname)


# Chunkers a finalized document can be ingested with: name -> (text chunker,
# also emit field/table chunks). The control plane picks one per corpus.
_FINALIZED_CHUNKERS = {
    "fixed_chunking":           (_chunker_fixed,    False),
    "semantic_chunking":        (_chunker_semantic, False),
    "sentence_window_chunking": (_chunker_window,   False),
    "structured_chunking":      (_chunker_fixed,    True),
}


# ---------------------------------------------------------------------------
# POST /ingest/finalized  — FinalizedDocument v1 from the AI Control Plane
# ---------------------------------------------------------------------------
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@app.post("/ingest/finalized", response_model=FinalizedIngestResponse, tags=["pipeline"])
def ingest_finalized(
    body            : Dict[str, Any] = Body(..., description="FinalizedDocument v1 JSON; "
                                            "may carry an extra `collection_name` key"),
    collection_name : Optional[str]  = Query(None, description="Target Qdrant collection "
                                             "(overrides body.collection_name)"),
    chunker         : Optional[str]  = Query(None, description="One of "
                                             + ", ".join(sorted(["fixed_chunking", "semantic_chunking",
                                                                 "sentence_window_chunking", "structured_chunking"]))
                                             + " (default fixed_chunking; overrides body.chunker)"),
    svc             : ServiceContext = Depends(service_context),
):
    """
    Ingest a finalized (reviewed, four-eyes approved) document.

    Checks: structure / schemaVersion (422), tenant — in service mode
    body.tenantId must equal X-Tenant-Id (403), contentSha256 over the
    canonical JSON of `content` (409), and that no newer version of the same
    documentId is already in the collection for this tenant (409).

    Then content.pages (or content.text when no pages) is fixed-chunked,
    embedded and upserted. Point ids are deterministic from
    (tenant, finalizedDocumentId, version, chunk_index): re-ingesting the same
    version replaces it; a new version of the same documentId removes the
    previous versions' points in this collection.
    """
    from ..components.base import embed, get_qdrant_client
    from ..pdf_processor import _ensure_collection, _delete_chunks_by_ids
    from ..finalized_ingest import (
        FinalizedDocumentError, validate_finalized_document, content_sha256, build_payloads,
    )
    from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

    t0  = time.time()
    doc = dict(body)
    body_collection = doc.pop("collection_name", None)   # not part of the contract
    body_chunker    = doc.pop("chunker", None)            # not part of the contract
    cname = collection_name or body_collection
    chunker_name = chunker or body_chunker or "fixed_chunking"
    if chunker_name not in _FINALIZED_CHUNKERS:
        raise HTTPException(status_code=400,
                            detail=f"chunker must be one of {sorted(_FINALIZED_CHUNKERS)}")
    if not isinstance(cname, str) or not _COLLECTION_RE.fullmatch(cname):
        raise HTTPException(status_code=400,
                            detail="collection_name (query or body) is required: ^[A-Za-z0-9_-]{1,128}$")

    try:
        validate_finalized_document(doc)
    except FinalizedDocumentError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})

    if svc.service_mode and doc["tenantId"] != svc.tenant_id:
        raise HTTPException(status_code=403, detail="tenantId does not match X-Tenant-Id")
    tenant = svc.tenant_id if svc.service_mode else doc["tenantId"]

    if content_sha256(doc["content"]) != doc["contentSha256"]:
        raise HTTPException(status_code=409,
                            detail="contentSha256 does not match the canonical JSON of content")

    version = doc["version"]
    qdrant  = get_qdrant_client()
    _ensure_collection(qdrant, cname)
    _ensure_filter_indexes(qdrant, cname)

    # Existing points of this document for this tenant (all versions).
    doc_filter = Filter(must=[
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant)),
        FieldCondition(key="doc_id",    match=MatchValue(value=doc["documentId"])),
    ])
    existing: Dict[str, int] = {}
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=cname, scroll_filter=doc_filter, limit=256, offset=offset,
            with_payload=["version"], with_vectors=False,
        )
        for pt in batch:
            existing[str(pt.id)] = int((pt.payload or {}).get("version") or 0)
        if offset is None:
            break
    newer = sorted({v for v in existing.values() if v > version})
    if newer:
        raise HTTPException(status_code=409,
                            detail=f"A newer version ({newer[-1]}) of {doc['documentId']} is already ingested")

    chunk_fn, structured = _FINALIZED_CHUNKERS[chunker_name]
    items  = build_payloads(doc, tenant, chunk_fn, structured=structured, chunker_name=chunker_name)
    points = [PointStruct(id=pid, vector=embed(payload["text"]), payload=payload)
              for pid, payload in items]
    if points:
        qdrant.upsert(collection_name=cname, points=points)

    new_ids = {pid for pid, _ in items}
    stale   = [pid for pid in existing if pid not in new_ids]
    deleted = _delete_chunks_by_ids(qdrant, cname, stale)

    _log.info("finalized ingest tenant=%s persona=%s doc=%s fd=%s v%s -> %s (%s): +%d -%d",
              tenant, svc.persona_id, doc["documentId"], doc["finalizedDocumentId"],
              version, cname, chunker_name, len(points), deleted)

    return FinalizedIngestResponse(
        collection_name       = cname,
        tenant_id             = tenant,
        document_id           = doc["documentId"],
        finalized_document_id = doc["finalizedDocumentId"],
        version               = version,
        content_sha256        = doc["contentSha256"],
        chunk_count           = len(items),
        chunker               = chunker_name,
        points_upserted       = len(points),
        points_deleted        = deleted,
        replaced_same_version = any(v == version for v in existing.values()),
        superseded_versions   = sorted({v for v in existing.values() if v < version}),
        elapsed_ms            = int((time.time() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse, tags=["pipeline"])
def query(req: QueryRequest, svc: ServiceContext = Depends(service_context)):
    """
    Dense retrieval + guarded generation.

    If `spec` is given, the spec's `retrieval.top_k` replaces `top_k`, and a
    `filtered_rag` / `multitenant_rag` retrieval strategy makes the tenant
    filter mandatory (400 without a tenant). In service mode the X-Tenant-Id
    header is always the tenant and every search is must-filtered on it.

    Limitation (Phase 0): only top_k and the tenant filter are honoured from
    the spec. Retrieval is always dense top-k; the spec's chunker, reranker,
    compression and agentic/generation settings are NOT executed here — the
    full Assembler path is Phase 2.
    """
    from ..components.base import embed, dense_search, llm_call, get_qdrant_client
    from ..guards import RetrievalGuard, GenerationGuard, AmbiguityGuard, SystemGuard

    t0 = time.time()
    tenant   = _effective_tenant(svc, req.tenant_id)
    top_k    = req.top_k
    strategy = None
    if req.spec:
        spec     = _load_spec(req.spec)
        top_k    = spec.retrieval.top_k
        strategy = spec.retrieval.strategy
        if strategy in _TENANT_STRATEGIES and not tenant:
            raise HTTPException(status_code=400,
                                detail=f"Spec strategy '{strategy}' requires a tenant_id")

    # Ambiguity guard first
    ctx: Dict[str, Any] = {
        "query": req.query, "retrieved_chunks": [],
        "tenant_id": tenant or "default",
    }
    a_result = AmbiguityGuard().run(ctx)
    effective_query = a_result.get("query", req.query)
    guard_log: Dict[str, Any] = {"ambiguity": a_result.get("guard_log",{}).get("ambiguity",[])}

    # Retrieval
    qdrant    = get_qdrant_client()
    vec       = embed(effective_query)
    qfilter   = _query_filter(tenant, req.filters)
    points    = dense_search(qdrant, req.collection_name, vec, top_k, filters=qfilter)
    chunks    = [p.payload for p in points]
    scores    = [p.score   for p in points]

    # Retrieval guard
    r_ctx    = {"query": effective_query, "retrieved_chunks": chunks,
                "collection_name": req.collection_name, "tenant_id": tenant or "default"}
    r_result = RetrievalGuard().run(r_ctx)
    guard_log["retrieval"] = r_result.get("guard_log",{}).get("retrieval",[])

    # System guard
    s_result  = SystemGuard().run({**r_ctx, "retrieved_chunks": chunks})
    chunks    = s_result.get("retrieved_chunks", chunks)
    guard_log["system"] = s_result.get("guard_log",{}).get("system",[])

    # Generation
    # Sentence-window chunks are matched on one sentence but answered from
    # the surrounding window, which is the point of that pattern.
    context = "\n\n".join((c.get("window_text") or c.get("text","")) for c in chunks)
    answer  = llm_call(
        f"Answer using only the provided context.\n\nContext:\n{context}\n\nQuestion: {req.query}",
        max_tokens=1024,
    )

    # Generation guard
    g_result = GenerationGuard().run({
        "query": req.query, "answer": answer, "retrieved_chunks": chunks,
        "tenant_id": tenant or "default",
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
        tenant_id=tenant,
        spec=req.spec,
        retrieval_strategy=strategy,
        top_k=top_k,
    )


_TENANT_STRATEGIES = {"filtered_rag", "multitenant_rag"}


# ---------------------------------------------------------------------------
# POST /query/decision — decision-grade answers (see rag_factory/decision_query.py)
# ---------------------------------------------------------------------------
@app.post("/query/decision", tags=["pipeline"])
def query_decision(req: DecisionQueryRequest, svc: ServiceContext = Depends(service_context)):
    """
    Answer from the corpus's current finalized versions only, with hybrid
    (dense + exact identifier) retrieval, mandatory citations, claim-level
    verification and abstention. Never rewrites the question.
    """
    from ..components import base
    from .. import decision_query as dq

    if not _COLLECTION_RE.fullmatch(req.collection_name):
        raise HTTPException(status_code=400, detail="invalid collection_name")
    tenant = _effective_tenant(svc, None)
    for key in (req.filters or {}):
        if key not in _QUERY_FILTER_KEYS:
            raise HTTPException(status_code=400, detail=f"filter key '{key}' is not allowed")
    out = dq.answer_question(
        qdrant=base.get_qdrant_client(), collection=req.collection_name, question=req.query,
        tenant=tenant, allowed_finalized_ids=req.allowed_finalized_ids, filters=req.filters,
        top_k=req.top_k, min_score=req.min_score, embed=base.embed, llm_call=base.llm_call,
        verify_with_judge=req.verify,
    )
    _log.info("decision query tenant=%s persona=%s coll=%s abstained=%s claims=%d",
              tenant, svc.persona_id, req.collection_name, out.get("abstained"), len(out.get("claims", [])))
    return out

# Payload keys a caller may filter on. tenant_id is NOT here: it comes only
# from the tenant (header in service mode), never from a caller filter.
_QUERY_FILTER_KEYS = {"doc_type", "doc_id", "finalized_document_id", "chunk_kind", "page_num", "region", "gxp"}


def _query_filter(tenant_id: Optional[str], filters: Optional[Dict[str, Any]]):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    must = []
    if tenant_id:
        must.append(FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)))
    for key, value in (filters or {}).items():
        if key not in _QUERY_FILTER_KEYS:
            raise HTTPException(status_code=400,
                                detail=f"filter key '{key}' is not allowed; use one of {sorted(_QUERY_FILTER_KEYS)}")
        if not isinstance(value, (str, int, bool)):
            raise HTTPException(status_code=400, detail=f"filter '{key}' must be a string, integer or boolean")
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=must) if must else None


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
def compare(req: CompareRequest, svc: ServiceContext = Depends(service_context)):
    from ..components.base import embed, dense_search, llm_call, get_qdrant_client
    from ..guards import GenerationGuard

    tenant = _effective_tenant(svc, None)

    def _run_spec(spec_name: str):
        spec   = _load_spec(spec_name)
        t0     = time.time()
        qdrant = get_qdrant_client()
        vec    = embed(req.query)
        k      = spec.retrieval.top_k
        if spec.retrieval.strategy in _TENANT_STRATEGIES and not tenant:
            raise HTTPException(status_code=400,
                                detail=f"Spec strategy '{spec.retrieval.strategy}' requires a tenant")
        points = dense_search(qdrant, req.collection_name, vec, k,
                              filters=_tenant_filter(tenant))
        chunks = [p.payload for p in points]
        ctx    = "\n\n".join(c.get("text","") for c in chunks)
        answer = llm_call(
            f"Context:\n{ctx}\n\nQuestion: {req.query}", max_tokens=1024
        )
        faith_result = GenerationGuard().run({
            "query": req.query, "answer": answer,
            "retrieved_chunks": chunks, "tenant_id": tenant or "default",
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
    svc               : ServiceContext = Depends(service_context),
):
    """
    Upload a single PDF/image and run the full IDP pipeline:
    extract → classify → skill → validate → incremental ingest.
    """
    import tempfile, os
    from fastapi import UploadFile
    from ..idp_pipeline import IDPPipeline

    tenant_id = _effective_tenant(svc, tenant_id)
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
            original_name     = file.filename or None,
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
    svc             : ServiceContext   = Depends(service_context),
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
    tenant_id = _effective_tenant(svc, tenant_id)
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
                original_name   = upload.filename or None,
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
def idp_collection_stats(collection_name: str,
                         svc: ServiceContext = Depends(service_context)):
    """
    Return incremental index stats for a collection:
    total chunks in Qdrant, indexed pages, last processed time.
    In service mode everything is restricted to the X-Tenant-Id tenant.
    """
    tenant = _effective_tenant(svc, None)
    from ..pdf_processor import IncrementalPDFProcessor
    from ..components.base import get_qdrant_client

    INDEX_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".index"
    )
    proc  = IncrementalPDFProcessor(index_dir=INDEX_DIR)
    stats = proc.get_index_stats(collection_name, tenant_id=tenant)

    # Count total Qdrant points
    total_chunks = 0
    doc_types: List[str] = []
    try:
        qdrant = get_qdrant_client()
        tfilter = _tenant_filter(tenant)
        if tfilter is not None:
            total_chunks = qdrant.count(collection_name, count_filter=tfilter, exact=True).count
        else:
            total_chunks = qdrant.get_collection(collection_name).points_count or 0
        # Sample payloads to find doc_types seen
        sample = qdrant.scroll(collection_name=collection_name, limit=100,
                               with_payload=True, scroll_filter=tfilter)
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
        doc_ids         = stats.get("doc_ids", []),
        doc_types_seen  = doc_types,
    )


