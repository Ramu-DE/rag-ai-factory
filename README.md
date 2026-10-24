# AI RAG Factory

**NVIDIA-inspired · Temporal-ready · AWS Bedrock + Qdrant**

A spec-first, composable pipeline engine over **33 RAG patterns** across 9 tiers.
Declare a pipeline in YAML — the factory assembles, guards, and runs it.

---

## Architecture

```
specs/production.yaml
        |
        v
  PipelineSpec  (Pydantic validated)
        |
        v
   Assembler  ->  [Chunker] -> [Retriever] -> [QueryOps] -> [Generator]
        |                  guards injected automatically
        v
  AssembledPipeline.run(query)
        |
        v
   ctx dict  {answer, retrieved_chunks, guard_log, ragas_scores, ...}
```

**Temporal** handles long-running agentic loops and ingestion workflows.
**Guards** (FM-R/G/A/S) are auto-injected from the failure-mode research.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # add AWS + Qdrant credentials

# Run a query against the simple pipeline
python -m rag_factory.cli run --spec specs/simple.yaml \
  --query "What are the Medicaid eligibility requirements?"

# Run the production pipeline (Temporal disabled for local test)
python -m rag_factory.cli run --spec specs/production.yaml \
  --query "Summarise the key coverage categories"
```

---

## Pipeline specs

| Spec | Chunker | Retrieval | Agentic | Guards | Temporal |
|---|---|---|---|---|---|
| `simple.yaml` | fixed | dense | none | R+G | off |
| `production.yaml` | contextual | hybrid_rrf + rerank | self_rag | all | on |
| `agentic.yaml` | sentence_window | fusion + rerank | iterative_rag | R+G+A | on |
| `multitenant.yaml` | fixed | multitenant | none | G+S | on |

---

## Failure mode guards (auto-injected)

| Guard | Failure modes covered |
|---|---|
| `retrieval_guard` | FM-R1 vocab gap, FM-R5 K-dilution, FM-R6 stale index |
| `generation_guard` | FM-G1 hallucination, FM-G2 lost-in-middle |
| `ambiguity_guard` | FM-A1 negation rewrite, FM-A3 multi-intent, FM-A5 global scope |
| `system_guard` | FM-S2 prompt injection, FM-S3 PII detection |

---

## Temporal workflows

| Workflow | Use case |
|---|---|
| `IngestWorkflow` | Durable ingestion — survives AWS token expiry |
| `AgenticQueryWorkflow` | Long-running iterative/recursive/agentic RAG |
| `ScheduledReindexWorkflow` | Cron re-index for stale content detection |

```bash
pip install temporalio
python -m rag_factory.temporal.worker  # start worker
```

---

## Component registry (33 patterns)

| Tier | Patterns |
|---|---|
| 1 Chunking | fixed, semantic, hierarchical, parent_child, sentence_window, contextual |
| 2 Retrieval | dense, hybrid_rrf, hyde, reranked, compressed, filtered, multi_doc |
| 3 Query | decompose, stepback, fusion, cot, react |
| 4 Agentic | corrective, self_rag, iterative, recursive, agentic |
| 5 Memory | memory_augmented, multiturn |
| 6 Ensemble | ensemble, adaptive |
| 7 Production | streaming, caching, evaluation, complete_pipeline |
| 8 Incremental | incremental |
| 9 Multi-Tenant | multitenant, federated |

---

## Qdrant configuration

`QDRANT_URL` and `QDRANT_API_KEY` are required. The factory no longer falls
back silently to an in-memory store; set `RAG_FACTORY_ALLOW_MEMORY=1` to opt
into a throwaway in-memory Qdrant (tests / demos only — a warning is logged).

Incremental ingestion keeps one hash index per document:
`.index/<collection>__<sha256(doc_id)[:16]>_index.json` (scope is
`<tenant_id>/<doc_id>` when a tenant is given), so documents sharing a
collection never overwrite each other. A legacy `<collection>_index.json`
is migrated the first time the document it describes is re-processed.

---

## Service mode (behind the AI Control Plane)

The FastAPI app (`uvicorn rag_factory.api.main:app`) can run as a **private
sidecar** reached only server-to-server by the AI Control Plane. It is
enabled by setting `SERVICE_TOKEN`; with it unset the API behaves as before.

| Env var | Meaning |
|---|---|
| `SERVICE_TOKEN` | Shared secret. Non-empty turns service mode on. |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (default: none, i.e. no CORS). |
| `QDRANT_URL`, `QDRANT_API_KEY` | Required (see above). |

Header contract — required on every route except `GET /health`:

| Header | Rule | On failure |
|---|---|---|
| `X-Service-Token` | equals `SERVICE_TOKEN` (constant-time compare) | 401 |
| `X-Tenant-Id` | `^[a-z0-9][a-z0-9-]{0,62}$` | 400 |
| `X-Persona-Id` | `^[A-Za-z0-9_-]{1,64}$` | 400 |

In service mode:

- The `X-Tenant-Id` tenant **overrides** any `tenant_id` in request bodies or
  forms. `/ingest`, `/ingest/finalized` and `/idp/*` stamp it on every point;
  `/query` and `/compare` must-filter every Qdrant search on `tenant_id`;
  `/idp/collections/{name}/stats` only reports that tenant's data.
- `/docs`, `/redoc` and `/openapi.json` are disabled; `spec` names must be
  plain file names inside `specs/`.
- Do not publish the port (e.g. through CloudFront); bind it to the private
  network only.

`/query` honours a requested spec's `retrieval.top_k`, and a
`filtered_rag` / `multitenant_rag` strategy makes the tenant filter mandatory.
Retrieval itself is still dense top-k: the full Assembler path (spec chunker,
rerank, agentic modes) is Phase 2.

### `POST /ingest/finalized`

Consumes a `FinalizedDocument v1` (control-plane contract
`finalized-document.v1.schema.json`). Target collection: query parameter
`?collection_name=...` or an extra body key `collection_name`.

| Check | Status |
|---|---|
| missing `collection_name` | 400 |
| `schemaVersion != "finalized-document/v1"`, missing required keys, bad ids / shapes | 422 |
| service mode and `tenantId != X-Tenant-Id` | 403 |
| `contentSha256 != sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())` | 409 |
| a newer `version` of the same `documentId` already ingested for the tenant | 409 |

`content.pages` (or `content.text` when there are no pages) is fixed-chunked
and embedded (Titan v2, 1024-d). Each point's payload carries `text, doc_id
(=documentId), finalized_document_id, version, tenant_id, doc_type, gxp,
region, page_num (when pages are given), chunk_index, source_sha256,
content_sha256` and `fields` (name → value, values truncated to 256 chars).
Point ids are deterministic from (tenant, finalizedDocumentId, version,
chunk_index): re-sending the same version replaces it; a new version of the
same `documentId` deletes the previous versions' points in that collection.

**Chunker per corpus.** `?chunker=` (or a body key `chunker`, not part of the
hash) selects how the document is chunked; the AI Control Plane chooses it once
per corpus from the corpus profile:

| `chunker` | Text chunks | Extra chunks | Query-time effect |
|---|---|---|---|
| `fixed_chunking` (default) | 500 chars, 50 overlap | — | — |
| `semantic_chunking` | one per sentence (> 20 chars) | — | — |
| `sentence_window_chunking` | one per sentence, with `window_text` (±3 sentences) | — | `/query` answers from `window_text` |
| `structured_chunking` | 500/50 | one per 40 reviewed fields, one per 20 table rows (header repeated) | filter with `chunk_kind` |

Every payload carries `chunk_kind` (`text`, `fields`, `table`) and `chunker`.
Re-ingesting the same version with another chunker replaces its points.
Keyword payload indexes are created on `tenant_id, doc_id,
finalized_document_id, doc_type, chunk_kind`.

`POST /query` also accepts `filters` (equality on `doc_type, doc_id,
finalized_document_id, chunk_kind, page_num, region, gxp`). `tenant_id` is not
accepted as a filter; the tenant always comes from the header.

Response:

```json
{"collection_name": "kb_acme", "tenant_id": "acme", "document_id": "doc-inv1",
 "finalized_document_id": "fd-abc2", "version": 2, "content_sha256": "…",
 "chunk_count": 8, "points_upserted": 8, "points_deleted": 10,
 "replaced_same_version": false, "superseded_versions": [1], "elapsed_ms": 412}
```

Tests (offline: in-memory Qdrant, fake embeddings):

```bash
pip install -r requirements.txt pytest
python -m pytest tests
```

---

## Source notebooks

All 33 patterns originate from:
**[rag-patterns-aws-Qdrant](https://github.com/Ramu-DE/rag-patterns-aws-Qdrant)**
(existing notebooks untouched — this repo is the factory layer on top)
