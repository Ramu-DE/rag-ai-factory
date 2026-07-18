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

## Source notebooks

All 33 patterns originate from:
**[rag-patterns-aws-Qdrant](https://github.com/Ramu-DE/rag-patterns-aws-Qdrant)**
(existing notebooks untouched — this repo is the factory layer on top)
