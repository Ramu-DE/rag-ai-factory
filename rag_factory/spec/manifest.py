# -*- coding: utf-8 -*-
"""
ComponentManifest — registry of every RAG component spec.
Single source of truth for the 33 patterns + 4 guard suites.
"""
from __future__ import annotations
from typing import Dict, List
from .component import BaseComponentSpec, ComponentRole

# ---------------------------------------------------------------------------
# Tier 1 — Chunking & Indexing
# ---------------------------------------------------------------------------
_TIER1: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="fixed_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/01_Simple_RAG.ipynb",
        description="Fixed-size token chunking — baseline pipeline",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count"],
        failure_modes=["FM-R2"],
    ),
    BaseComponentSpec(
        name="semantic_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/02_Semantic_Chunking.ipynb",
        description="Cosine breakpoint — variable-length semantically coherent chunks",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count"],
        timeout_secs=120,
        failure_modes=["FM-R2"],
    ),
    BaseComponentSpec(
        name="hierarchical_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/03_Hierarchical_RAG.ipynb",
        description="Multi-level chunk tree: summary -> detail",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count", "hierarchy_map"],
        timeout_secs=180,
    ),
    BaseComponentSpec(
        name="parent_child_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/04_Parent_Child_RAG.ipynb",
        description="Index child chunks; retrieve parent for context",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count", "parent_map"],
        timeout_secs=180,
    ),
    BaseComponentSpec(
        name="sentence_window_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/05_Sentence_Window_RAG.ipynb",
        description="Per-sentence index; expand +/-k window at retrieval",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count", "window_map"],
        failure_modes=["FM-R2", "FM-A4"],
        guards_applied=["FM-R2", "FM-A4"],
    ),
    BaseComponentSpec(
        name="contextual_chunking", role=ComponentRole.CHUNKER, tier=1,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/06_Contextual_Retrieval.ipynb",
        description="LLM context prefix before embedding — 67% fewer retrieval failures",
        input_schema=["raw_text"],
        output_schema=["chunks", "chunk_count", "context_map"],
        timeout_secs=300, is_async=True,
        failure_modes=["FM-R3"],
        guards_applied=["FM-R3"],
    ),
]

# ---------------------------------------------------------------------------
# Tier 2 — Retrieval Quality
# ---------------------------------------------------------------------------
_TIER2: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="dense_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier1_chunking_indexing/01_Simple_RAG.ipynb",
        description="Titan Embeddings V2 dense ANN via Qdrant",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores"],
        failure_modes=["FM-R1", "FM-R4", "FM-R5"],
    ),
    BaseComponentSpec(
        name="hybrid_rrf_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/07_Hybrid_Search.ipynb",
        description="Dense + BM25 RRF fusion — Recall@5 +17pp vs dense",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores", "rrf_scores"],
        failure_modes=["FM-R1"],
        guards_applied=["FM-R1"],
    ),
    BaseComponentSpec(
        name="hyde_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/08_HyDE.ipynb",
        description="Hypothetical Document Embeddings — query vocabulary bridge",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores", "hypothesis"],
        timeout_secs=90, is_async=True,
        failure_modes=["FM-R4"],
        guards_applied=["FM-R4"],
    ),
    BaseComponentSpec(
        name="reranked_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/09_Reranking.ipynb",
        description="LLM cross-encoder reranking of initial candidates",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["retrieved_chunks", "scores", "rerank_scores"],
        timeout_secs=120, is_async=True,
        failure_modes=["FM-G2"],
        guards_applied=["FM-G2"],
    ),
    BaseComponentSpec(
        name="compressed_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/10_Contextual_Compression.ipynb",
        description="Extract only relevant sentences from retrieved chunks",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["retrieved_chunks", "scores", "compression_ratio"],
        timeout_secs=90, is_async=True,
        failure_modes=["FM-R5"],
        guards_applied=["FM-R5"],
    ),
    BaseComponentSpec(
        name="filtered_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/11_Metadata_Filtering.ipynb",
        description="Pre-filter by structured metadata before ANN",
        input_schema=["query", "collection_name", "filters"],
        output_schema=["retrieved_chunks", "scores"],
        failure_modes=["FM-S3"],
    ),
    BaseComponentSpec(
        name="multi_doc_retrieval", role=ComponentRole.RETRIEVER, tier=2,
        notebook_ref="qdrant_notebooks/tier2_retrieval_quality/12_Multi_Document_RAG.ipynb",
        description="Source-aware retrieval across heterogeneous corpus",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores", "source_map"],
    ),
]

# ---------------------------------------------------------------------------
# Tier 3 — Query Handling
# ---------------------------------------------------------------------------
_TIER3: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="query_decomposition", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/13_Query_Decomposition.ipynb",
        description="Break complex queries into atomic sub-queries",
        input_schema=["query"],
        output_schema=["sub_queries", "query"],
        timeout_secs=60, is_async=True,
        failure_modes=["FM-A3"],
        guards_applied=["FM-A3"],
    ),
    BaseComponentSpec(
        name="stepback_prompting", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/14_Step_Back_Prompting.ipynb",
        description="Abstract query to higher-level principle before retrieval",
        input_schema=["query"],
        output_schema=["abstract_query", "query"],
        timeout_secs=60, is_async=True,
    ),
    BaseComponentSpec(
        name="fusion_retrieval", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/15_Fusion_Retrieval.ipynb",
        description="Generate 4 query variants; retrieve each; RRF merge",
        input_schema=["query", "collection_name"],
        output_schema=["retrieved_chunks", "scores", "query_variants"],
        timeout_secs=120, is_async=True,
        failure_modes=["FM-R1"],
        guards_applied=["FM-R1"],
    ),
    BaseComponentSpec(
        name="cot_rag", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/16_Chain_of_Thought_RAG.ipynb",
        description="Interleave reasoning steps with retrieval",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["answer", "reasoning_trace"],
        timeout_secs=120, is_async=True,
    ),
    BaseComponentSpec(
        name="react_rag", role=ComponentRole.QUERY_OPS, tier=3,
        notebook_ref="qdrant_notebooks/tier3_query_handling/17_ReAct_RAG.ipynb",
        description="Reasoning + Acting: LLM decides when to retrieve",
        input_schema=["query"],
        output_schema=["answer", "action_trace"],
        timeout_secs=240, is_async=True,
    ),
]

# ---------------------------------------------------------------------------
# Tier 4 — Agentic
# ---------------------------------------------------------------------------
_TIER4: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="corrective_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/18_Corrective_RAG.ipynb",
        description="Retrieval evaluator triggers correction or fallback",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["answer", "correction_log"],
        timeout_secs=180, is_async=True,
        failure_modes=["FM-S4"],
        guards_applied=["FM-S4"],
    ),
    BaseComponentSpec(
        name="self_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/19_Self_RAG.ipynb",
        description="4 reflection tokens: Retrieve / IsRel / IsSup / IsUse",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["answer", "reflection_tokens", "citation_scores"],
        timeout_secs=300, is_async=True,
        failure_modes=["FM-G1"],
        guards_applied=["FM-G1"],
    ),
    BaseComponentSpec(
        name="iterative_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/20_Iterative_RAG.ipynb",
        description="Multiple retrieval rounds guided by gap analysis",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "iteration_log"],
        timeout_secs=600, is_async=True,
    ),
    BaseComponentSpec(
        name="recursive_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/21_Recursive_RAG.ipynb",
        description="Recursively decompose until all sub-queries answerable",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "decomp_tree"],
        timeout_secs=600, is_async=True,
    ),
    BaseComponentSpec(
        name="agentic_rag", role=ComponentRole.GENERATOR, tier=4,
        notebook_ref="qdrant_notebooks/tier4_agentic/22_Agentic_RAG.ipynb",
        description="Autonomous agent with multiple retrieval tools",
        input_schema=["query"],
        output_schema=["answer", "tool_calls", "agent_trace"],
        timeout_secs=900, is_async=True,
    ),
]

# ---------------------------------------------------------------------------
# Tier 5 — Memory & Conversation
# ---------------------------------------------------------------------------
_TIER5: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="memory_augmented_rag", role=ComponentRole.MEMORY, tier=5,
        notebook_ref="qdrant_notebooks/tier5_memory_conversation/23_Memory_Augmented_RAG.ipynb",
        description="Short-term + long-term memory alongside document retrieval",
        input_schema=["query", "session_id"],
        output_schema=["answer", "memory_hits", "updated_memory"],
        timeout_secs=120, is_async=True,
    ),
    BaseComponentSpec(
        name="multiturn_rag", role=ComponentRole.MEMORY, tier=5,
        notebook_ref="qdrant_notebooks/tier5_memory_conversation/24_Multi_Turn_Conversational_RAG.ipynb",
        description="History-aware query rewriter for follow-up resolution",
        input_schema=["query", "conversation_history"],
        output_schema=["answer", "rewritten_query"],
        timeout_secs=90, is_async=True,
        failure_modes=["FM-A4"],
        guards_applied=["FM-A4"],
    ),
]

# ---------------------------------------------------------------------------
# Tier 6 — Ensemble & Meta
# ---------------------------------------------------------------------------
_TIER6: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="ensemble_rag", role=ComponentRole.ROUTER, tier=6,
        notebook_ref="qdrant_notebooks/tier6_ensemble_meta/25_Ensemble_RAG.ipynb",
        description="Run multiple retrieval strategies; vote/merge results",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "strategy_scores", "ensemble_log"],
        timeout_secs=300, is_async=True,
    ),
    BaseComponentSpec(
        name="adaptive_rag", role=ComponentRole.ROUTER, tier=6,
        notebook_ref="qdrant_notebooks/tier6_ensemble_meta/26_Adaptive_RAG.ipynb",
        description="Query classifier routes to optimal retrieval strategy",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "chosen_strategy", "routing_reason"],
        timeout_secs=120, is_async=True,
    ),
]

# ---------------------------------------------------------------------------
# Tier 7 — Production
# ---------------------------------------------------------------------------
_TIER7: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="streaming_rag", role=ComponentRole.GENERATOR, tier=7,
        notebook_ref="qdrant_notebooks/tier7_production/27_Streaming_RAG.ipynb",
        description="Stream LLM tokens — lowest TTFT",
        input_schema=["query", "retrieved_chunks"],
        output_schema=["answer_stream", "ttft_ms", "tokens_per_sec"],
        is_streaming=True,
    ),
    BaseComponentSpec(
        name="caching_rag", role=ComponentRole.RETRIEVER, tier=7,
        notebook_ref="qdrant_notebooks/tier7_production/28_Caching_RAG.ipynb",
        description="Semantic cache: near-duplicate query hits cache",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "cache_hit", "cache_latency_ms"],
    ),
    BaseComponentSpec(
        name="evaluation_rag", role=ComponentRole.EVALUATOR, tier=7,
        notebook_ref="qdrant_notebooks/tier7_production/29_Evaluation_RAG.ipynb",
        description="RAGAS 4-metric scoring on every pipeline run",
        input_schema=["query", "answer", "retrieved_chunks", "ground_truth"],
        output_schema=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        timeout_secs=180, is_async=True,
    ),
    BaseComponentSpec(
        name="complete_pipeline", role=ComponentRole.GENERATOR, tier=7,
        notebook_ref="qdrant_notebooks/tier7_production/30_Complete_Pipeline_RAG.ipynb",
        description="All Tier 1-7 components in one production pipeline",
        input_schema=["query", "collection_name"],
        output_schema=["answer", "pipeline_trace", "ragas_scores"],
        timeout_secs=600, is_async=True,
    ),
]

# ---------------------------------------------------------------------------
# Tier 8 — Incremental
# ---------------------------------------------------------------------------
_TIER8: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="incremental_rag", role=ComponentRole.CHUNKER, tier=8,
        notebook_ref="qdrant_notebooks/tier8_incremental/31_Incremental_RAG.ipynb",
        description="SHA-256 page manifest + diff engine — 90%+ embed savings",
        input_schema=["raw_text", "doc_id", "collection_name"],
        output_schema=["chunks", "chunk_count", "manifest", "pages_changed"],
        timeout_secs=300, is_async=True,
        failure_modes=["FM-S1", "FM-R6"],
        guards_applied=["FM-S1", "FM-R6"],
    ),
]

# ---------------------------------------------------------------------------
# Tier 9 — Multi-Tenant & Federated
# ---------------------------------------------------------------------------
_TIER9: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="multitenant_rag", role=ComponentRole.RETRIEVER, tier=9,
        notebook_ref="qdrant_notebooks/tier9_multi_tenant/32_Multi_Tenant_RAG.ipynb",
        description="Payload-based tenant isolation — FieldCondition on every query",
        input_schema=["query", "collection_name", "tenant_id"],
        output_schema=["retrieved_chunks", "scores"],
        failure_modes=["FM-S3"],
        guards_applied=["FM-S3"],
    ),
    BaseComponentSpec(
        name="federated_rag", role=ComponentRole.RETRIEVER, tier=9,
        notebook_ref="qdrant_notebooks/tier9_multi_tenant/33_Federated_RAG.ipynb",
        description="Parallel fan-out across collections + RRF merge + LLM router",
        input_schema=["query", "federation_config"],
        output_schema=["retrieved_chunks", "scores", "federate_scores"],
        timeout_secs=120, is_async=True,
    ),
]

# ---------------------------------------------------------------------------
# Guard suites — one per failure category
# ---------------------------------------------------------------------------
_GUARDS: List[BaseComponentSpec] = [
    BaseComponentSpec(
        name="retrieval_guard", role=ComponentRole.GUARD, tier=1,
        notebook_ref="research/failure_simulations/FM1_Retrieval_Failures.ipynb",
        description="FM-R1-R6: vocab gap / boundary / context / HyDE / K-dilution / stale",
        input_schema=["retrieved_chunks", "query", "collection_name"],
        output_schema=["retrieved_chunks", "guard_log"],
        guards_applied=["FM-R1","FM-R2","FM-R3","FM-R4","FM-R5","FM-R6"],
    ),
    BaseComponentSpec(
        name="generation_guard", role=ComponentRole.GUARD, tier=1,
        notebook_ref="research/failure_simulations/FM2_Generation_Failures.ipynb",
        description="FM-G1-G5: faithfulness / lost-in-middle / over-reliance / multi-hop / counterfactual",
        input_schema=["answer", "retrieved_chunks", "query"],
        output_schema=["answer", "guard_log", "faithfulness_score"],
        timeout_secs=90, is_async=True,
        guards_applied=["FM-G1","FM-G2","FM-G3","FM-G4","FM-G5"],
    ),
    BaseComponentSpec(
        name="ambiguity_guard", role=ComponentRole.GUARD, tier=1,
        notebook_ref="research/failure_simulations/FM3_Ambiguity_Failures.ipynb",
        description="FM-A1-A5: negation rewrite / temporal / multi-intent / coreference / scope",
        input_schema=["query"],
        output_schema=["query", "guard_log", "rewrite_applied"],
        timeout_secs=45, is_async=True,
        guards_applied=["FM-A1","FM-A2","FM-A3","FM-A4","FM-A5"],
    ),
    BaseComponentSpec(
        name="system_guard", role=ComponentRole.GUARD, tier=1,
        notebook_ref="research/failure_simulations/FM4_System_Failures.ipynb",
        description="FM-S1-S5: index drift / prompt injection / PII / cascade / fragmentation",
        input_schema=["retrieved_chunks", "query", "tenant_id"],
        output_schema=["retrieved_chunks", "query", "guard_log"],
        guards_applied=["FM-S1","FM-S2","FM-S3","FM-S4","FM-S5"],
    ),
]

ALL_SPECS: List[BaseComponentSpec] = (
    _TIER1 + _TIER2 + _TIER3 + _TIER4 +
    _TIER5 + _TIER6 + _TIER7 + _TIER8 +
    _TIER9 + _GUARDS
)


class ComponentManifest:
    """Registry of all RAG component specs."""

    def __init__(self, specs: List[BaseComponentSpec]):
        self._by_name: Dict[str, BaseComponentSpec] = {s.name: s for s in specs}
        self._by_role: Dict[str, List[BaseComponentSpec]] = {}
        self._by_tier: Dict[int, List[BaseComponentSpec]] = {}
        for s in specs:
            self._by_role.setdefault(s.role, []).append(s)
            self._by_tier.setdefault(s.tier, []).append(s)

    def get(self, name: str) -> BaseComponentSpec:
        if name not in self._by_name:
            available = sorted(self._by_name.keys())
            raise KeyError(f"Component '{name}' not in manifest. Available: {available}")
        return self._by_name[name]

    def by_role(self, role: ComponentRole) -> List[BaseComponentSpec]:
        return self._by_role.get(role, [])

    def by_tier(self, tier: int) -> List[BaseComponentSpec]:
        return self._by_tier.get(tier, [])

    def covering_fm(self, fm_code: str) -> List[BaseComponentSpec]:
        return [s for s in self._by_name.values() if fm_code in s.guards_applied]

    def async_specs(self) -> List[BaseComponentSpec]:
        return [s for s in self._by_name.values() if s.is_async]

    def summary(self) -> str:
        lines = ["ComponentManifest"]
        for tier in sorted(self._by_tier):
            names = [s.name for s in self._by_tier[tier]]
            lines.append(f"  Tier {tier}: {names}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._by_name)


# Singleton — import this everywhere
MANIFEST = ComponentManifest(ALL_SPECS)
