# -*- coding: utf-8 -*-
"""
Assembler — reads a PipelineSpec and chains components into a callable pipeline.
Guards are injected automatically at assembly time.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Type

from .spec import PipelineSpec, MANIFEST, VALIDATOR
from .components.base import BaseComponent

# Component name -> class mapping
_REGISTRY: Dict[str, Type[BaseComponent]] = {}


def _build_registry():
    from .components.chunking import (
        FixedChunking, SemanticChunking, HierarchicalChunking,
        ParentChildChunking, SentenceWindowChunking, ContextualChunking,
    )
    from .components.retrieval import (
        DenseRetrieval, HybridRRFRetrieval, HyDERetrieval,
        RerankedRetrieval, CompressedRetrieval,
        FilteredRetrieval, MultiDocRetrieval,
    )
    from .components.query import (
        QueryDecomposition, StepbackPrompting, FusionRetrieval,
        CoTRAG, ReActRAG,
    )
    from .components.agentic import (
        CorrectiveRAG, SelfRAG, IterativeRAG, RecursiveRAG, AgenticRAG,
    )
    from .components.memory import MemoryAugmentedRAG, MultiTurnRAG
    from .components.production import (
        EnsembleRAG, AdaptiveRAG, CachingRAG,
        EvaluationRAG, IncrementalRAG, MultiTenantRAG, FederatedRAG,
    )
    return {
        "fixed_chunking":          FixedChunking,
        "semantic_chunking":       SemanticChunking,
        "hierarchical_chunking":   HierarchicalChunking,
        "parent_child_chunking":   ParentChildChunking,
        "sentence_window_chunking":SentenceWindowChunking,
        "contextual_chunking":     ContextualChunking,
        "dense_retrieval":         DenseRetrieval,
        "hybrid_rrf_retrieval":    HybridRRFRetrieval,
        "hyde_retrieval":          HyDERetrieval,
        "reranked_retrieval":      RerankedRetrieval,
        "compressed_retrieval":    CompressedRetrieval,
        "filtered_retrieval":      FilteredRetrieval,
        "multi_doc_retrieval":     MultiDocRetrieval,
        "query_decomposition":     QueryDecomposition,
        "stepback_prompting":      StepbackPrompting,
        "fusion_retrieval":        FusionRetrieval,
        "cot_rag":                 CoTRAG,
        "react_rag":               ReActRAG,
        "corrective_rag":          CorrectiveRAG,
        "self_rag":                SelfRAG,
        "iterative_rag":           IterativeRAG,
        "recursive_rag":           RecursiveRAG,
        "agentic_rag":             AgenticRAG,
        "memory_augmented_rag":    MemoryAugmentedRAG,
        "multiturn_rag":           MultiTurnRAG,
        "ensemble_rag":            EnsembleRAG,
        "adaptive_rag":            AdaptiveRAG,
        "caching_rag":             CachingRAG,
        "evaluation_rag":          EvaluationRAG,
        "incremental_rag":         IncrementalRAG,
        "multitenant_rag":         MultiTenantRAG,
        "federated_rag":           FederatedRAG,
    }


class AssembledPipeline:
    """A validated, assembled pipeline ready to run."""

    def __init__(self, spec: PipelineSpec, steps: List[BaseComponent],
                 guards: List[BaseComponent]):
        self.spec   = spec
        self.steps  = steps
        self.guards = guards

    def run(self, query: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {
            "query":           query,
            "collection_name": self.spec.ingestion.collection_name,
            "top_k":           self.spec.retrieval.top_k,
            "filters":         self.spec.retrieval.filters,
            "max_sub_queries": self.spec.query.max_sub_queries,
        }
        if self.spec.ingestion.tenant_id:
            ctx["tenant_id"] = self.spec.ingestion.tenant_id
        if extra:
            ctx.update(extra)

        t0 = time.time()

        # Pre-retrieval guards (ambiguity, system)
        for g in self.guards:
            if g.SPEC.name in ("ambiguity_guard", "system_guard"):
                ctx.update(g.run(ctx))

        # Main component chain
        for step in self.steps:
            ctx.update(step.run(ctx))

        # Post-generation guards (retrieval, generation)
        for g in self.guards:
            if g.SPEC.name in ("retrieval_guard", "generation_guard"):
                ctx.update(g.run(ctx))

        ctx["_elapsed_ms"] = int((time.time() - t0) * 1000)
        return ctx


class Assembler:
    """Assembles a PipelineSpec into a runnable pipeline."""

    def __init__(self):
        global _REGISTRY
        if not _REGISTRY:
            _REGISTRY = _build_registry()

    def assemble(self, spec: PipelineSpec) -> AssembledPipeline:
        result = VALIDATOR.validate(spec)
        if not result.valid:
            raise ValueError(f"Invalid spec:\n{result}")

        steps:  List[BaseComponent] = []
        guards: List[BaseComponent] = []

        # 1. Chunker
        chunker_cls = _REGISTRY.get(spec.ingestion.chunker)
        if chunker_cls:
            steps.append(chunker_cls())

        # 2. Retrieval strategy
        retrieval_cls = _REGISTRY.get(spec.retrieval.strategy)
        if retrieval_cls:
            steps.append(retrieval_cls())

        # 3. Optional rerank
        if spec.retrieval.rerank:
            steps.append(_REGISTRY["reranked_retrieval"]())

        # 4. Optional compress
        if spec.retrieval.compress:
            steps.append(_REGISTRY["compressed_retrieval"]())

        # 5. Query strategy (if not direct)
        if spec.query.strategy != "direct":
            q_map = {
                "decompose": "query_decomposition",
                "stepback":  "stepback_prompting",
                "fusion":    "fusion_retrieval",
                "cot":       "cot_rag",
                "react":     "react_rag",
            }
            q_cls = _REGISTRY.get(q_map.get(spec.query.strategy, ""))
            if q_cls:
                steps.append(q_cls())

        # 6. Agentic mode
        if spec.generation.agentic_mode != "none":
            a_cls = _REGISTRY.get(spec.generation.agentic_mode)
            if a_cls:
                steps.append(a_cls())

        # 7. Guards (auto-injected)
        from .guards import (
            RetrievalGuard, GenerationGuard, AmbiguityGuard, SystemGuard
        )
        guard_map = {
            "retrieval_guard" : RetrievalGuard,
            "generation_guard": GenerationGuard,
            "ambiguity_guard" : AmbiguityGuard,
            "system_guard"    : SystemGuard,
        }
        for guard_name in spec.active_guards():
            g_cls = guard_map.get(guard_name)
            if g_cls:
                guards.append(g_cls())

        return AssembledPipeline(spec, steps, guards)

    def assemble_from_yaml(self, path: str) -> AssembledPipeline:
        return self.assemble(PipelineSpec.from_yaml(path))
