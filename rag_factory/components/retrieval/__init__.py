# -*- coding: utf-8 -*-
"""Tier 2 — Retrieval Quality component wrappers."""
from __future__ import annotations
from typing import Any, Dict, List
from ..base import BaseComponent, embed, dense_search, llm_call, get_qdrant_client
from ...spec.manifest import MANIFEST


def _rrf(ranked_lists: List[List], k: int = 60) -> List:
    """Reciprocal Rank Fusion over multiple ranked point lists."""
    scores: Dict[str, float] = {}
    id_to_point: Dict[str, Any] = {}
    for ranked in ranked_lists:
        for rank, pt in enumerate(ranked):
            pid = str(pt.id)
            scores[pid]     = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            id_to_point[pid] = pt
    merged = sorted(id_to_point.values(),
                    key=lambda p: scores[str(p.id)], reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------
class DenseRetrieval(BaseComponent):
    SPEC = MANIFEST.get("dense_retrieval")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        vec    = embed(ctx["query"])
        k      = ctx.get("top_k", 5)
        points = dense_search(get_qdrant_client(), ctx["collection_name"], vec, k)
        return {
            "retrieved_chunks": [p.payload for p in points],
            "scores":           [p.score   for p in points],
        }


# ---------------------------------------------------------------------------
# Hybrid RRF
# ---------------------------------------------------------------------------
class HybridRRFRetrieval(BaseComponent):
    SPEC = MANIFEST.get("hybrid_rrf_retrieval")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        from rank_bm25 import BM25Okapi

        query      = ctx["query"]
        cname      = ctx["collection_name"]
        k          = ctx.get("top_k", 5)
        qdrant     = get_qdrant_client()

        # Dense pass
        vec          = embed(query)
        dense_points = dense_search(qdrant, cname, vec, k * 3)

        # BM25 pass over retrieved corpus
        corpus       = [p.payload.get("text", "") for p in dense_points]
        tokenized    = [t.lower().split() for t in corpus]
        bm25         = BM25Okapi(tokenized)
        bm25_scores  = bm25.get_scores(query.lower().split())
        bm25_ranked  = [dense_points[i]
                        for i in sorted(range(len(bm25_scores)),
                                        key=lambda x: bm25_scores[x], reverse=True)]

        merged = _rrf([dense_points, bm25_ranked])[:k]
        return {
            "retrieved_chunks": [p.payload for p in merged],
            "scores":           [p.score   for p in merged],
            "rrf_scores":       [1.0 / (60 + i + 1) for i in range(len(merged))],
        }


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------
class HyDERetrieval(BaseComponent):
    SPEC = MANIFEST.get("hyde_retrieval")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        query = ctx["query"]
        k     = ctx.get("top_k", 5)

        hypothesis = llm_call(
            f"Write a short passage that would directly answer: {query}",
            max_tokens=200,
        )
        vec    = embed(hypothesis)
        points = dense_search(get_qdrant_client(), ctx["collection_name"], vec, k)
        return {
            "retrieved_chunks": [p.payload for p in points],
            "scores":           [p.score   for p in points],
            "hypothesis":       hypothesis,
        }


# ---------------------------------------------------------------------------
# Reranked retrieval
# ---------------------------------------------------------------------------
class RerankedRetrieval(BaseComponent):
    SPEC = MANIFEST.get("reranked_retrieval")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "retrieved_chunks")
        query  = ctx["query"]
        chunks = ctx["retrieved_chunks"]
        k      = ctx.get("top_k", 5)

        scored = []
        for ch in chunks:
            text  = ch.get("text", "")
            score = llm_call(
                f"Query: {query}\nPassage: {text}\n\n"
                "Rate relevance 0.0-1.0 (one number only):",
                max_tokens=5,
            )
            try:
                s = float(score.strip())
            except ValueError:
                s = 0.0
            scored.append((s, ch))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        return {
            "retrieved_chunks": [c for _, c in top],
            "scores":           [s for s, _ in top],
            "rerank_scores":    [s for s, _ in top],
        }


# ---------------------------------------------------------------------------
# Contextual compression
# ---------------------------------------------------------------------------
class CompressedRetrieval(BaseComponent):
    SPEC = MANIFEST.get("compressed_retrieval")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "retrieved_chunks")
        query  = ctx["query"]
        chunks = ctx["retrieved_chunks"]

        compressed = []
        original_len = sum(len(c.get("text","")) for c in chunks)

        for ch in chunks:
            text = ch.get("text", "")
            extracted = llm_call(
                f"Query: {query}\n\nFrom the passage below, extract ONLY the "
                "sentences directly relevant to the query. Return empty string "
                "if none are relevant.\n\nPassage:\n" + text,
                max_tokens=300,
            ).strip()
            if extracted:
                new_ch = dict(ch)
                new_ch["text"] = extracted
                compressed.append(new_ch)

        compressed_len = sum(len(c.get("text","")) for c in compressed)
        ratio = compressed_len / original_len if original_len > 0 else 1.0

        return {
            "retrieved_chunks": compressed,
            "scores": ctx.get("scores", []),
            "compression_ratio": round(ratio, 3),
        }


# ---------------------------------------------------------------------------
# Filtered retrieval (metadata pre-filter)
# ---------------------------------------------------------------------------
class FilteredRetrieval(BaseComponent):
    SPEC = MANIFEST.get("filtered_retrieval")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query   = ctx["query"]
        cname   = ctx["collection_name"]
        k       = ctx.get("top_k", 5)
        filters = ctx.get("filters", {})
        qdrant  = get_qdrant_client()
        vec     = embed(query)

        qfilter = None
        if filters:
            conditions = [
                FieldCondition(key=fk, match=MatchValue(value=fv))
                for fk, fv in filters.items()
            ]
            qfilter = Filter(must=conditions)

        points = dense_search(qdrant, cname, vec, k, filters=qfilter)
        return {
            "retrieved_chunks": [p.payload for p in points],
            "scores":           [p.score   for p in points],
        }


# ---------------------------------------------------------------------------
# Multi-document retrieval
# ---------------------------------------------------------------------------
class MultiDocRetrieval(BaseComponent):
    SPEC = MANIFEST.get("multi_doc_retrieval")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        query  = ctx["query"]
        cname  = ctx["collection_name"]
        k      = ctx.get("top_k", 5)
        qdrant = get_qdrant_client()
        vec    = embed(query)
        points = dense_search(qdrant, cname, vec, k * 2)

        source_map: Dict[str, List[Dict]] = {}
        for p in points:
            src = p.payload.get("source", "unknown")
            source_map.setdefault(src, []).append(p.payload)

        # Take top-k across all sources (already sorted by score)
        top = points[:k]
        return {
            "retrieved_chunks": [p.payload for p in top],
            "scores":           [p.score   for p in top],
            "source_map":       source_map,
        }


__all__ = [
    "DenseRetrieval", "HybridRRFRetrieval", "HyDERetrieval",
    "RerankedRetrieval", "CompressedRetrieval",
    "FilteredRetrieval", "MultiDocRetrieval",
]
