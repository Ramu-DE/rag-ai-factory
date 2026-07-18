# -*- coding: utf-8 -*-
"""Tier 3 — Query Handling component wrappers."""
from __future__ import annotations
from typing import Any, Dict, List
from ..base import BaseComponent, embed, dense_search, llm_call, get_qdrant_client
from ...spec.manifest import MANIFEST


def _rrf(lists: List[List], k: int = 60) -> List:
    scores: Dict[str, float] = {}
    id_map: Dict[str, Any] = {}
    for ranked in lists:
        for rank, pt in enumerate(ranked):
            pid = str(pt.id)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            id_map[pid]  = pt
    return sorted(id_map.values(), key=lambda p: scores[str(p.id)], reverse=True)


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------
class QueryDecomposition(BaseComponent):
    SPEC = MANIFEST.get("query_decomposition")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query")
        max_sub = ctx.get("max_sub_queries", 4)
        raw = llm_call(
            f"Break this query into at most {max_sub} independent sub-queries. "
            "Return ONLY a numbered list, one per line.\n\nQuery: " + ctx["query"],
            max_tokens=200,
        )
        sub_queries = []
        for line in raw.strip().splitlines():
            line = line.strip().lstrip("0123456789.-) ").strip()
            if line:
                sub_queries.append(line)
        return {"sub_queries": sub_queries, "query": ctx["query"]}


# ---------------------------------------------------------------------------
# Step-back prompting
# ---------------------------------------------------------------------------
class StepbackPrompting(BaseComponent):
    SPEC = MANIFEST.get("stepback_prompting")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query")
        abstract = llm_call(
            "Rewrite this specific question as a more general principle-level question "
            "that would help retrieve foundational background knowledge.\n\n"
            "Original: " + ctx["query"] + "\n\nGeneral version:",
            max_tokens=100,
        ).strip()
        return {"abstract_query": abstract, "query": ctx["query"]}


# ---------------------------------------------------------------------------
# Fusion retrieval
# ---------------------------------------------------------------------------
class FusionRetrieval(BaseComponent):
    SPEC = MANIFEST.get("fusion_retrieval")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        query  = ctx["query"]
        cname  = ctx["collection_name"]
        k      = ctx.get("top_k", 5)
        qdrant = get_qdrant_client()

        variants_raw = llm_call(
            f"Generate 3 alternative phrasings of this query. "
            "Return ONLY the variants, one per line.\n\nQuery: {query}",
            max_tokens=150,
        )
        variants = [query] + [
            l.strip().lstrip("0123456789.-) ")
            for l in variants_raw.strip().splitlines() if l.strip()
        ][:3]

        all_results = []
        for v in variants:
            vec    = embed(v)
            points = dense_search(qdrant, cname, vec, k)
            all_results.append(points)

        merged = _rrf(all_results)[:k]
        return {
            "retrieved_chunks": [p.payload for p in merged],
            "scores":           [p.score   for p in merged],
            "query_variants":   variants,
        }


# ---------------------------------------------------------------------------
# Chain-of-Thought RAG
# ---------------------------------------------------------------------------
class CoTRAG(BaseComponent):
    SPEC = MANIFEST.get("cot_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "retrieved_chunks")
        context = "\n\n".join(
            c.get("text", "") for c in ctx["retrieved_chunks"]
        )
        answer = llm_call(
            f"Answer the question step by step using only the context provided.\n\n"
            f"Context:\n{context}\n\nQuestion: {ctx['query']}\n\nStep-by-step reasoning:",
            max_tokens=1024,
        )
        # Extract reasoning trace (steps) and final answer
        lines  = answer.strip().splitlines()
        steps  = [l for l in lines if l.strip().startswith(("Step","1.","2.","3.","-"))]
        return {"answer": answer, "reasoning_trace": steps}


# ---------------------------------------------------------------------------
# ReAct RAG
# ---------------------------------------------------------------------------
class ReActRAG(BaseComponent):
    SPEC = MANIFEST.get("react_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query")
        query  = ctx["query"]
        cname  = ctx.get("collection_name", "")
        qdrant = get_qdrant_client()

        action_trace: List[Dict] = []
        accumulated  = []

        for step in range(4):   # max 4 iterations
            thought = llm_call(
                f"Question: {query}\n"
                f"Previous observations: {accumulated}\n\n"
                "Thought: What do I need to find next? "
                "(Reply 'DONE' if you can answer now, else state what to search for.)",
                max_tokens=100,
            ).strip()

            action_trace.append({"step": step, "thought": thought})

            if "DONE" in thought.upper() or not cname:
                break

            search_query = llm_call(
                f"Convert this thought into a search query (one line): {thought}",
                max_tokens=50,
            ).strip()
            vec    = embed(search_query)
            points = dense_search(qdrant, cname, vec, 3)
            obs    = [p.payload.get("text","")[:200] for p in points]
            accumulated.extend(obs)
            action_trace[-1]["action"]      = search_query
            action_trace[-1]["observation"] = obs

        context = "\n".join(accumulated)
        answer  = llm_call(
            f"Question: {query}\n\nContext gathered:\n{context}\n\nFinal answer:",
            max_tokens=512,
        )
        return {"answer": answer, "action_trace": action_trace}


__all__ = [
    "QueryDecomposition", "StepbackPrompting", "FusionRetrieval",
    "CoTRAG", "ReActRAG",
]
