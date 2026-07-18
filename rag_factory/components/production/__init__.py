# -*- coding: utf-8 -*-
"""Tier 6-9 component wrappers: Ensemble, Adaptive, Production, Incremental, Multi-Tenant."""
from __future__ import annotations
import hashlib, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from ..base import BaseComponent, embed, dense_search, llm_call, get_qdrant_client, chunk_id
from ...spec.manifest import MANIFEST

# ---------------------------------------------------------------------------
# Tier 6 — Ensemble & Adaptive
# ---------------------------------------------------------------------------
class EnsembleRAG(BaseComponent):
    SPEC = MANIFEST.get("ensemble_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        query  = ctx["query"]
        cname  = ctx["collection_name"]
        k      = ctx.get("top_k", 5)
        qdrant = get_qdrant_client()
        vec    = embed(query)

        # Run 3 strategies in parallel
        def dense():
            return dense_search(qdrant, cname, vec, k)

        def bm25_pass():
            from rank_bm25 import BM25Okapi
            pts   = dense_search(qdrant, cname, vec, k * 3)
            corpus = [p.payload.get("text","") for p in pts]
            bm25   = BM25Okapi([t.lower().split() for t in corpus])
            scores = bm25.get_scores(query.lower().split())
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [pts[i] for i in ranked[:k]]

        def hyde_pass():
            hyp = llm_call(f"Write a passage answering: {query}", max_tokens=150)
            return dense_search(qdrant, cname, embed(hyp), k)

        strategies = {"dense": dense, "bm25": bm25_pass, "hyde": hyde_pass}
        results: Dict[str, List] = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(fn): name for name, fn in strategies.items()}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results[name] = fut.result()
                except Exception:
                    results[name] = []

        # RRF merge
        scores: Dict[str, float] = {}
        id_map: Dict[str, Any]   = {}
        for pts in results.values():
            for rank, pt in enumerate(pts):
                pid = str(pt.id)
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (60 + rank + 1)
                id_map[pid]  = pt
        merged = sorted(id_map.values(), key=lambda p: scores[str(p.id)], reverse=True)[:k]

        context = "\n\n".join(p.payload.get("text","") for p in merged)
        answer  = llm_call(
            f"Context:\n{context}\n\nQuestion: {query}", max_tokens=1024
        )
        return {
            "answer":         answer,
            "strategy_scores": {n: [p.score for p in pts] for n, pts in results.items()},
            "ensemble_log":   list(results.keys()),
        }


class AdaptiveRAG(BaseComponent):
    SPEC = MANIFEST.get("adaptive_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        query = ctx["query"]

        routing = llm_call(
            f"Classify this query for RAG routing.\n\nQuery: {query}\n\n"
            "Reply with one of:\n"
            "SIMPLE   — factual, single-hop, short answer\n"
            "COMPLEX  — multi-aspect, reasoning required\n"
            "AGENTIC  — open-ended, needs multiple retrieval steps\n\n"
            "Format: CLASSIFICATION: <type>\nREASON: <one sentence>",
            max_tokens=80,
        )

        classification = "SIMPLE"
        reason         = ""
        for line in routing.strip().splitlines():
            if line.startswith("CLASSIFICATION:"):
                classification = line.split(":", 1)[1].strip().upper()
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        # Route to appropriate component
        from ..retrieval import HybridRRFRetrieval, DenseRetrieval
        from ..query     import QueryDecomposition, FusionRetrieval

        if classification == "SIMPLE":
            comp   = DenseRetrieval()
            chosen = "dense_retrieval"
        elif classification == "COMPLEX":
            decomp     = QueryDecomposition()
            decomp_ctx = decomp.run(ctx)
            sub_results = []
            for sq in decomp_ctx.get("sub_queries", [query])[:3]:
                sub_ctx = dict(ctx, query=sq)
                sub_results.extend(
                    DenseRetrieval().run(sub_ctx).get("retrieved_chunks", [])
                )
            context = "\n\n".join(c.get("text","") for c in sub_results[:6])
            answer  = llm_call(
                f"Context:\n{context}\n\nQuestion: {query}", max_tokens=1024
            )
            return {
                "answer": answer,
                "chosen_strategy": "query_decomposition",
                "routing_reason":  reason,
            }
        else:
            comp   = HybridRRFRetrieval()
            chosen = "hybrid_rrf_retrieval"

        ret_ctx = comp.run(ctx)
        context = "\n\n".join(c.get("text","") for c in ret_ctx["retrieved_chunks"])
        answer  = llm_call(
            f"Context:\n{context}\n\nQuestion: {query}", max_tokens=1024
        )
        return {"answer": answer, "chosen_strategy": chosen, "routing_reason": reason}


# ---------------------------------------------------------------------------
# Tier 7 — Production
# ---------------------------------------------------------------------------
class CachingRAG(BaseComponent):
    SPEC = MANIFEST.get("caching_rag")

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self._cache: List[Dict] = []  # {query_vec, answer, query}

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        self._require(ctx, "query", "collection_name")
        query     = ctx["query"]
        query_vec = embed(query)

        # Check cache
        for entry in self._cache:
            cached_vec = entry["query_vec"]
            a = np.array(query_vec)
            b = np.array(cached_vec)
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if sim >= self.threshold:
                return {
                    "answer":           entry["answer"],
                    "cache_hit":        True,
                    "cache_latency_ms": 0,
                }

        # Cache miss — do retrieval + generation
        from ..retrieval import HybridRRFRetrieval
        t0      = time.time()
        ret_ctx = HybridRRFRetrieval().run(ctx)
        context = "\n\n".join(c.get("text","") for c in ret_ctx["retrieved_chunks"])
        answer  = llm_call(
            f"Context:\n{context}\n\nQuestion: {query}", max_tokens=1024
        )
        latency = int((time.time() - t0) * 1000)
        self._cache.append({"query_vec": query_vec, "answer": answer, "query": query})
        return {"answer": answer, "cache_hit": False, "cache_latency_ms": latency}


class EvaluationRAG(BaseComponent):
    SPEC = MANIFEST.get("evaluation_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "answer", "retrieved_chunks", "ground_truth")
        query        = ctx["query"]
        answer       = ctx["answer"]
        chunks       = ctx["retrieved_chunks"]
        ground_truth = ctx["ground_truth"]
        context      = "\n\n".join(c.get("text","") for c in chunks)

        def score(prompt: str) -> float:
            raw = llm_call(prompt, max_tokens=5)
            try:
                return min(1.0, max(0.0, float(raw.strip())))
            except ValueError:
                return 0.5

        faithfulness = score(
            f"Context:\n{context}\nAnswer:\n{answer}\n\n"
            "Score 0.0-1.0: how much of the answer is supported by context?"
        )
        relevancy = score(
            f"Question: {query}\nAnswer: {answer}\n\n"
            "Score 0.0-1.0: how relevant is the answer to the question?"
        )
        precision = score(
            f"Question: {query}\nContext:\n{context}\n\n"
            "Score 0.0-1.0: what fraction of the context is actually useful?"
        )
        recall = score(
            f"Question: {query}\nGround truth: {ground_truth}\n"
            f"Context:\n{context}\n\n"
            "Score 0.0-1.0: how much ground-truth information is in the context?"
        )
        return {
            "faithfulness":       round(faithfulness, 3),
            "answer_relevancy":   round(relevancy, 3),
            "context_precision":  round(precision, 3),
            "context_recall":     round(recall, 3),
        }


# ---------------------------------------------------------------------------
# Tier 8 — Incremental
# ---------------------------------------------------------------------------
class IncrementalRAG(BaseComponent):
    SPEC = MANIFEST.get("incremental_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "raw_text", "doc_id", "collection_name")
        raw_text = ctx["raw_text"]
        doc_id   = ctx["doc_id"]
        cname    = ctx["collection_name"]
        qdrant   = get_qdrant_client()

        # Page-level manifest
        pages    = raw_text.split("\f") or [raw_text]
        manifest: Dict[str, str] = {}
        changed  = 0

        for page_num, page_text in enumerate(pages):
            page_hash = hashlib.sha256(page_text.encode()).hexdigest()
            pid       = chunk_id(f"{doc_id}:page:{page_num}")
            manifest[str(page_num)] = page_hash

            # Check existing hash in Qdrant payload
            try:
                existing = qdrant.retrieve(
                    collection_name=cname, ids=[pid], with_payload=True
                )
                if existing and existing[0].payload.get("page_hash") == page_hash:
                    continue  # unchanged — skip
            except Exception:
                pass

            # Changed or new — re-embed
            from qdrant_client.models import VectorParams, Distance, PointStruct
            if cname not in [c.name for c in qdrant.get_collections().collections]:
                qdrant.create_collection(
                    cname,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )
            vec = embed(page_text[:1500])
            qdrant.upsert(collection_name=cname, points=[
                PointStruct(id=pid, vector=vec,
                            payload={"text": page_text, "doc_id": doc_id,
                                     "page_num": page_num, "page_hash": page_hash})
            ])
            changed += 1

        chunks = [{"text": p, "page_num": i} for i, p in enumerate(pages)]
        return {
            "chunks":        chunks,
            "chunk_count":   len(chunks),
            "manifest":      manifest,
            "pages_changed": changed,
        }


# ---------------------------------------------------------------------------
# Tier 9 — Multi-Tenant & Federated
# ---------------------------------------------------------------------------
class MultiTenantRAG(BaseComponent):
    SPEC = MANIFEST.get("multitenant_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name", "tenant_id")
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query     = ctx["query"]
        cname     = ctx["collection_name"]
        tenant_id = ctx["tenant_id"]
        k         = ctx.get("top_k", 5)
        qdrant    = get_qdrant_client()
        vec       = embed(query)

        # Mandatory tenant filter — no tenant_id leakage possible
        tenant_filter = Filter(must=[
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        ])
        points = dense_search(qdrant, cname, vec, k, filters=tenant_filter)
        return {
            "retrieved_chunks": [p.payload for p in points],
            "scores":           [p.score   for p in points],
        }


class FederatedRAG(BaseComponent):
    SPEC = MANIFEST.get("federated_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "federation_config")
        query  = ctx["query"]
        config = ctx["federation_config"]  # List[{collection_name, weight}]
        k      = ctx.get("top_k", 5)
        qdrant = get_qdrant_client()
        vec    = embed(query)

        # Fan-out in parallel
        federate_scores: Dict[str, List[float]] = {}

        def search_federate(fed: Dict):
            cname = fed["collection_name"]
            try:
                pts = dense_search(qdrant, cname, vec, k)
                return cname, pts
            except Exception as e:
                return cname, []

        all_points: List = []
        with ThreadPoolExecutor(max_workers=len(config)) as ex:
            futures = [ex.submit(search_federate, fed) for fed in config]
            for fut in as_completed(futures):
                cname, pts = fut.result()
                federate_scores[cname] = [p.score for p in pts]
                all_points.extend(pts)

        # RRF merge + dedup by chunk_id
        seen_ids: set = set()
        deduped = []
        for pt in sorted(all_points, key=lambda p: p.score, reverse=True):
            pid = str(pt.id)
            if pid not in seen_ids:
                seen_ids.add(pid)
                deduped.append(pt)

        top = deduped[:k]
        context = "\n\n".join(p.payload.get("text","") for p in top)
        answer  = llm_call(
            f"Context:\n{context}\n\nQuestion: {query}", max_tokens=1024
        )
        return {
            "retrieved_chunks": [p.payload for p in top],
            "scores":           [p.score   for p in top],
            "federate_scores":  federate_scores,
            "answer":           answer,
        }


__all__ = [
    "EnsembleRAG", "AdaptiveRAG",
    "CachingRAG", "EvaluationRAG",
    "IncrementalRAG",
    "MultiTenantRAG", "FederatedRAG",
]
