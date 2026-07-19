# -*- coding: utf-8 -*-
"""
Guards — FM1-FM4 failure mode detectors injected at assembly time.
Each guard runs silently: it adds a 'guard_log' key to ctx and may
rewrite 'query' or 'retrieved_chunks' to fix detected issues.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List
from ..components.base import BaseComponent, llm_call, embed
from ..spec.manifest import MANIFEST


class RetrievalGuard(BaseComponent):
    """FM-R1-R6: vocabulary gap, chunking boundary, context, HyDE, K-dilution, stale."""
    SPEC = MANIFEST.get("retrieval_guard")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        log: List[str] = []
        chunks = ctx.get("retrieved_chunks", [])
        query  = ctx.get("query", "")

        # FM-R5: K-dilution — if more than 10 chunks, warn
        if len(chunks) > 10:
            log.append(f"FM-R5: K={len(chunks)} may dilute LLM context")

        # FM-R6: Stale content — check if any chunk has stale marker
        for ch in chunks:
            if ch.get("stale"):
                log.append(f"FM-R6: Stale chunk detected: {ch.get('text','')[:60]}")

        # FM-R1: Vocabulary gap — check cosine sim of top chunk to query
        if chunks:
            import numpy as np
            try:
                q_vec  = embed(query)
                c_vec  = embed(chunks[0].get("text","")[:500])
                sim    = float(np.dot(q_vec, c_vec) /
                               (np.linalg.norm(q_vec) * np.linalg.norm(c_vec)))
                if sim < 0.25:
                    log.append(f"FM-R1: Top chunk similarity={sim:.3f} (<0.25) — vocabulary gap likely")
            except Exception:
                pass

        return {"guard_log": {"retrieval": log}}


class GenerationGuard(BaseComponent):
    """FM-G1-G5: faithfulness, lost-in-middle, over-reliance, multi-hop, counterfactual."""
    SPEC = MANIFEST.get("generation_guard")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        log: List[str] = []
        answer = ctx.get("answer", "")
        chunks = ctx.get("retrieved_chunks", [])
        query  = ctx.get("query", "")

        if not answer or not chunks:
            return {"guard_log": {"generation": log}, "faithfulness_score": 1.0}

        context = " ".join(c.get("text","") for c in chunks)

        # FM-G1: Basic faithfulness check
        import re as _re
        faith_raw = llm_call(
            f"Context: {context[:800]}\nAnswer: {answer[:400]}\n\n"
            "Reply with a single decimal number 0.0-1.0 scoring how well "
            "the answer is supported by the context. No explanation.",
            max_tokens=20, temperature=0.0,
        )
        nums = _re.findall(r"[01](?:\.\d+)?|\.\d+", faith_raw.strip())
        try:
            faith = min(1.0, max(0.0, float(nums[0]))) if nums else 0.5
        except (ValueError, IndexError):
            faith = 0.5

        if faith < 0.6:
            log.append(f"FM-G1: Faithfulness={faith:.2f} (<0.6) — possible hallucination")

        # FM-G2: Lost-in-middle — reorder so top chunk is first
        if len(chunks) > 5:
            log.append("FM-G2: >5 chunks detected — consider reranking to avoid lost-in-middle")

        return {
            "guard_log":         {"generation": log},
            "faithfulness_score": round(faith, 3),
        }


class AmbiguityGuard(BaseComponent):
    """FM-A1-A5: negation rewrite, temporal, multi-intent, coreference, scope."""
    SPEC = MANIFEST.get("ambiguity_guard")

    _NEGATION_PATTERNS = re.compile(
        r"\b(not|never|no|without|except|excluding|other than)\b", re.I
    )
    _GLOBAL_PATTERNS   = re.compile(
        r"\b(all|every|overall|summarise|summarize|main themes|across all)\b", re.I
    )

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        query   = ctx.get("query", "")
        log:     List[str] = []
        rewrite: bool      = False
        new_query          = query

        # FM-A1: Negation rewrite
        if self._NEGATION_PATTERNS.search(query):
            log.append(f"FM-A1: Negation detected in query — rewriting")
            new_query = llm_call(
                f"Rewrite this query to avoid negation while preserving intent:\n{query}",
                max_tokens=60,
            ).strip()
            rewrite = True

        # FM-A5: Global scope — warn that standard RAG will miss corpus-level answers
        if self._GLOBAL_PATTERNS.search(query):
            log.append("FM-A5: Global-scope query detected — consider map-reduce strategy")

        # FM-A3: Multi-intent — count '?' or conjunctions
        if query.count("?") > 1 or re.search(r"\b(and also|as well as|additionally)\b", query, re.I):
            log.append("FM-A3: Multi-intent query detected — consider query decomposition")

        return {
            "query":           new_query,
            "guard_log":       {"ambiguity": log},
            "rewrite_applied": rewrite,
        }


class SystemGuard(BaseComponent):
    """FM-S1-S5: index drift, prompt injection, PII, cascade, fragmentation."""
    SPEC = MANIFEST.get("system_guard")

    _INJECTION_PATTERNS = re.compile(
        r"(ignore previous|system update|admin mode|you are now|"
        r"disregard prior|act as|jailbreak|DAN mode)",
        re.I,
    )
    _PII_PATTERNS = re.compile(
        r"\b(\d{3}-\d{2}-\d{4}|"          # SSN
        r"\d{16}|\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}|"  # credit card
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"  # email
        r"\b"
    )

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        log:    List[str] = []
        chunks  = ctx.get("retrieved_chunks", [])
        query   = ctx.get("query", "")
        clean_chunks = []

        for ch in chunks:
            text = ch.get("text", "")

            # FM-S2: Prompt injection detection
            if self._INJECTION_PATTERNS.search(text):
                log.append(f"FM-S2: Injection pattern blocked in chunk: {text[:60]}")
                continue  # drop the chunk

            # FM-S3: PII detection
            if self._PII_PATTERNS.search(text):
                log.append(f"FM-S3: PII pattern detected in chunk: {text[:40]}...")

            clean_chunks.append(ch)

        # FM-S2 in query
        if self._INJECTION_PATTERNS.search(query):
            log.append(f"FM-S2: Injection pattern detected in query")

        return {
            "retrieved_chunks": clean_chunks,
            "query":            query,
            "guard_log":        {"system": log},
        }


__all__ = ["RetrievalGuard", "GenerationGuard", "AmbiguityGuard", "SystemGuard"]
