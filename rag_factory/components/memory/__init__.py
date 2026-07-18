# -*- coding: utf-8 -*-
"""Tier 5 — Memory & Conversation component wrappers."""
from __future__ import annotations
from collections import deque
from typing import Any, Deque, Dict, List
from ..base import BaseComponent, embed, dense_search, llm_call, get_qdrant_client
from ...spec.manifest import MANIFEST

# In-process session store (replace with Redis/DynamoDB in production)
_SESSION_STORE: Dict[str, Deque] = {}
_MEMORY_STORE:  Dict[str, List]  = {}


class MemoryAugmentedRAG(BaseComponent):
    SPEC = MANIFEST.get("memory_augmented_rag")

    def __init__(self, short_term_size: int = 5):
        self.short_term_size = short_term_size

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "session_id")
        query      = ctx["query"]
        session_id = ctx["session_id"]
        cname      = ctx.get("collection_name", "")
        qdrant     = get_qdrant_client()

        # Load session memory
        short_term = _SESSION_STORE.setdefault(
            session_id, deque(maxlen=self.short_term_size)
        )
        long_term  = _MEMORY_STORE.setdefault(session_id, [])

        # Retrieve from knowledge base
        retrieved = []
        if cname:
            vec    = embed(query)
            points = dense_search(qdrant, cname, vec, 5)
            retrieved = [p.payload.get("text","") for p in points]

        # Find relevant long-term memories
        memory_hits = [m for m in long_term if any(
            w.lower() in m.lower() for w in query.split() if len(w) > 4
        )]

        context = "\n".join(retrieved)
        history = "\n".join(list(short_term))
        memory  = "\n".join(memory_hits)

        answer = llm_call(
            f"Conversation history:\n{history}\n\n"
            f"Relevant memories:\n{memory}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}",
            max_tokens=1024,
        )

        # Update memory
        short_term.append(f"User: {query}\nAssistant: {answer[:200]}")
        if len(answer) > 100:
            long_term.append(f"Q: {query[:80]} A: {answer[:120]}")

        return {
            "answer":         answer,
            "memory_hits":    memory_hits,
            "updated_memory": list(short_term),
        }


class MultiTurnRAG(BaseComponent):
    SPEC = MANIFEST.get("multiturn_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "conversation_history")
        query   = ctx["query"]
        history = ctx["conversation_history"]  # List[Dict{role,content}]
        cname   = ctx.get("collection_name", "")
        qdrant  = get_qdrant_client()

        history_text = "\n".join(
            f"{m.get('role','user').capitalize()}: {m.get('content','')}"
            for m in history[-6:]  # last 3 turns
        )

        # Rewrite follow-up query in context of history
        rewritten = llm_call(
            f"Conversation so far:\n{history_text}\n\n"
            f"Follow-up: {query}\n\n"
            "Rewrite the follow-up as a fully self-contained search query "
            "(resolve all pronouns and ellipsis). One line only:",
            max_tokens=80,
        ).strip()

        retrieved = []
        if cname:
            vec    = embed(rewritten)
            points = dense_search(qdrant, cname, vec, 5)
            retrieved = [p.payload.get("text","") for p in points]

        context = "\n".join(retrieved)
        answer  = llm_call(
            f"Context:\n{context}\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Question: {query}",
            max_tokens=1024,
        )
        return {"answer": answer, "rewritten_query": rewritten}


__all__ = ["MemoryAugmentedRAG", "MultiTurnRAG"]
