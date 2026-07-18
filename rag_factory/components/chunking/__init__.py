# -*- coding: utf-8 -*-
"""
Tier 1 — Chunking & Indexing component wrappers.
Each class wraps the logic from the corresponding notebook into run(ctx)->dict.
"""
from __future__ import annotations
import re, hashlib, uuid
from typing import Any, Dict, List
from ..base import BaseComponent, embed, chunk_id, llm_call, get_qdrant_client
from ...spec.manifest import MANIFEST
from qdrant_client.models import VectorParams, Distance, PointStruct


def _upsert(collection_name: str, chunks: List[Dict], dim: int = 1024) -> None:
    qdrant = get_qdrant_client()
    if collection_name not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.create_collection(
            collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    points = []
    for ch in chunks:
        vec = embed(ch["text"])
        points.append(PointStruct(
            id=chunk_id(f"{collection_name}:{ch['text'][:80]}"),
            vector=vec,
            payload=ch,
        ))
    qdrant.upsert(collection_name=collection_name, points=points)


# ---------------------------------------------------------------------------
# 01 — Fixed chunking
# ---------------------------------------------------------------------------
class FixedChunking(BaseComponent):
    SPEC = MANIFEST.get("fixed_chunking")

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "raw_text")
        text   = ctx["raw_text"]
        chunks = []
        start  = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append({"text": text[start:end], "chunk_index": len(chunks)})
            start += self.chunk_size - self.overlap
        if ctx.get("collection_name"):
            _upsert(ctx["collection_name"], chunks)
        return {"chunks": chunks, "chunk_count": len(chunks)}


# ---------------------------------------------------------------------------
# 02 — Semantic chunking
# ---------------------------------------------------------------------------
class SemanticChunking(BaseComponent):
    SPEC = MANIFEST.get("semantic_chunking")

    def __init__(self, threshold: float = 0.3, min_chunk_size: int = 100):
        self.threshold      = threshold
        self.min_chunk_size = min_chunk_size

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "raw_text")
        import numpy as np
        sentences = re.split(r'(?<=[.!?])\s+', ctx["raw_text"].strip())
        sentences = [s for s in sentences if len(s) > 20]
        if len(sentences) < 2:
            chunks = [{"text": ctx["raw_text"], "chunk_index": 0}]
            return {"chunks": chunks, "chunk_count": 1}

        embeddings = [embed(s) for s in sentences]
        sims = []
        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i])
            b = np.array(embeddings[i + 1])
            sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))

        # Find breakpoints where similarity drops below threshold
        breaks = [0]
        for i, s in enumerate(sims):
            if s < self.threshold:
                breaks.append(i + 1)
        breaks.append(len(sentences))

        chunks = []
        for i in range(len(breaks) - 1):
            segment = " ".join(sentences[breaks[i]:breaks[i + 1]])
            if len(segment) >= self.min_chunk_size:
                chunks.append({"text": segment, "chunk_index": len(chunks)})

        if not chunks:
            chunks = [{"text": ctx["raw_text"], "chunk_index": 0}]

        if ctx.get("collection_name"):
            _upsert(ctx["collection_name"], chunks)
        return {"chunks": chunks, "chunk_count": len(chunks)}


# ---------------------------------------------------------------------------
# 03 — Hierarchical chunking
# ---------------------------------------------------------------------------
class HierarchicalChunking(BaseComponent):
    SPEC = MANIFEST.get("hierarchical_chunking")

    def __init__(self, parent_size: int = 1000, child_size: int = 200):
        self.parent_size = parent_size
        self.child_size  = child_size

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "raw_text")
        text     = ctx["raw_text"]
        parents  = []
        children = []
        hierarchy_map: Dict[str, List[str]] = {}

        p_start = 0
        while p_start < len(text):
            p_text  = text[p_start: p_start + self.parent_size]
            p_id    = chunk_id(f"parent:{p_text[:60]}")
            summary = llm_call(
                f"Summarise in one sentence:\n\n{p_text}",
                max_tokens=80,
            )
            parents.append({"text": p_text, "summary": summary,
                            "chunk_id": p_id, "level": "parent"})

            c_ids = []
            c_start = p_start
            while c_start < p_start + self.parent_size and c_start < len(text):
                c_text = text[c_start: c_start + self.child_size]
                c_id   = chunk_id(f"child:{c_text[:60]}")
                children.append({"text": c_text, "parent_id": p_id,
                                  "chunk_id": c_id, "level": "child"})
                c_ids.append(c_id)
                c_start += self.child_size

            hierarchy_map[p_id] = c_ids
            p_start += self.parent_size

        all_chunks = parents + children
        if ctx.get("collection_name"):
            _upsert(ctx["collection_name"], all_chunks)

        return {
            "chunks": all_chunks,
            "chunk_count": len(all_chunks),
            "hierarchy_map": hierarchy_map,
        }


# ---------------------------------------------------------------------------
# 04 — Parent-child chunking
# ---------------------------------------------------------------------------
class ParentChildChunking(BaseComponent):
    SPEC = MANIFEST.get("parent_child_chunking")

    def __init__(self, parent_size: int = 512, child_size: int = 128):
        self.parent_size = parent_size
        self.child_size  = child_size

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "raw_text")
        text       = ctx["raw_text"]
        chunks     = []
        parent_map: Dict[str, str] = {}

        p_start = 0
        while p_start < len(text):
            p_text = text[p_start: p_start + self.parent_size]
            p_id   = chunk_id(f"parent:{p_text[:60]}")
            chunks.append({"text": p_text, "chunk_id": p_id, "level": "parent"})

            c_start = p_start
            while c_start < p_start + self.parent_size and c_start < len(text):
                c_text = text[c_start: c_start + self.child_size]
                c_id   = chunk_id(f"child:{c_text[:60]}")
                chunks.append({"text": c_text, "chunk_id": c_id,
                               "parent_id": p_id, "level": "child"})
                parent_map[c_id] = p_id
                c_start += self.child_size

            p_start += self.parent_size

        if ctx.get("collection_name"):
            child_chunks = [c for c in chunks if c["level"] == "child"]
            _upsert(ctx["collection_name"], child_chunks)

        return {"chunks": chunks, "chunk_count": len(chunks), "parent_map": parent_map}


# ---------------------------------------------------------------------------
# 05 — Sentence-window chunking
# ---------------------------------------------------------------------------
class SentenceWindowChunking(BaseComponent):
    SPEC = MANIFEST.get("sentence_window_chunking")

    def __init__(self, window: int = 3):
        self.window = window

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "raw_text")
        sentences = re.split(r'(?<=[.!?])\s+', ctx["raw_text"].strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        chunks     = []
        window_map: Dict[str, List[int]] = {}

        for i, sent in enumerate(sentences):
            c_id = chunk_id(f"sent:{i}:{sent[:50]}")
            lo   = max(0, i - self.window)
            hi   = min(len(sentences), i + self.window + 1)
            window_text = " ".join(sentences[lo:hi])
            chunks.append({
                "text": sent,
                "window_text": window_text,
                "chunk_id": c_id,
                "sentence_index": i,
            })
            window_map[c_id] = list(range(lo, hi))

        if ctx.get("collection_name"):
            _upsert(ctx["collection_name"], chunks)

        return {"chunks": chunks, "chunk_count": len(chunks), "window_map": window_map}


# ---------------------------------------------------------------------------
# 06 — Contextual chunking
# ---------------------------------------------------------------------------
class ContextualChunking(BaseComponent):
    SPEC = MANIFEST.get("contextual_chunking")

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "raw_text")
        text = ctx["raw_text"]

        # Split into base chunks
        base_chunks = []
        start = 0
        while start < len(text):
            base_chunks.append(text[start: start + self.chunk_size])
            start += self.chunk_size - self.overlap

        chunks      = []
        context_map: Dict[str, str] = {}

        for i, chunk_text in enumerate(base_chunks):
            context_prefix = llm_call(
                f"Document excerpt:\n\n{chunk_text}\n\n"
                "Provide a one-sentence context that situates this excerpt "
                "within the broader document (entity, date, topic if present):",
                max_tokens=60,
            )
            contextualized = f"{context_prefix} {chunk_text}"
            c_id = chunk_id(f"ctx:{i}:{chunk_text[:50]}")
            chunks.append({
                "text": chunk_text,
                "contextualized_text": contextualized,
                "context_prefix": context_prefix,
                "chunk_id": c_id,
                "chunk_index": i,
            })
            context_map[c_id] = context_prefix

        if ctx.get("collection_name"):
            # Embed the contextualized version, store original text
            from qdrant_client.models import VectorParams, Distance, PointStruct
            qdrant = get_qdrant_client()
            cname  = ctx["collection_name"]
            if cname not in [c.name for c in qdrant.get_collections().collections]:
                qdrant.create_collection(
                    cname,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )
            points = []
            for ch in chunks:
                vec = embed(ch["contextualized_text"])
                pts = PointStruct(id=ch["chunk_id"], vector=vec,
                                  payload={"text": ch["text"],
                                           "context_prefix": ch["context_prefix"],
                                           "chunk_index": ch["chunk_index"]})
                points.append(pts)
            qdrant.upsert(collection_name=cname, points=points)

        return {"chunks": chunks, "chunk_count": len(chunks), "context_map": context_map}


__all__ = [
    "FixedChunking", "SemanticChunking", "HierarchicalChunking",
    "ParentChildChunking", "SentenceWindowChunking", "ContextualChunking",
]
