# -*- coding: utf-8 -*-
"""Offline test fixtures: in-memory Qdrant, deterministic fake embeddings."""
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fake_embed(text: str):
    """Deterministic 1024-d unit-ish vector; never calls Bedrock."""
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    vals = [(seed[i % 32] - 128) / 128.0 + 1e-3 for i in range(1024)]
    return vals


@pytest.fixture
def memory_qdrant(monkeypatch):
    """Force a fresh in-memory Qdrant and offline embeddings for one test."""
    from rag_factory.components import base

    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("RAG_FACTORY_ALLOW_MEMORY", "1")
    monkeypatch.setattr(base, "_QDRANT_SINGLETON", None)
    monkeypatch.setattr(base, "embed", fake_embed)
    client = base.get_qdrant_client()
    yield client
    monkeypatch.setattr(base, "_QDRANT_SINGLETON", None)
