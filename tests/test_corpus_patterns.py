# -*- coding: utf-8 -*-
"""Per-corpus patterns: the chunker chosen for a corpus at ingest, and the
query-time behaviour that goes with it (sentence windows, payload filters)."""
import copy

from rag_factory.finalized_ingest import build_payloads, content_sha256, structured_chunks
from test_service_mode import HDRS, _finalized, _points, client  # noqa: F401  (fixture)


def _with_tables(**over):
    doc = _finalized(**over)
    doc["content"]["tables"] = [{"name": "Lines", "columns": ["item", "qty"],
                                 "rows": [[f"widget-{i}", i] for i in range(25)]}]
    doc["contentSha256"] = content_sha256(doc["content"])
    return doc


def test_structured_chunks_cover_fields_and_table_rows():
    doc = _with_tables()
    kinds = structured_chunks(doc["content"], "invoice")
    assert [k for k, _ in kinds] == ["fields", "table", "table"], "25 rows -> 2 blocks of <= 20"
    assert "invoice_number: INV-1" in kinds[0][1] and "(invoice)" in kinds[0][1]
    assert kinds[2][1].startswith("Lines (invoice)\n| item | qty |"), "header repeated on every block"


def test_build_payloads_marks_kind_and_keeps_ids_deterministic():
    from rag_factory.api.main import _chunker_fixed
    doc = _with_tables()
    plain = build_payloads(doc, "acme", _chunker_fixed)
    rich = build_payloads(doc, "acme", _chunker_fixed, structured=True, chunker_name="structured_chunking")
    assert {p["chunk_kind"] for _, p in plain} == {"text"}
    assert len(rich) == len(plain) + 3
    assert rich[-1][1]["chunk_kind"] == "table" and rich[-1][1]["chunker"] == "structured_chunking"
    assert [i for i, _ in rich] == [i for i, _ in build_payloads(doc, "acme", _chunker_fixed, structured=True)]


def test_ingest_finalized_with_structured_chunker(client, memory_qdrant):
    r = client.post("/ingest/finalized?collection_name=c_struct&chunker=structured_chunking",
                    headers=HDRS, json=_with_tables())
    assert r.status_code == 200, r.text
    assert r.json()["chunker"] == "structured_chunking"
    kinds = [p.payload["chunk_kind"] for p in _points(memory_qdrant, "c_struct")]
    assert kinds.count("fields") == 1 and kinds.count("table") == 2


def test_rechunking_the_same_version_replaces_its_points(client, memory_qdrant):
    doc = _with_tables()
    client.post("/ingest/finalized?collection_name=c_re&chunker=structured_chunking", headers=HDRS, json=doc)
    before = len(_points(memory_qdrant, "c_re"))
    r = client.post("/ingest/finalized?collection_name=c_re&chunker=fixed_chunking", headers=HDRS, json=doc)
    assert r.status_code == 200
    after = _points(memory_qdrant, "c_re")
    assert len(after) == before - 3 and {p.payload["chunk_kind"] for p in after} == {"text"}


def test_unknown_chunker_is_refused(client):
    r = client.post("/ingest/finalized?collection_name=c_x&chunker=magic", headers=HDRS, json=_finalized())
    assert r.status_code == 400


def test_chunker_in_body_is_not_part_of_the_hash(client, memory_qdrant):
    doc = copy.deepcopy(_finalized())
    doc["collection_name"] = "c_body"
    doc["chunker"] = "sentence_window_chunking"
    r = client.post("/ingest/finalized", headers=HDRS, json=doc)
    assert r.status_code == 200, r.text
    pts = _points(memory_qdrant, "c_body")
    assert all("window_text" in p.payload for p in pts)


def test_sentence_window_answers_from_the_window(client, memory_qdrant, monkeypatch):
    from rag_factory.api import main
    from rag_factory.components import base
    seen = {}

    def fake_llm(prompt, *a, **k):
        seen.setdefault("prompt", prompt)
        return "0.9"
    monkeypatch.setattr(base, "llm_call", fake_llm)
    doc = _finalized()
    doc["content"]["pages"] = None
    doc["content"]["text"] = "Alpha is first. Beta follows alpha closely. Gamma ends the list here."
    doc["contentSha256"] = content_sha256(doc["content"])
    client.post("/ingest/finalized?collection_name=c_win&chunker=sentence_window_chunking", headers=HDRS, json=doc)
    r = client.post("/query", headers=HDRS, json={"query": "beta", "collection_name": "c_win", "top_k": 1})
    assert r.status_code == 200, r.text
    assert "Alpha is first. Beta follows alpha closely. Gamma ends the list here." in seen["prompt"]
    assert main  # imported for the monkeypatch target


def test_query_filters_are_allowlisted_and_tenant_cannot_be_overridden(client, memory_qdrant):
    client.post("/ingest/finalized?collection_name=c_f&chunker=structured_chunking", headers=HDRS, json=_with_tables())
    r = client.post("/query", headers=HDRS, json={
        "query": "widget", "collection_name": "c_f", "top_k": 10, "filters": {"chunk_kind": "table"}})
    assert r.status_code == 200, r.text
    assert {c["chunk_kind"] for c in r.json()["retrieved_chunks"]} == {"table"}
    r = client.post("/query", headers=HDRS, json={
        "query": "x", "collection_name": "c_f", "filters": {"tenant_id": "other"}})
    assert r.status_code == 400
