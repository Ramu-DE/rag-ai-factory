# -*- coding: utf-8 -*-
"""Two documents in one collection must not overwrite each other's index/chunks."""
import json
import os
from types import SimpleNamespace

import pytest

from rag_factory.pdf_processor import IncrementalPDFProcessor


def _make_pdf(path, pages):
    import pymupdf
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _doc_points(qdrant, collection, doc_id):
    pts, _ = qdrant.scroll(collection_name=collection, limit=1000, with_payload=True)
    return {str(p.id) for p in pts if p.payload.get("doc_id") == doc_id}


LONG_A = "Alpha document page {} with plenty of text to make a chunk. " * 3
LONG_B = "Bravo document page {} completely different content here. " * 3


def test_two_documents_same_collection_do_not_overwrite(tmp_path, memory_qdrant):
    coll = "test_two_docs"
    a_pdf, b_pdf = tmp_path / "a.pdf", tmp_path / "b.pdf"
    _make_pdf(a_pdf, [LONG_A.replace("{}", "1"), LONG_A.replace("{}", "2")])
    _make_pdf(b_pdf, [LONG_B.replace("{}", "1")])

    proc = IncrementalPDFProcessor(index_dir=str(tmp_path / "idx"))

    r_a = proc.process(str(a_pdf), coll, doc_id="doc-a")
    assert (r_a.pages_added, r_a.pages_deleted, r_a.chunks_deleted) == (2, 0, 0)
    a_ids = _doc_points(memory_qdrant, coll, "doc-a")
    assert a_ids

    # Second document: previously page 0 was "UPDATED" and page 1 "DELETED".
    r_b = proc.process(str(b_pdf), coll, doc_id="doc-b")
    assert r_b.incremental is False
    assert (r_b.pages_added, r_b.pages_updated, r_b.pages_deleted, r_b.chunks_deleted) == (1, 0, 0, 0)
    assert _doc_points(memory_qdrant, coll, "doc-a") == a_ids
    assert _doc_points(memory_qdrant, coll, "doc-b")

    # Re-processing A is fully UNCHANGED and still leaves B alone.
    r_a2 = proc.process(str(a_pdf), coll, doc_id="doc-a")
    assert r_a2.incremental is True
    assert (r_a2.pages_skipped, r_a2.chunks_embedded, r_a2.chunks_deleted) == (2, 0, 0)

    # Updating A (drop page 2, change page 1) only touches A's chunks.
    b_ids = _doc_points(memory_qdrant, coll, "doc-b")
    _make_pdf(a_pdf, ["Alpha REVISED first page, enough characters to be chunked properly."])
    r_a3 = proc.process(str(a_pdf), coll, doc_id="doc-a")
    assert (r_a3.pages_updated, r_a3.pages_deleted) == (1, 1)
    assert r_a3.chunks_deleted == len(a_ids)
    assert _doc_points(memory_qdrant, coll, "doc-b") == b_ids

    # Per-document and per-collection stats.
    assert proc.get_index_stats(coll, doc_id="doc-b")["indexed_pages"] == 1
    agg = proc.get_index_stats(coll)
    assert agg["documents"] == 2 and sorted(agg["doc_ids"]) == ["doc-a", "doc-b"]
    assert agg["indexed_pages"] == 2 and agg["doc_id"] is None

    # Per-document reset leaves the other document's index.
    proc.reset_index(coll, doc_id="doc-a")
    assert proc.get_index_stats(coll)["doc_ids"] == ["doc-b"]
    proc.reset_index(coll)
    assert proc.get_index_stats(coll)["documents"] == 0


def test_same_doc_id_different_tenants_are_separate(tmp_path, memory_qdrant):
    coll = "test_tenants"
    pdf = tmp_path / "invoice.pdf"
    _make_pdf(pdf, [LONG_A.replace("{}", "1")])
    proc = IncrementalPDFProcessor(index_dir=str(tmp_path / "idx"))
    proc.process(str(pdf), coll, doc_id="invoice.pdf", tenant_id="acme")
    r = proc.process(str(pdf), coll, doc_id="invoice.pdf", tenant_id="globex")
    assert r.pages_added == 1 and r.chunks_deleted == 0
    pts, _ = memory_qdrant.scroll(collection_name=coll, limit=100, with_payload=True)
    assert {p.payload["tenant_id"] for p in pts} == {"acme", "globex"}
    assert proc.get_index_stats(coll, tenant_id="acme")["documents"] == 1


def test_legacy_collection_index_is_migrated(tmp_path, memory_qdrant):
    coll = "test_legacy"
    idx = tmp_path / "idx"
    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf, [LONG_A.replace("{}", "1")])
    proc = IncrementalPDFProcessor(index_dir=str(idx))
    proc.process(str(pdf), coll, doc_id="doc-a")
    # Rewrite as the legacy per-collection layout.
    (new_path,) = [p for p in idx.iterdir()]
    data = json.loads(new_path.read_text())
    new_path.unlink()
    (idx / f"{coll}_index.json").write_text(json.dumps(data))

    other = proc.process(str(pdf), coll, doc_id="doc-other")
    assert other.incremental is False and other.chunks_deleted == 0
    again = proc.process(str(pdf), coll, doc_id="doc-a")
    assert again.incremental is True and again.pages_skipped == 1
    assert not (idx / f"{coll}_index.json").exists()


def test_idp_textract_path_is_per_document(tmp_path, memory_qdrant):
    from rag_factory.idp_pipeline import IDPPipeline

    def extracted(text):
        page = SimpleNamespace(page_num=0, text=text, rich_text=text)
        return SimpleNamespace(method="textract_forms", pages=[page])

    pipe = IDPPipeline(collection_name="test_idp", index_dir=str(tmp_path / "idx"))
    r1 = pipe._ingest(extracted(LONG_A.replace("{}", "1")), "scan1.png", display_name="scan1.png")
    r2 = pipe._ingest(extracted(LONG_B.replace("{}", "1")), "scan2.png", display_name="scan2.png")
    assert r1.pages_added == 1
    assert (r2.pages_added, r2.pages_updated, r2.chunks_deleted) == (1, 0, 0)
    assert _doc_points(memory_qdrant, "test_idp", "scan1.png")
    assert _doc_points(memory_qdrant, "test_idp", "scan2.png")


def test_qdrant_requires_config_unless_memory_allowed(monkeypatch):
    from rag_factory.components import base

    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("RAG_FACTORY_ALLOW_MEMORY", raising=False)
    monkeypatch.setattr(base, "_QDRANT_SINGLETON", None)
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        base.get_qdrant_client()
