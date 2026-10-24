# -*- coding: utf-8 -*-
"""Service mode (behind the AI Control Plane) and /ingest/finalized."""
import copy
import importlib

import pytest
from fastapi.testclient import TestClient

from rag_factory.finalized_ingest import content_sha256
from conftest import fake_embed

TOKEN = "test-service-token"
HDRS = {"X-Service-Token": TOKEN, "X-Tenant-Id": "acme", "X-Persona-Id": "p_analyst-1"}


@pytest.fixture
def client(monkeypatch, memory_qdrant):
    from rag_factory import guards
    from rag_factory.components import base
    from rag_factory.api import main

    monkeypatch.setenv("SERVICE_TOKEN", TOKEN)
    monkeypatch.setattr(base, "llm_call", lambda *a, **k: "0.9")
    monkeypatch.setattr(guards, "llm_call", lambda *a, **k: "0.9")
    monkeypatch.setattr(guards, "embed", fake_embed)
    return TestClient(main.app)


def _finalized(**over):
    content = {
        "text": "Invoice INV-1 from Acme Corp. Total due 1,200 USD by 2026-11-30. " * 12,
        "pages": [
            {"pageNumber": 1, "text": "Invoice INV-1 from Acme Corp. " * 20},
            {"pageNumber": 2, "text": "Total due 1,200 USD by 2026-11-30. " * 20},
        ],
        "fields": [
            {"name": "invoice_number", "value": "INV-1", "confidence": 0.98, "edited": False},
            {"name": "total", "value": 1200, "edited": True, "originalValue": 1100},
            {"name": "notes", "value": "x" * 1000, "edited": False},
        ],
        "tables": [],
        "summary": None,
    }
    doc = {
        "schemaVersion": "finalized-document/v1",
        "finalizedDocumentId": "fd-abc1",
        "version": 1,
        "supersedes": None,
        "tenantId": "acme",
        "documentId": "doc-inv1",
        "source": {"fileName": "inv.pdf", "mimeType": "application/pdf",
                   "sizeBytes": 1234, "sha256": "a" * 64, "pageCount": 2},
        "classification": {"docType": "invoice", "gxp": False, "region": "US"},
        "content": content,
        "pii": {"redaction": "none-detected", "detected": []},
        "provenance": {
            "methodsRun": [{"method": "textract", "status": "complete"}],
            "selectedMethod": "textract", "capabilities": ["kv_extraction"],
            "submittedBy": {"personaId": "p1", "name": "Maker"},
            "submittedAt": "2026-10-01T10:00:00Z",
            "finalizedBy": {"personaId": "p2", "name": "Checker"},
            "finalizedAt": "2026-10-01T11:00:00Z",
        },
    }
    doc.update(over)
    doc["contentSha256"] = content_sha256(doc["content"])
    return doc


def _points(qdrant, coll):
    pts, _ = qdrant.scroll(collection_name=coll, limit=1000, with_payload=True)
    return pts


# ── auth ──────────────────────────────────────────────────────────────────────
def test_health_is_open(client):
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("headers,status", [
    ({}, 401),
    ({**HDRS, "X-Service-Token": "wrong"}, 401),
    ({k: v for k, v in HDRS.items() if k != "X-Tenant-Id"}, 400),
    ({**HDRS, "X-Tenant-Id": "Acme_Corp"}, 400),
    ({k: v for k, v in HDRS.items() if k != "X-Persona-Id"}, 400),
    ({**HDRS, "X-Persona-Id": "bad persona!"}, 400),
])
def test_token_and_identity_headers_required(client, headers, status):
    assert client.get("/specs", headers=headers).status_code == status
    assert client.post("/idp/classify", json={"text": "hi"}, headers=headers).status_code == status


def test_valid_headers_pass(client):
    assert client.get("/specs", headers=HDRS).status_code == 200


def test_no_service_token_keeps_legacy_behaviour(client, monkeypatch):
    monkeypatch.delenv("SERVICE_TOKEN")
    assert client.get("/specs").status_code == 200


# ── tenant override ───────────────────────────────────────────────────────────
def test_header_tenant_overrides_body_tenant(client, memory_qdrant):
    r = client.post("/ingest", headers=HDRS, json={
        "text": "Acme quarterly report. " * 30, "collection_name": "svc_ingest",
        "doc_id": "rep1", "tenant_id": "globex"})
    assert r.status_code == 200, r.text
    assert {p.payload["tenant_id"] for p in _points(memory_qdrant, "svc_ingest")} == {"acme"}

    # Another tenant's data in the same collection is never returned.
    client.post("/ingest", headers={**HDRS, "X-Tenant-Id": "globex"}, json={
        "text": "Globex secret plan. " * 30, "collection_name": "svc_ingest", "doc_id": "sec"})
    r = client.post("/query", headers=HDRS, json={
        "query": "plan", "collection_name": "svc_ingest", "top_k": 50, "tenant_id": "globex"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tenant_id"] == "acme"
    assert body["retrieved_chunks"] and all(c["tenant_id"] == "acme" for c in body["retrieved_chunks"])


def test_query_honours_spec_top_k(client, memory_qdrant):
    client.post("/ingest", headers=HDRS, json={
        "text": "Sentence about acme. " * 200, "collection_name": "svc_spec", "doc_id": "d"})
    r = client.post("/query", headers=HDRS, json={
        "query": "acme", "collection_name": "svc_spec", "top_k": 1, "spec": "multitenant.yaml"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["top_k"] == 5 and body["retrieval_strategy"] == "multitenant_rag"
    assert len(body["retrieved_chunks"]) == 5
    # Spec paths outside specs/ are refused in service mode.
    r = client.post("/query", headers=HDRS, json={
        "query": "x", "collection_name": "svc_spec", "spec": "../specs/simple.yaml"})
    assert r.status_code == 400


def test_multitenant_spec_requires_tenant_outside_service_mode(client, monkeypatch):
    monkeypatch.delenv("SERVICE_TOKEN")
    r = client.post("/query", json={"query": "x", "collection_name": "c", "spec": "multitenant.yaml"})
    assert r.status_code == 400


# ── /ingest/finalized ─────────────────────────────────────────────────────────
def test_finalized_rejects_bad_schema_version(client):
    doc = _finalized(schemaVersion="finalized-document/v0")
    r = client.post("/ingest/finalized?collection_name=fin", json=doc, headers=HDRS)
    assert r.status_code == 422
    assert any("schemaVersion" in e for e in r.json()["detail"]["errors"])


def test_finalized_rejects_missing_keys(client):
    doc = _finalized()
    del doc["provenance"]
    r = client.post("/ingest/finalized?collection_name=fin", json=doc, headers=HDRS)
    assert r.status_code == 422


def test_finalized_rejects_tenant_mismatch(client):
    doc = _finalized(tenantId="globex")
    r = client.post("/ingest/finalized?collection_name=fin", json=doc, headers=HDRS)
    assert r.status_code == 403


def test_finalized_rejects_hash_mismatch(client):
    doc = _finalized()
    doc["content"]["fields"][0]["value"] = "INV-TAMPERED"
    r = client.post("/ingest/finalized?collection_name=fin", json=doc, headers=HDRS)
    assert r.status_code == 409


def test_finalized_requires_collection(client):
    r = client.post("/ingest/finalized", json=_finalized(), headers=HDRS)
    assert r.status_code == 400


def test_finalized_accepts_valid_doc_and_is_idempotent(client, memory_qdrant):
    doc = _finalized()
    r = client.post("/ingest/finalized?collection_name=fin", json=doc, headers=HDRS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tenant_id"] == "acme" and body["version"] == 1
    assert body["chunk_count"] == body["points_upserted"] > 0 and body["points_deleted"] == 0

    pts = _points(memory_qdrant, "fin")
    assert len(pts) == body["chunk_count"]
    p = pts[0].payload
    for key in ("text", "doc_id", "finalized_document_id", "version", "tenant_id", "doc_type",
                "gxp", "region", "page_num", "chunk_index", "source_sha256", "content_sha256",
                "fields"):
        assert key in p, key
    assert p["doc_id"] == "doc-inv1" and p["doc_type"] == "invoice" and p["region"] == "US"
    assert p["fields"]["invoice_number"] == "INV-1" and p["fields"]["total"] == 1200
    assert len(p["fields"]["notes"]) == 256
    assert {pt.payload["page_num"] for pt in pts} == {1, 2}

    # Same version again (collection via body field): replaced, no duplicates.
    again = dict(copy.deepcopy(doc), collection_name="fin")
    r2 = client.post("/ingest/finalized", json=again, headers=HDRS)
    assert r2.status_code == 200, r2.text
    assert r2.json()["replaced_same_version"] is True and r2.json()["points_deleted"] == 0
    assert {str(x.id) for x in _points(memory_qdrant, "fin")} == {str(x.id) for x in pts}

    # New version of the same documentId supersedes the old points.
    v2 = _finalized(finalizedDocumentId="fd-abc2", version=2, supersedes="fd-abc1")
    v2["content"] = {**v2["content"], "pages": [{"pageNumber": 1, "text": "Revised. " * 10}]}
    v2["contentSha256"] = content_sha256(v2["content"])
    r3 = client.post("/ingest/finalized?collection_name=fin", json=v2, headers=HDRS)
    assert r3.status_code == 200, r3.text
    assert r3.json()["superseded_versions"] == [1]
    assert r3.json()["points_deleted"] == len(pts)
    left = _points(memory_qdrant, "fin")
    assert {x.payload["version"] for x in left} == {2}
    assert {x.payload["finalized_document_id"] for x in left} == {"fd-abc2"}

    # Replaying the old version afterwards is refused.
    r4 = client.post("/ingest/finalized?collection_name=fin", json=doc, headers=HDRS)
    assert r4.status_code == 409


def test_finalized_same_ids_other_tenant_do_not_collide(client, memory_qdrant):
    hdr_g = {**HDRS, "X-Tenant-Id": "globex"}
    assert client.post("/ingest/finalized?collection_name=fin2", json=_finalized(),
                       headers=HDRS).status_code == 200
    r = client.post("/ingest/finalized?collection_name=fin2",
                    json=_finalized(tenantId="globex"), headers=hdr_g)
    assert r.status_code == 200 and r.json()["points_deleted"] == 0
    tenants = [x.payload["tenant_id"] for x in _points(memory_qdrant, "fin2")]
    assert tenants.count("acme") == tenants.count("globex") > 0


# ── CORS ──────────────────────────────────────────────────────────────────────
def test_cors_restricted_in_service_mode(monkeypatch):
    from rag_factory.api import main
    monkeypatch.setenv("SERVICE_TOKEN", TOKEN)
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://cp.example.com")
    try:
        mod = importlib.reload(main)
        c = TestClient(mod.app)
        pre = {"Access-Control-Request-Method": "POST"}
        ok = c.options("/query", headers={**pre, "Origin": "https://cp.example.com"})
        assert ok.headers.get("access-control-allow-origin") == "https://cp.example.com"
        bad = c.options("/query", headers={**pre, "Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in bad.headers
        assert c.get("/openapi.json").status_code == 404
    finally:
        monkeypatch.delenv("SERVICE_TOKEN")
        monkeypatch.delenv("ALLOWED_ORIGINS")
        importlib.reload(main)
