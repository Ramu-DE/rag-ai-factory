# -*- coding: utf-8 -*-
"""Decision-grade answers: hybrid retrieval, version safety, citations, claim checks, abstention."""
import json

from rag_factory import decision_query as dq
from rag_factory.finalized_ingest import content_sha256
from test_service_mode import HDRS, _finalized, client  # noqa: F401  (fixture)


# ---------------------------------------------------------------- pure helpers

def test_identifier_tokens_pick_codes_not_plain_numbers():
    assert dq.identifier_tokens("assay for batch VF-24-0917 and invoice INV-2026-0042 at 99.4 %") == ["VF-24-0917", "INV-2026-0042"]
    assert dq.identifier_tokens("what is the total?") == []


def test_rrf_and_latest_versions():
    s = dq.rrf([["a", "b"], ["b", "c"]])
    assert max(s, key=s.get) == "b"
    chunks = [{"doc_id": "d1", "version": 1}, {"doc_id": "d1", "version": 2}, {"doc_id": "d2", "version": 1}]
    assert dq.latest_versions_only(chunks) == [{"doc_id": "d1", "version": 2}, {"doc_id": "d2", "version": 1}]


def test_claims_need_citations_and_grounded_numbers():
    blocks = {1: {"text_for_check": "Assay (HPLC) 95.0 - 105.0 % 99.4 %. Batch VF-24-0917."}}
    ok = dq.check_claims("The assay result was 99.4 % for batch VF-24-0917 [1].", blocks)
    assert ok[0]["supported"], ok
    wrong = dq.check_claims("The assay result was 99.8 % [1].", blocks)
    assert not wrong[0]["supported"] and "99.8" in wrong[0]["problems"][0]
    uncited = dq.check_claims("The batch was released.", blocks)
    assert uncited[0]["problems"] == ["no citation"]
    ghost = dq.check_claims("Water was 1.4 % [3].", blocks)
    assert "does not exist" in ghost[0]["problems"][0]


def test_split_claims_keeps_abbreviations_and_splits_on_citations():
    a = "Microbial limits complied with Ph. Eur. [2][7]. Water was 1.4 % [2]. Dr. H. Sato approved it [3]."
    assert dq.split_claims(a) == ["Microbial limits complied with Ph. Eur. [2][7].", "Water was 1.4 % [2].", "Dr. H. Sato approved it [3]."]


def test_split_claims_skips_headings():
    claims = dq.split_claims("## Result\n- Assay was 99.4 % [1].\n- Released [1].\nSummary:")
    assert claims == ["Assay was 99.4 % [1].", "Released [1]."]


# ---------------------------------------------------------------- endpoint

def _coa(fd="fd-c1", doc="doc-c1", version=1, assay="99.4 %"):
    d = _finalized(finalizedDocumentId=fd, documentId=doc, version=version)
    d["content"]["pages"] = None
    d["content"]["text"] = f"CERTIFICATE OF ANALYSIS. Batch Number: VF-24-0917. Assay (HPLC) result {assay}. Batch released."
    d["content"]["fields"] = [{"name": "batch_number", "value": "VF-24-0917", "edited": False},
                              {"name": "assay", "value": assay, "edited": False}]
    d["classification"]["docType"] = "Certificate of Analysis"
    d["contentSha256"] = content_sha256(d["content"])
    return d


def _scripted(monkeypatch, answers):
    """Replace the LLM: answers for generation, a judge that accepts grounded claims."""
    from rag_factory.components import base
    calls = {"gen": 0, "judge": 0}

    def fake(prompt, system="", max_tokens=0, temperature=0.0):
        if "strict verifier" in prompt:
            calls["judge"] += 1
            n = prompt.count("Claim ")
            return json.dumps({"verdicts": [{"claim": i, "supported": True, "reason": "ok"} for i in range(n)]})
        calls["gen"] += 1
        return answers[min(calls["gen"], len(answers)) - 1]
    monkeypatch.setattr(base, "llm_call", fake)
    return calls


def _ingest(client, doc, coll="dq"):
    r = client.post(f"/ingest/finalized?collection_name={coll}&chunker=structured_chunking", headers=HDRS, json=doc)
    assert r.status_code == 200, r.text


def test_grounded_answer_is_returned_with_citations(client, memory_qdrant, monkeypatch):
    _ingest(client, _coa())
    calls = _scripted(monkeypatch, ["The assay result for batch VF-24-0917 was 99.4 % [1]."])
    r = client.post("/query/decision", headers=HDRS, json={"query": "assay result for batch VF-24-0917?", "collection_name": "dq"})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["abstained"] is False, body
    assert body["claims"][0]["supported"] and body["claims"][0]["citations"] == [1]
    assert body["citations"][0]["finalized_document_id"] == "fd-c1"
    assert body["retrieval"]["tokens"] == ["VF-24-0917"]
    assert calls["judge"] == 1


def test_wrong_number_is_retried_then_abstains(client, memory_qdrant, monkeypatch):
    _ingest(client, _coa())
    _scripted(monkeypatch, ["The assay was 99.8 % [1].", "The assay was 98 % [1]."])
    body = client.post("/query/decision", headers=HDRS, json={"query": "assay result?", "collection_name": "dq", "min_score": 0}).json()
    assert body["abstained"] is True and body.get("attempts") == 2, body
    assert "verified" in body["reason"]


def test_not_found_and_retry_success(client, memory_qdrant, monkeypatch):
    _ingest(client, _coa())
    _scripted(monkeypatch, ["NOT_FOUND"])
    body = client.post("/query/decision", headers=HDRS, json={"query": "who is the CEO?", "collection_name": "dq", "min_score": 0}).json()
    assert body["abstained"] is True and "do not contain" in body["reason"]
    _scripted(monkeypatch, ["Assay was 99.8 % [1].", "Assay was 99.4 % [1]."])
    body = client.post("/query/decision", headers=HDRS, json={"query": "assay?", "collection_name": "dq", "min_score": 0}).json()
    assert body["abstained"] is False and body["attempts"] == 2


def test_only_allowed_current_versions_are_used(client, memory_qdrant, monkeypatch):
    _ingest(client, _coa(fd="fd-c1", version=1, assay="99.4 %"), coll="dqv")
    _ingest(client, _coa(fd="fd-c2", version=2, assay="98.7 %"), coll="dqv")
    _scripted(monkeypatch, ["Assay was 98.7 % [1]."])
    body = client.post("/query/decision", headers=HDRS, json={"query": "assay VF-24-0917", "collection_name": "dqv",
                       "allowed_finalized_ids": ["fd-c2"], "min_score": 0}).json()
    assert {c["finalized_document_id"] for c in body["citations"]} == {"fd-c2"}
    body = client.post("/query/decision", headers=HDRS, json={"query": "assay VF-24-0917", "collection_name": "dqv",
                       "allowed_finalized_ids": ["fd-other"], "min_score": 0}).json()
    assert body["abstained"] is True and body["citations"] == []


def test_negation_is_never_rewritten(client, memory_qdrant, monkeypatch):
    from rag_factory.guards import AmbiguityGuard
    out = AmbiguityGuard().run({"query": "which tests did not meet specification?"})
    assert out["query"] == "which tests did not meet specification?"
    assert out["rewrite_applied"] is False
