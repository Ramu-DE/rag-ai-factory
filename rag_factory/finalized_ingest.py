# -*- coding: utf-8 -*-
"""
FinalizedDocument v1 ingestion
==============================
Consumes the AI Control Plane's `FinalizedDocument v1` contract
(docs/contracts/finalized-document.v1.schema.json in the control plane repo):
an immutable, versioned, hash-verified IDP result approved by four eyes.

`validate_finalized_document` performs the structural checks the factory
relies on (not a full JSON-Schema validation; the control plane validates the
full contract before sending). `content_sha256` reproduces the control
plane's canonical JSON hash. `build_points` turns the document into
deterministic Qdrant points so re-ingestion is idempotent.
"""
from __future__ import annotations
import hashlib, json, re, uuid
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "finalized-document/v1"

REQUIRED_KEYS = (
    "schemaVersion", "finalizedDocumentId", "version", "tenantId", "documentId",
    "source", "classification", "content", "pii", "provenance", "contentSha256",
)

_FD_ID_RE  = re.compile(r"^fd-[a-z0-9]+$")
_DOC_ID_RE = re.compile(r"^doc-[a-z0-9]+$")
_HEX64_RE  = re.compile(r"^[a-f0-9]{64}$")

FIELD_VALUE_MAX = 256   # chars per field value stored on each chunk
FIELDS_MAX      = 100   # fields stored on each chunk


class FinalizedDocumentError(ValueError):
    """Structural problem with a FinalizedDocument (maps to HTTP 422)."""

    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def content_sha256(content: Any) -> str:
    return hashlib.sha256(canonical_json(content)).hexdigest()


def validate_finalized_document(doc: Any) -> None:
    """Raise FinalizedDocumentError listing every structural problem found."""
    if not isinstance(doc, dict):
        raise FinalizedDocumentError(["document must be a JSON object"])
    errs: List[str] = []
    missing = [k for k in REQUIRED_KEYS if k not in doc]
    if missing:
        errs.append(f"missing required keys: {missing}")
    if doc.get("schemaVersion") != SCHEMA_VERSION:
        errs.append(f'schemaVersion must be "{SCHEMA_VERSION}"')
    if "finalizedDocumentId" in doc and not (
            isinstance(doc["finalizedDocumentId"], str) and _FD_ID_RE.fullmatch(doc["finalizedDocumentId"])):
        errs.append("finalizedDocumentId must match ^fd-[a-z0-9]+$")
    if "documentId" in doc and not (
            isinstance(doc["documentId"], str) and _DOC_ID_RE.fullmatch(doc["documentId"])):
        errs.append("documentId must match ^doc-[a-z0-9]+$")
    v = doc.get("version")
    if "version" in doc and not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
        errs.append("version must be an integer >= 1")
    if "tenantId" in doc and not (isinstance(doc["tenantId"], str) and doc["tenantId"]):
        errs.append("tenantId must be a non-empty string")
    if "contentSha256" in doc and not (
            isinstance(doc["contentSha256"], str) and _HEX64_RE.fullmatch(doc["contentSha256"])):
        errs.append("contentSha256 must be 64 lowercase hex chars")

    for key in ("source", "classification", "content", "pii", "provenance"):
        if key in doc and not isinstance(doc[key], dict):
            errs.append(f"{key} must be an object")
    cls = doc.get("classification")
    if isinstance(cls, dict):
        for k in ("docType", "gxp", "region"):
            if k not in cls:
                errs.append(f"classification.{k} is required")
    content = doc.get("content")
    if isinstance(content, dict):
        if not isinstance(content.get("text"), str):
            errs.append("content.text must be a string")
        if not isinstance(content.get("fields"), list):
            errs.append("content.fields must be an array")
        if not isinstance(content.get("tables"), list):
            errs.append("content.tables must be an array")
        pages = content.get("pages")
        if pages is not None:
            if not isinstance(pages, list) or not all(
                    isinstance(p, dict) and isinstance(p.get("pageNumber"), int)
                    and isinstance(p.get("text"), str) for p in pages):
                errs.append("content.pages must be an array of {pageNumber:int, text:str}")
    if errs:
        raise FinalizedDocumentError(errs)


def compact_fields(fields: List[Any]) -> Dict[str, Any]:
    """name -> value, values truncated, for filtering / display on chunks."""
    out: Dict[str, Any] = {}
    for f in fields[:FIELDS_MAX]:
        if not isinstance(f, dict) or not isinstance(f.get("name"), str):
            continue
        val = f.get("value")
        if isinstance(val, str):
            val = val[:FIELD_VALUE_MAX]
        elif not (val is None or isinstance(val, (bool, int, float))):
            val = json.dumps(val, ensure_ascii=False, default=str)[:FIELD_VALUE_MAX]
        out[f["name"][:FIELD_VALUE_MAX]] = val
    return out


def point_id(tenant_id: str, finalized_document_id: str, version: int, chunk_index: int) -> str:
    """Deterministic point id (tenant-scoped so tenants can never collide)."""
    key = f"finalized:{tenant_id}:{finalized_document_id}:v{version}:c{chunk_index}"
    return str(uuid.UUID(hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]))


def chunk_document(content: Dict[str, Any], chunker) -> List[Tuple[Optional[int], str]]:
    """[(page_num or None, chunk_text)] — per page when pages are given."""
    out: List[Tuple[Optional[int], str]] = []
    pages = content.get("pages") or []
    if pages:
        for p in pages:
            for ch in chunker(p["text"]):
                if ch["text"].strip():
                    out.append((p["pageNumber"], ch["text"]))
    else:
        for ch in chunker(content.get("text", "")):
            if ch["text"].strip():
                out.append((None, ch["text"]))
    return out


def build_payloads(doc: Dict[str, Any], tenant_id: str, chunker) -> List[Tuple[str, Dict[str, Any]]]:
    """[(point_id, payload)] for every chunk of the document."""
    content = doc["content"]
    cls     = doc.get("classification") or {}
    source  = doc.get("source") or {}
    fields  = compact_fields(content.get("fields") or [])
    out: List[Tuple[str, Dict[str, Any]]] = []
    for ci, (page_num, text) in enumerate(chunk_document(content, chunker)):
        payload: Dict[str, Any] = {
            "text":                  text,
            "doc_id":                doc["documentId"],
            "finalized_document_id": doc["finalizedDocumentId"],
            "version":               doc["version"],
            "tenant_id":             tenant_id,
            "doc_type":              cls.get("docType"),
            "gxp":                   cls.get("gxp"),
            "region":                cls.get("region"),
            "chunk_index":           ci,
            "source_sha256":         source.get("sha256"),
            "content_sha256":        doc["contentSha256"],
            "fields":                fields,
        }
        if page_num is not None:
            payload["page_num"] = page_num
        out.append((point_id(tenant_id, doc["finalizedDocumentId"], doc["version"], ci), payload))
    return out
