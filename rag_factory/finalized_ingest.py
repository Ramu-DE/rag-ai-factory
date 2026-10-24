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


def chunk_document(content: Dict[str, Any], chunker) -> List[Tuple[Optional[int], Dict[str, Any]]]:
    """[(page_num or None, chunk)] — per page when pages are given.

    A chunk is the chunker's dict: at least {"text"}; sentence-window chunkers
    also carry "window_text", the surrounding sentences used at answer time.
    """
    out: List[Tuple[Optional[int], Dict[str, Any]]] = []
    pages = content.get("pages") or []
    if pages:
        for p in pages:
            for ch in chunker(p["text"]):
                if ch["text"].strip():
                    out.append((p["pageNumber"], ch))
    else:
        for ch in chunker(content.get("text", "")):
            if ch["text"].strip():
                out.append((None, ch))
    return out


FIELDS_PER_CHUNK = 40
ROWS_PER_CHUNK   = 20


def _cell(v: Any) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s.replace("|", "/").replace("\n", " ")[:FIELD_VALUE_MAX]


def structured_chunks(content: Dict[str, Any], doc_type: Optional[str]) -> List[Tuple[str, str]]:
    """[(chunk_kind, text)] for the reviewed fields and tables.

    Extracted fields and tables are the most precise part of a finalized
    document, and plain text chunking dilutes them across arbitrary windows.
    Here each group of fields, and each block of table rows (with its header
    repeated), becomes a self-contained chunk that retrieval can hit directly.
    """
    out: List[Tuple[str, str]] = []
    label = f" ({doc_type})" if doc_type else ""
    fields = [f for f in (content.get("fields") or []) if isinstance(f, dict) and isinstance(f.get("name"), str)]
    for i in range(0, len(fields), FIELDS_PER_CHUNK):
        lines = [f"{f['name']}: {_cell(f.get('value'))}" for f in fields[i:i + FIELDS_PER_CHUNK]]
        out.append(("fields", f"Document fields{label}:\n" + "\n".join(lines)))
    for ti, t in enumerate(content.get("tables") or []):
        if not isinstance(t, dict):
            continue
        cols = [_cell(c) for c in (t.get("columns") or [])]
        rows = [r for r in (t.get("rows") or []) if isinstance(r, list)]
        name = t.get("name") or f"Table {ti + 1}"
        head = f"{name}{label}\n| " + " | ".join(cols) + " |"
        for ri in range(0, max(len(rows), 1), ROWS_PER_CHUNK):
            body = ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows[ri:ri + ROWS_PER_CHUNK]]
            out.append(("table", head + ("\n" + "\n".join(body) if body else "")))
    return out


def build_payloads(doc: Dict[str, Any], tenant_id: str, chunker,
                   structured: bool = False, chunker_name: Optional[str] = None
                   ) -> List[Tuple[str, Dict[str, Any]]]:
    """[(point_id, payload)] for every chunk of the document.

    `structured=True` appends field and table chunks (see structured_chunks)
    after the text chunks; `chunk_kind` tells them apart.
    """
    content = doc["content"]
    cls     = doc.get("classification") or {}
    source  = doc.get("source") or {}
    fields  = compact_fields(content.get("fields") or [])
    items: List[Tuple[Optional[int], str, Dict[str, Any]]] = [
        (page_num, "text", ch) for page_num, ch in chunk_document(content, chunker)
    ]
    if structured:
        items += [(None, kind, {"text": text}) for kind, text in structured_chunks(content, cls.get("docType"))]
    out: List[Tuple[str, Dict[str, Any]]] = []
    for ci, (page_num, kind, ch) in enumerate(items):
        payload: Dict[str, Any] = {
            "text":                  ch["text"],
            "doc_id":                doc["documentId"],
            "finalized_document_id": doc["finalizedDocumentId"],
            "version":               doc["version"],
            "tenant_id":             tenant_id,
            "doc_type":              cls.get("docType"),
            "gxp":                   cls.get("gxp"),
            "region":                cls.get("region"),
            "chunk_index":           ci,
            "chunk_kind":            kind,
            "source_sha256":         source.get("sha256"),
            "content_sha256":        doc["contentSha256"],
            "fields":                fields,
        }
        if chunker_name:
            payload["chunker"] = chunker_name
        if isinstance(ch.get("window_text"), str):
            payload["window_text"] = ch["window_text"]
        if page_num is not None:
            payload["page_num"] = page_num
        out.append((point_id(tenant_id, doc["finalizedDocumentId"], doc["version"], ci), payload))
    return out
