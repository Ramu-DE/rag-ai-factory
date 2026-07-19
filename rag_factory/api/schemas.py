# -*- coding: utf-8 -*-
"""Request / Response Pydantic models for the factory API."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text            : str           = Field(..., description="Raw document text to ingest")
    collection_name : str           = Field(..., description="Target Qdrant collection")
    chunker         : str           = Field("fixed_chunking")
    doc_id          : str           = Field("doc_001")
    tenant_id       : Optional[str] = None
    use_incremental : bool          = False


class IngestResponse(BaseModel):
    collection_name : str
    chunk_count     : int
    pages_changed   : Optional[int] = None
    elapsed_ms      : int


class QueryRequest(BaseModel):
    query           : str
    collection_name : str
    spec            : Optional[str] = None   # path to YAML spec file
    top_k           : int           = Field(5, ge=1, le=50)
    tenant_id       : Optional[str] = None
    ground_truth    : Optional[str] = None   # for evaluation


class QueryResponse(BaseModel):
    answer          : str
    retrieved_chunks: List[Dict[str, Any]] = []
    scores          : List[float]          = []
    guard_log       : Dict[str, Any]       = {}
    faithfulness    : Optional[float]      = None
    elapsed_ms      : int                  = 0


class EvaluateRequest(BaseModel):
    query           : str
    answer          : str
    retrieved_chunks: List[Dict[str, Any]]
    ground_truth    : str


class EvaluateResponse(BaseModel):
    faithfulness      : float
    answer_relevancy  : float
    context_precision : float
    context_recall    : float


class CompareRequest(BaseModel):
    query           : str
    collection_name : str
    spec_a          : str   # YAML spec path or name
    spec_b          : str
    ground_truth    : Optional[str] = None


class CompareResponse(BaseModel):
    spec_a          : str
    spec_b          : str
    answer_a        : str
    answer_b        : str
    faithfulness_a  : Optional[float] = None
    faithfulness_b  : Optional[float] = None
    elapsed_ms_a    : int = 0
    elapsed_ms_b    : int = 0


class SpecListResponse(BaseModel):
    specs           : List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status          : str = "ok"
    manifest_size   : int = 0
    temporal        : bool = False


# ── IDP schemas ───────────────────────────────────────────────────────────────
class IDPClassifyRequest(BaseModel):
    text        : str           = Field(..., description="Document text to classify")
    filename    : str           = Field("", description="Original filename for filename-based hints")


class IDPClassifyResponse(BaseModel):
    doc_type        : str
    confidence      : float
    heuristic       : bool
    reason          : str
    extraction_mode : str


class IDPProcessRequest(BaseModel):
    collection_name   : str            = Field(..., description="Target Qdrant collection")
    doc_type_override : Optional[str]  = None
    extraction_mode   : str            = "auto"
    chunk_size        : int            = Field(400, ge=100, le=1500)
    force_reindex     : bool           = False
    tenant_id         : Optional[str]  = None


class IDPFileResult(BaseModel):
    file_name       : str
    doc_type        : str
    confidence      : float
    is_scanned      : bool
    method          : str
    pages_total     : int
    pages_added     : int
    pages_updated   : int
    pages_skipped   : int
    chunks_embedded : int
    chunks_deleted  : int
    elapsed_ms      : int
    incremental     : bool
    fields          : Dict[str, str]           = {}
    tables          : List[Dict[str, Any]]     = []
    extraction_confidence : Optional[float]    = None
    validation_valid      : Optional[bool]     = None
    validation_errors     : List[str]          = []
    validation_warnings   : List[str]          = []
    error           : Optional[str]            = None   # set if this file failed


class IDPBatchResponse(BaseModel):
    collection_name : str
    files_submitted : int
    files_succeeded : int
    files_failed    : int
    total_pages     : int
    total_chunks    : int
    elapsed_ms      : int
    results         : List[IDPFileResult]


class IDPCollectionStats(BaseModel):
    collection_name : str
    total_chunks    : int
    indexed_pages   : int
    last_processed  : Optional[str]
    doc_id          : Optional[str]
    doc_types_seen  : List[str]     = []
