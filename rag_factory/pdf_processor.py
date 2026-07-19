# -*- coding: utf-8 -*-
"""
AI RAG Factory — Incremental PDF Processor
==========================================
Industry-grade PDF ingestion with per-page content hashing.

Processing flow per upload:
  1. Extract text per page (PyMuPDF / pdfminer fallback)
  2. SHA-256 hash each page's content
  3. Load persisted hash index (.index/<collection>_index.json)
  4. Classify every page: ADDED | UPDATED | UNCHANGED | DELETED
  5. Skip UNCHANGED pages   -- zero re-embedding cost
  6. Delete Qdrant chunks for UPDATED / DELETED pages (by stored chunk IDs)
  7. Chunk + embed + upsert ADDED / UPDATED pages
  8. Persist updated hash index

Hash index schema (per collection):
  {
    "doc_id":         "medicaid_2024.pdf",
    "collection":     "pdf_medicaid",
    "total_pages":    12,
    "last_processed": "2026-07-18T10:30:00",
    "pages": {
      "0": {"hash": "abc123...", "chunk_ids": ["uuid1", "uuid2"], "char_count": 512},
      ...
    }
  }
"""
from __future__ import annotations
import hashlib, json, os, uuid as _uuid_mod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ─── result types ─────────────────────────────────────────────────────────────
@dataclass
class PageRecord:
    page_num:   int
    text:       str
    char_count: int
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingReport:
    collection_name:  str
    doc_id:           str
    pdf_name:         str
    pages_total:      int
    pages_added:      int
    pages_updated:    int
    pages_skipped:    int
    pages_deleted:    int
    chunks_embedded:  int
    chunks_deleted:   int
    elapsed_ms:       int
    incremental:      bool  # True if hash index existed before this run

    def summary(self) -> str:
        tag = "INCREMENTAL" if self.incremental else "FULL"
        return (
            f"[{tag}] {self.pdf_name} -> {self.collection_name} | "
            f"+{self.pages_added} added, ~{self.pages_updated} updated, "
            f"={self.pages_skipped} skipped, -{self.pages_deleted} deleted | "
            f"{self.chunks_embedded} chunks embedded in {self.elapsed_ms}ms"
        )


# ─── PDF extraction ───────────────────────────────────────────────────────────
def _extract_pages_pymupdf(pdf_path: str) -> List[PageRecord]:
    import fitz  # PyMuPDF
    records = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            try:
                label = page.get_label() or str(i + 1)
            except Exception:
                label = str(i + 1)
            records.append(PageRecord(
                page_num=i, text=text, char_count=len(text),
                metadata={"page_label": label},
            ))
    return records


def _extract_pages_image(img_path: str) -> List[PageRecord]:
    """OCR a single image file — returns one PageRecord."""
    from .ocr.local_engine import ocr_file_local
    r = ocr_file_local(img_path, mode="forms")
    return [PageRecord(
        page_num=0, text=r.raw_text, char_count=len(r.raw_text),
        metadata={"method": r.method},
    )]


def _extract_pages_pdfminer(pdf_path: str) -> List[PageRecord]:
    from pdfminer.high_level import extract_pages as pm_extract
    from pdfminer.layout     import LTTextContainer
    records = []
    for i, layout in enumerate(pm_extract(pdf_path)):
        text = "".join(
            el.get_text() for el in layout if isinstance(el, LTTextContainer)
        ).strip()
        records.append(PageRecord(page_num=i, text=text, char_count=len(text)))
    return records


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}


def extract_pages(pdf_path: str) -> List[PageRecord]:
    """Extract pages; auto-routes images through local OCR, PDFs through PyMuPDF."""
    ext = os.path.splitext(pdf_path)[1].lower()
    if ext in _IMAGE_EXTS:
        return _extract_pages_image(pdf_path)
    try:
        return _extract_pages_pymupdf(pdf_path)
    except ImportError:
        return _extract_pages_pdfminer(pdf_path)


# ─── hashing ──────────────────────────────────────────────────────────────────
def _page_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _chunk_id(collection: str, doc_id: str, page_num: int, chunk_idx: int, text: str) -> str:
    key = f"{collection}:{doc_id}:p{page_num}:c{chunk_idx}:{text[:80]}"
    return str(_uuid_mod.UUID(hashlib.sha256(key.encode()).hexdigest()[:32]))


# ─── fixed chunker (page-aware) ───────────────────────────────────────────────
def _chunk_page(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


# ─── hash index persistence ───────────────────────────────────────────────────
def _index_path(base_dir: str, collection: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{collection}_index.json")


def _load_index(base_dir: str, collection: str) -> Dict[str, Any]:
    path = _index_path(base_dir, collection)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"collection": collection, "pages": {}}


def _save_index(base_dir: str, collection: str, index: Dict[str, Any]) -> None:
    import datetime
    index["last_processed"] = datetime.datetime.utcnow().isoformat()
    path = _index_path(base_dir, collection)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


# ─── Qdrant helpers ───────────────────────────────────────────────────────────
def _ensure_collection(qdrant, collection: str, dim: int = 1024) -> None:
    from qdrant_client.models import VectorParams, Distance
    existing = {c.name for c in qdrant.get_collections().collections}
    if collection not in existing:
        qdrant.create_collection(
            collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def _delete_chunks_by_ids(qdrant, collection: str, chunk_ids: List[str]) -> int:
    if not chunk_ids:
        return 0
    from qdrant_client.models import PointIdsList
    qdrant.delete(
        collection_name=collection,
        points_selector=PointIdsList(points=chunk_ids),
    )
    return len(chunk_ids)


# ─── main processor ───────────────────────────────────────────────────────────
class IncrementalPDFProcessor:
    """
    Processes a PDF file incrementally.
    Re-embeds only pages whose content changed since the last run.
    Stores a per-collection hash index in `index_dir`.
    """

    def __init__(self, index_dir: str = ".index", chunk_size: int = 400, overlap: int = 50):
        self.index_dir  = index_dir
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def process(
        self,
        pdf_path:        str,
        collection_name: str,
        doc_id:          Optional[str] = None,
        tenant_id:       Optional[str] = None,
    ) -> ProcessingReport:
        import time
        from .components.base import embed, get_qdrant_client

        t0     = time.time()
        qdrant = get_qdrant_client()
        pdf_name = os.path.basename(pdf_path)
        if doc_id is None:
            doc_id = hashlib.sha256(pdf_name.encode()).hexdigest()[:16]

        # ── 1. Extract ──────────────────────────────────────────────────────
        pages = extract_pages(pdf_path)

        # ── 2. Load index ───────────────────────────────────────────────────
        index       = _load_index(self.index_dir, collection_name)
        was_indexed = bool(index.get("pages"))
        old_pages   = index.get("pages", {})

        # ── 3. Classify pages ───────────────────────────────────────────────
        added, updated, skipped, deleted_page_nums = [], [], [], []

        for rec in pages:
            key     = str(rec.page_num)
            ph      = _page_hash(rec.text)
            old     = old_pages.get(key)
            if old is None:
                added.append(rec)
            elif old["hash"] != ph:
                updated.append(rec)
            else:
                skipped.append(rec)

        # Pages that existed before but no longer exist in the PDF
        current_keys = {str(r.page_num) for r in pages}
        for k in old_pages:
            if k not in current_keys:
                deleted_page_nums.append(int(k))

        # ── 4. Delete chunks for updated + deleted pages ────────────────────
        _ensure_collection(qdrant, collection_name)
        total_deleted = 0
        for page_num in [r.page_num for r in updated] + deleted_page_nums:
            ids = old_pages.get(str(page_num), {}).get("chunk_ids", [])
            total_deleted += _delete_chunks_by_ids(qdrant, collection_name, ids)
            index["pages"].pop(str(page_num), None)

        # ── 5. Embed + upsert added / updated pages ─────────────────────────
        from qdrant_client.models import PointStruct
        total_embedded = 0

        for rec in added + updated:
            if not rec.text.strip():
                # Empty page — store as placeholder, no embedding
                index["pages"][str(rec.page_num)] = {
                    "hash":       _page_hash(rec.text),
                    "chunk_ids":  [],
                    "char_count": 0,
                }
                continue

            raw_chunks = _chunk_page(rec.text, self.chunk_size, self.overlap)
            points     = []
            chunk_ids  = []

            for ci, chunk_text in enumerate(raw_chunks):
                cid = _chunk_id(collection_name, doc_id, rec.page_num, ci, chunk_text)
                vec = embed(chunk_text)
                payload: Dict[str, Any] = {
                    "text":        chunk_text,
                    "doc_id":      doc_id,
                    "pdf_name":    pdf_name,
                    "page_num":    rec.page_num,
                    "page_label":  rec.metadata.get("page_label", str(rec.page_num + 1)),
                    "chunk_index": ci,
                    "page_hash":   _page_hash(rec.text),
                }
                if tenant_id:
                    payload["tenant_id"] = tenant_id
                points.append(PointStruct(id=cid, vector=vec, payload=payload))
                chunk_ids.append(cid)

            if points:
                qdrant.upsert(collection_name=collection_name, points=points)

            index["pages"][str(rec.page_num)] = {
                "hash":       _page_hash(rec.text),
                "chunk_ids":  chunk_ids,
                "char_count": rec.char_count,
            }
            total_embedded += len(points)

        # ── 6. Update index metadata ─────────────────────────────────────────
        index["doc_id"]       = doc_id
        index["collection"]   = collection_name
        index["total_pages"]  = len(pages)
        _save_index(self.index_dir, collection_name, index)

        elapsed = int((time.time() - t0) * 1000)

        return ProcessingReport(
            collection_name = collection_name,
            doc_id          = doc_id,
            pdf_name        = pdf_name,
            pages_total     = len(pages),
            pages_added     = len(added),
            pages_updated   = len(updated),
            pages_skipped   = len(skipped),
            pages_deleted   = len(deleted_page_nums),
            chunks_embedded = total_embedded,
            chunks_deleted  = total_deleted,
            elapsed_ms      = elapsed,
            incremental     = was_indexed,
        )

    def get_index_stats(self, collection_name: str) -> Dict[str, Any]:
        """Return index metadata for a collection without processing."""
        index = _load_index(self.index_dir, collection_name)
        pages = index.get("pages", {})
        return {
            "collection":      collection_name,
            "doc_id":          index.get("doc_id"),
            "total_pages":     index.get("total_pages", 0),
            "indexed_pages":   len(pages),
            "total_chunks":    sum(len(p.get("chunk_ids", [])) for p in pages.values()),
            "last_processed":  index.get("last_processed"),
        }

    def reset_index(self, collection_name: str) -> None:
        """Force full re-processing on next run by deleting the hash index."""
        path = _index_path(self.index_dir, collection_name)
        if os.path.isfile(path):
            os.remove(path)
