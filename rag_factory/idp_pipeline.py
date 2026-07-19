# -*- coding: utf-8 -*-
"""
IDP Pipeline — full end-to-end document processing
====================================================
Wires together:
  1. OCR / extraction      (PyMuPDF for clean PDFs, Textract for scans)
  2. Document classification (heuristic → LLM)
  3. Skill execution        (invoice / contract / medical / id / custom)
  4. Field validation       (required fields, format rules, cross-field)
  5. Incremental indexing   (per-page hash → skip unchanged pages)
  6. RAG ingestion          (chunk rich_text → embed → upsert Qdrant)

Returned IDPResult:
  doc_type        : str
  classification  : ClassificationResult
  extraction      : ExtractionResult
  validation      : ValidationReport
  ingest_report   : ProcessingReport
  elapsed_ms      : int

This is the single call you make from the API or Streamlit UI.
"""
from __future__ import annotations
import os, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IDPResult:
    file_name:      str
    doc_type:       str
    classification: Any   # ClassificationResult
    extraction:     Any   # ExtractionResult  (None if no skill)
    validation:     Any   # ValidationReport  (None if no skill)
    ingest_report:  Any   # ProcessingReport
    elapsed_ms:     int
    is_scanned:     bool
    method:         str   # extraction method used

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "file_name":   self.file_name,
            "doc_type":    self.doc_type,
            "is_scanned":  self.is_scanned,
            "method":      self.method,
            "elapsed_ms":  self.elapsed_ms,
            "classification": self.classification.to_dict() if self.classification else {},
            "ingest":      {
                "collection":       self.ingest_report.collection_name,
                "pages_total":      self.ingest_report.pages_total,
                "pages_added":      self.ingest_report.pages_added,
                "pages_updated":    self.ingest_report.pages_updated,
                "pages_skipped":    self.ingest_report.pages_skipped,
                "chunks_embedded":  self.ingest_report.chunks_embedded,
                "incremental":      self.ingest_report.incremental,
            },
        }
        if self.extraction:
            d["extraction"] = self.extraction.to_dict()
        if self.validation:
            d["validation"] = self.validation.to_dict()
        return d


class IDPPipeline:
    """
    End-to-end IDP pipeline.

    Usage:
        pipeline = IDPPipeline(collection_name="my_docs", index_dir=".index")
        result   = pipeline.process("invoice.pdf")
        print(result.extraction.fields["vendor_name"])
        print(result.validation.valid)
    """

    def __init__(
        self,
        collection_name:  str  = "idp_documents",
        index_dir:        str  = ".index",
        chunk_size:       int  = 400,
        overlap:          int  = 50,
        scan_threshold:   int  = 100,
        force_reindex:    bool = False,
        tenant_id:        Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.index_dir       = index_dir
        self.chunk_size      = chunk_size
        self.overlap         = overlap
        self.scan_threshold  = scan_threshold
        self.force_reindex   = force_reindex
        self.tenant_id       = tenant_id

    def process(
        self,
        file_path:          str,
        doc_type_override:  Optional[str] = None,
        extraction_mode:    str = "auto",
    ) -> IDPResult:
        t0 = time.time()

        # ── 1. Extract ────────────────────────────────────────────────────
        from .ocr.extractor import extract_document
        extracted = extract_document(
            file_path=file_path,
            extraction_mode=extraction_mode,
            scan_threshold=self.scan_threshold,
        )

        # ── 2. Classify ───────────────────────────────────────────────────
        from .document_classifier import CLASSIFIER
        fname = os.path.basename(file_path)
        if doc_type_override:
            from .document_classifier import ClassificationResult, _DOC_TYPE_TO_MODE
            classification = ClassificationResult(
                doc_type=doc_type_override,
                confidence=1.0, heuristic=True,
                reason="Manual override.",
                extraction_mode=_DOC_TYPE_TO_MODE.get(doc_type_override, "auto"),
            )
        else:
            # Pass filename so classifier can use it as a strong prior
            classification = CLASSIFIER.classify(extracted.full_text, filename=fname)

        doc_type = classification.doc_type

        # ── 3. Re-extract with correct Textract mode if needed ────────────
        if (classification.extraction_mode in ("expense", "id")
                and extracted.method == "pymupdf"):
            extracted = extract_document(
                file_path=file_path,
                extraction_mode=classification.extraction_mode,
            )

        # ── 4. Run skill — only for doc types that have a skill ───────────
        from .skills.registry import get_skill
        _SKILL_DOC_TYPES = {"invoice", "contract", "medical", "id_document"}
        skill = get_skill(doc_type) if doc_type in _SKILL_DOC_TYPES else None
        extraction = skill.run(extracted) if skill else None

        # ── 5. Validate — only when a skill ran ───────────────────────────
        from .field_validator import VALIDATOR
        if extraction and doc_type in _SKILL_DOC_TYPES:
            validation = VALIDATOR.validate(
                fields=extraction.fields,
                doc_type=doc_type,
                confidence=extraction.confidence,
            )
        else:
            validation = None

        # ── 6. Incremental ingest into Qdrant ─────────────────────────────
        ingest_report = self._ingest(extracted, file_path)

        elapsed = int((time.time() - t0) * 1000)

        return IDPResult(
            file_name=os.path.basename(file_path),
            doc_type=doc_type,
            classification=classification,
            extraction=extraction,
            validation=validation,
            ingest_report=ingest_report,
            elapsed_ms=elapsed,
            is_scanned=extracted.is_scanned,
            method=extracted.method,
        )

    def _ingest(self, extracted, file_path: str):
        """Chunk rich_text per page and incrementally ingest into Qdrant."""
        import hashlib, tempfile
        from .pdf_processor import IncrementalPDFProcessor

        proc = IncrementalPDFProcessor(
            index_dir=self.index_dir,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        # For OCR results write a temporary plain-text file per page
        # (IncrementalPDFProcessor normally handles PDFs via PyMuPDF;
        #  for Textract results we synthesise a text-only temp PDF)
        if extracted.method.startswith("textract"):
            return self._ingest_textract_pages(extracted, proc, file_path)

        # PyMuPDF extraction — process original file directly
        if self.force_reindex:
            proc.reset_index(self.collection_name)

        return proc.process(
            pdf_path=file_path,
            collection_name=self.collection_name,
            doc_id=os.path.basename(file_path).replace(" ", "_"),
            tenant_id=self.tenant_id,
        )

    def _ingest_textract_pages(self, extracted, proc, original_path: str):
        """
        Ingest Textract-extracted pages as rich text.
        Writes a temporary text file and re-uses the incremental processor logic.
        """
        import time as _time
        from .components.base import embed, get_qdrant_client, chunk_id
        from .pdf_processor import _page_hash, _chunk_page, _ensure_collection, _load_index, _save_index, _index_path, _delete_chunks_by_ids, ProcessingReport
        from qdrant_client.models import PointStruct

        t0     = _time.time()
        qdrant = get_qdrant_client()
        fname  = os.path.basename(original_path).replace(" ", "_")
        cname  = self.collection_name

        if self.force_reindex:
            proc.reset_index(cname)

        index      = _load_index(self.index_dir, cname)
        was_indexed = bool(index.get("pages"))
        old_pages  = index.get("pages", {})

        _ensure_collection(qdrant, cname)

        added = updated = skipped = deleted_count = 0
        total_embedded = total_deleted = 0

        for page in extracted.pages:
            key  = str(page.page_num)
            text = page.rich_text or page.text
            ph   = _page_hash(text)
            old  = old_pages.get(key)

            if old is not None and old["hash"] == ph:
                skipped += 1
                continue

            if old is not None:
                total_deleted += _delete_chunks_by_ids(qdrant, cname, old.get("chunk_ids", []))
                updated += 1
            else:
                added += 1

            if not text.strip():
                index["pages"][key] = {"hash": ph, "chunk_ids": [], "char_count": 0}
                continue

            raw_chunks = _chunk_page(text, self.chunk_size, self.overlap)
            points, chunk_ids = [], []

            for ci, chunk_text in enumerate(raw_chunks):
                from .pdf_processor import _chunk_id
                cid = _chunk_id(cname, fname, page.page_num, ci, chunk_text)
                vec = embed(chunk_text)
                payload = {
                    "text":       chunk_text,
                    "doc_id":     fname,
                    "pdf_name":   fname,
                    "page_num":   page.page_num,
                    "chunk_index": ci,
                    "page_hash":  ph,
                    "extraction_method": extracted.method,
                }
                if self.tenant_id:
                    payload["tenant_id"] = self.tenant_id
                points.append(PointStruct(id=cid, vector=vec, payload=payload))
                chunk_ids.append(cid)

            if points:
                qdrant.upsert(collection_name=cname, points=points)
            index["pages"][key] = {"hash": ph, "chunk_ids": chunk_ids, "char_count": len(text)}
            total_embedded += len(points)

        # Handle deleted pages
        current_keys = {str(p.page_num) for p in extracted.pages}
        for k in list(old_pages.keys()):
            if k not in current_keys:
                total_deleted += _delete_chunks_by_ids(qdrant, cname, old_pages[k].get("chunk_ids",[]))
                index["pages"].pop(k, None)
                deleted_count += 1

        index["doc_id"]      = fname
        index["collection"]  = cname
        index["total_pages"] = len(extracted.pages)
        _save_index(self.index_dir, cname, index)

        from .pdf_processor import ProcessingReport
        return ProcessingReport(
            collection_name=cname, doc_id=fname, pdf_name=fname,
            pages_total=len(extracted.pages),
            pages_added=added, pages_updated=updated,
            pages_skipped=skipped, pages_deleted=deleted_count,
            chunks_embedded=total_embedded, chunks_deleted=total_deleted,
            elapsed_ms=int((_time.time() - t0) * 1000),
            incremental=was_indexed,
        )
