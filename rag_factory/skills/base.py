# -*- coding: utf-8 -*-
"""
BaseDocumentSkill — abstract class every document skill extends.

A skill takes an ExtractedDocument and returns an ExtractionResult:
  - fields    : Dict[str, str]        structured key-value pairs
  - tables    : List[Dict]            extracted tables as row-dicts
  - confidence: float                 overall extraction confidence 0-1
  - warnings  : List[str]             missing required fields, low confidence
  - doc_type  : str                   inferred document type
  - raw_text  : str                   full text for RAG ingestion

Usage:
    from rag_factory.skills import InvoiceSkill
    skill  = InvoiceSkill()
    result = skill.run(extracted_doc)
    print(result.fields["vendor_name"], result.fields["total_amount"])
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractionResult:
    doc_type:   str
    fields:     Dict[str, str]         = field(default_factory=dict)
    tables:     List[Dict[str, Any]]   = field(default_factory=list)
    confidence: float                  = 0.0
    warnings:   List[str]              = field(default_factory=list)
    raw_text:   str                    = ""
    metadata:   Dict[str, Any]         = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_type":   self.doc_type,
            "fields":     self.fields,
            "tables":     self.tables,
            "confidence": round(self.confidence, 3),
            "warnings":   self.warnings,
        }

    def missing_required(self, required: List[str]) -> List[str]:
        return [k for k in required if not self.fields.get(k)]


class BaseDocumentSkill(ABC):
    """Every document skill inherits from this."""

    doc_type:        str = "unknown"
    required_fields: List[str] = []
    description:     str = ""

    @abstractmethod
    def run(self, doc: Any) -> ExtractionResult:
        """
        Process an ExtractedDocument and return an ExtractionResult.
        doc — ExtractedDocument from rag_factory.ocr.extractor
        """
        ...

    def _validate(self, result: ExtractionResult) -> ExtractionResult:
        """Add warnings for missing required fields."""
        for rf in self.required_fields:
            if not result.fields.get(rf):
                result.warnings.append(f"Missing required field: {rf}")
        return result

    def _llm_extract(self, text: str, prompt: str, max_tokens: int = 512) -> str:
        from rag_factory.components.base import llm_call
        return llm_call(prompt.format(text=text[:4000]), max_tokens=max_tokens, temperature=0.0)

    def _parse_kv(self, raw: str) -> Dict[str, str]:
        """Parse LLM output in KEY: VALUE format."""
        result: Dict[str, str] = {}
        for line in raw.strip().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip().lower().replace(" ", "_")
                v = v.strip()
                if k and v and v.lower() not in ("n/a", "not found", "none", "unknown", ""):
                    result[k] = v
        return result
