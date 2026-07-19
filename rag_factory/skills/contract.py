# -*- coding: utf-8 -*-
"""
Contract Skill
==============
LLM-driven extraction of key contract fields.
Works on any contract type: NDA, SaaS, employment, vendor, lease.

Extracted fields:
  party_1, party_2, effective_date, expiry_date, governing_law,
  contract_type, notice_period, payment_terms, termination_clause,
  auto_renewal, jurisdiction, liability_cap
"""
from __future__ import annotations
from typing import Any
from .base import BaseDocumentSkill, ExtractionResult

_REQUIRED = ["party_1", "party_2", "effective_date", "contract_type"]

_CLAUSES_PROMPT = """\
You are a contract analysis assistant.
Extract the following fields from the contract text below.
Return ONLY key: value pairs, one per line.
Fields: party_1, party_2, effective_date, expiry_date, governing_law,
contract_type, notice_period, payment_terms, termination_clause,
auto_renewal, jurisdiction, liability_cap.
If a field is not present write: key: N/A

Contract text:
{text}
"""

_OBLIGATIONS_PROMPT = """\
List the top 5 key obligations and restrictions from this contract.
Return as a numbered list (1. ...).

Contract text:
{text}
"""


class ContractSkill(BaseDocumentSkill):
    doc_type        = "contract"
    required_fields = _REQUIRED
    description     = "Extracts parties, dates, clauses, and obligations from legal contracts"

    def run(self, doc: Any) -> ExtractionResult:
        text = doc.full_rich_text

        # Field extraction
        raw    = self._llm_extract(text, _CLAUSES_PROMPT, max_tokens=600)
        fields = self._parse_kv(raw)

        # Key obligations (stored as a table row)
        obligations_raw = self._llm_extract(text, _OBLIGATIONS_PROMPT, max_tokens=400)
        tables = [{"obligations": obligations_raw}] if obligations_raw else []

        # Confidence: 0.80 if required fields found, 0.60 otherwise
        found    = sum(1 for f in _REQUIRED if fields.get(f))
        confidence = 0.60 + (found / len(_REQUIRED)) * 0.30

        result = ExtractionResult(
            doc_type=self.doc_type,
            fields=fields,
            tables=tables,
            confidence=confidence,
            raw_text=text,
        )
        return self._validate(result)
