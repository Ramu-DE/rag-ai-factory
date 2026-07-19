# -*- coding: utf-8 -*-
"""
ID Document Skill
=================
Uses AWS Textract AnalyzeID for structured field extraction.
Handles: passports, driver's licences, national ID cards.

Extracted fields:
  document_type, id_number, first_name, last_name, date_of_birth,
  expiry_date, issue_date, issuing_country, issuing_state, address, mrz
"""
from __future__ import annotations
from typing import Any
from .base import BaseDocumentSkill, ExtractionResult

_REQUIRED = ["id_number", "last_name", "date_of_birth"]

_TEXTRACT_ID_MAP = {
    "DOCUMENT_NUMBER":  "id_number",
    "FIRST_NAME":       "first_name",
    "LAST_NAME":        "last_name",
    "DATE_OF_BIRTH":    "date_of_birth",
    "EXPIRATION_DATE":  "expiry_date",
    "DATE_OF_ISSUE":    "issue_date",
    "PLACE_OF_BIRTH":   "place_of_birth",
    "ADDRESS":          "address",
    "COUNTY":           "county",
    "STATE_IN_ADDRESS": "state",
    "MRZ_CODE":         "mrz",
    "ID_TYPE":          "document_type",
}


class IDCardSkill(BaseDocumentSkill):
    doc_type        = "id_document"
    required_fields = _REQUIRED
    description     = "Extracts structured fields from passports, driver licences, and national ID cards via Textract AnalyzeID"

    def run(self, doc: Any) -> ExtractionResult:
        fields = {}

        # Use Textract AnalyzeID fields when available
        if doc.id_fields:
            for textract_key, mapped in _TEXTRACT_ID_MAP.items():
                val = doc.id_fields.get(textract_key)
                if val:
                    fields[mapped] = val
            confidence = 0.92
        else:
            # Fallback: parse from raw text
            text = doc.full_text
            for line in text.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    k = k.strip().lower().replace(" ", "_")
                    v = v.strip()
                    if k and v:
                        fields[k] = v
            confidence = 0.55

        result = ExtractionResult(
            doc_type=self.doc_type,
            fields=fields,
            confidence=confidence,
            raw_text=doc.full_text,
        )
        return self._validate(result)
