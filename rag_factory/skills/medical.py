# -*- coding: utf-8 -*-
"""
Medical Document Skill
======================
Handles two sub-types automatically:

  CLINICAL RECORD  — patient notes, lab results, prescriptions, discharge
    → extracts: patient_name, provider_name, diagnoses, medications, plan

  MEDICAL REPORT   — research papers, textbooks, informatics docs, guidelines
    → extracts: title, authors, document_date, abstract, key_findings, topics

Sub-type detection: tries clinical extraction first; if >=2 required clinical
fields are missing → falls back to report-style extraction.
"""
from __future__ import annotations
from typing import Any
from .base import BaseDocumentSkill, ExtractionResult

_CLINICAL_REQUIRED = ["patient_name", "document_date", "provider_name"]
_REPORT_REQUIRED   = ["title", "document_date"]

_CLINICAL_PROMPT = """\
You are a medical document extraction assistant.
Extract the following fields from the clinical document text.
Return ONLY key: value pairs, one per line.
For list fields (medications, diagnosis_codes, procedure_codes) join values with semicolons.
Fields: patient_name, patient_dob, patient_id, provider_name, provider_npi,
facility_name, document_date, diagnosis_codes, procedure_codes, medications,
chief_complaint, assessment, plan.
If a field is not present write: key: N/A

Document text:
{text}
"""

_REPORT_PROMPT = """\
You are a document extraction assistant for medical research and informatics documents.
Extract the following fields from the document text.
Return ONLY key: value pairs, one per line.
Fields: title, authors, document_date, publisher, abstract,
key_findings, topics, keywords, chapter_count, target_audience, version.
If a field is not present write: key: N/A

Document text:
{text}
"""

import re

# Patterns that appear in actual clinical records (not textbooks about them)
_CLINICAL_PATTERNS = [
    re.compile(r"patient\s*(name|id|dob|no\.?|number)\s*:", re.I),     # "Patient Name:" label
    re.compile(r"chief\s*complaint\s*:", re.I),
    re.compile(r"(?:diagnosis|assessment|plan)\s*:", re.I),
    re.compile(r"\bRx\s*:", re.I),
    re.compile(r"(?:physician|provider|npi)\s*:", re.I),
    re.compile(r"discharge\s*summary", re.I),
    re.compile(r"lab\s*result", re.I),
    re.compile(r"\bICD-1[0-9]\b.*[A-Z]\d{2}", re.I),                   # ICD code + actual code value
    re.compile(r"date\s*of\s*birth\s*:", re.I),
    re.compile(r"dob\s*:", re.I),
    re.compile(r"medications?\s*:", re.I),
    re.compile(r"prescribed\s*:", re.I),
]

# Report/textbook signals — if these dominate, treat as report
_REPORT_PATTERNS = [
    re.compile(r"\b(chapter|edition|textbook|handbook|introduction to|overview of)\b", re.I),
    re.compile(r"\b(table of contents|bibliography|references|appendix)\b", re.I),
    re.compile(r"\b(healthcare informatics|health information|information systems)\b", re.I),
    re.compile(r"published\s+\d{4}|copyright\s+\d{4}|isbn\s*:", re.I),
]


def _is_clinical_record(text: str) -> bool:
    """True if document looks like a patient record, False if it looks like a report/textbook."""
    sample = text[:4000]
    clinical_hits = sum(1 for p in _CLINICAL_PATTERNS if p.search(sample))
    report_hits   = sum(1 for p in _REPORT_PATTERNS   if p.search(sample))
    # Clinical if >=3 clinical signals and not dominated by report signals
    return clinical_hits >= 3 and report_hits <= 1


class MedicalSkill(BaseDocumentSkill):
    doc_type        = "medical"
    required_fields = _CLINICAL_REQUIRED
    description     = "Extracts patient/provider fields from clinical records; falls back to report extraction for medical research documents"

    def run(self, doc: Any) -> ExtractionResult:
        text = doc.full_rich_text
        if _is_clinical_record(text):
            return self._extract_clinical(doc, text)
        return self._extract_report(doc, text)

    def _extract_clinical(self, doc: Any, text: str) -> ExtractionResult:
        raw    = self._llm_extract(text, _CLINICAL_PROMPT, max_tokens=700)
        fields = self._parse_kv(raw)

        # Supplement with OCR form fields
        for ff in doc.all_forms:
            k = ff.key.strip().lower().replace(" ", "_")
            if k and ff.value and k not in fields:
                fields[k] = ff.value

        found      = sum(1 for f in _CLINICAL_REQUIRED if fields.get(f))
        confidence = 0.60 + (found / len(_CLINICAL_REQUIRED)) * 0.30

        # If >=2 required clinical fields still missing, fall back to report
        missing = [f for f in _CLINICAL_REQUIRED if not fields.get(f)]
        if len(missing) >= 2:
            return self._extract_report(doc, text, forced=True)

        self.required_fields = _CLINICAL_REQUIRED
        result = ExtractionResult(
            doc_type=self.doc_type,
            fields=fields, tables=[],
            confidence=confidence, raw_text=text,
            metadata={"sub_type": "clinical_record"},
        )
        return self._validate(result)

    def _extract_report(self, doc: Any, text: str, forced: bool = False) -> ExtractionResult:
        raw    = self._llm_extract(text, _REPORT_PROMPT, max_tokens=600)
        fields = self._parse_kv(raw)

        # Supplement with OCR form fields
        for ff in doc.all_forms:
            k = ff.key.strip().lower().replace(" ", "_")
            if k and ff.value and k not in fields:
                fields[k] = ff.value

        found      = sum(1 for f in _REPORT_REQUIRED if fields.get(f))
        confidence = 0.65 + (found / len(_REPORT_REQUIRED)) * 0.25
        if forced:
            confidence = max(confidence, 0.70)

        self.required_fields = _REPORT_REQUIRED
        result = ExtractionResult(
            doc_type=self.doc_type,
            fields=fields, tables=[],
            confidence=confidence, raw_text=text,
            metadata={"sub_type": "medical_report"},
        )
        return self._validate(result)
