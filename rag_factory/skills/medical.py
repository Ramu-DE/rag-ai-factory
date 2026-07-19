# -*- coding: utf-8 -*-
"""
Medical Document Skill
======================
Handles: clinical notes, lab reports, prescriptions, discharge summaries,
insurance claims, medical forms.

Extracted fields:
  patient_name, patient_dob, patient_id, provider_name, provider_npi,
  facility_name, document_date, diagnosis_codes (ICD-10), procedure_codes (CPT),
  medications, chief_complaint, assessment, plan
"""
from __future__ import annotations
from typing import Any
from .base import BaseDocumentSkill, ExtractionResult

_REQUIRED = ["patient_name", "document_date", "provider_name"]

_MEDICAL_PROMPT = """\
You are a medical document extraction assistant.
Extract the following fields from the medical document text.
Return ONLY key: value pairs, one per line.
For list fields (medications, diagnosis_codes, procedure_codes) join values with semicolons.
Fields: patient_name, patient_dob, patient_id, provider_name, provider_npi,
facility_name, document_date, diagnosis_codes, procedure_codes, medications,
chief_complaint, assessment, plan.
If a field is not present write: key: N/A

Document text:
{text}
"""


class MedicalSkill(BaseDocumentSkill):
    doc_type        = "medical"
    required_fields = _REQUIRED
    description     = "Extracts patient, provider, diagnoses, procedures, and medications from medical documents"

    def run(self, doc: Any) -> ExtractionResult:
        text   = doc.full_rich_text
        raw    = self._llm_extract(text, _MEDICAL_PROMPT, max_tokens=700)
        fields = self._parse_kv(raw)

        # Use Textract form fields for any key-value pairs found
        for ff in doc.all_forms:
            k = ff.key.strip().lower().replace(" ", "_")
            if k and ff.value and k not in fields:
                fields[k] = ff.value

        found      = sum(1 for f in _REQUIRED if fields.get(f))
        confidence = 0.60 + (found / len(_REQUIRED)) * 0.30

        result = ExtractionResult(
            doc_type=self.doc_type,
            fields=fields,
            tables=[],
            confidence=confidence,
            raw_text=text,
        )
        return self._validate(result)
