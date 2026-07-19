# -*- coding: utf-8 -*-
"""
Custom Document Skill
=====================
YAML-defined field extractor — no code required.

Define a skill in YAML:

  name: purchase_order
  doc_type: purchase_order
  description: "Extracts PO fields"
  required_fields: [po_number, vendor, total]
  fields:
    - name: po_number
      description: "Purchase Order number"
      example: "PO-2024-001"
    - name: vendor
      description: "Vendor or supplier name"
    - name: order_date
      description: "Date the PO was issued"
    - name: total
      description: "Total order amount including tax"
    - name: delivery_date
      description: "Expected delivery date"

Load and run:
  from rag_factory.skills.custom import CustomSkill
  skill  = CustomSkill.from_yaml("skills/purchase_order.yaml")
  result = skill.run(extracted_doc)
"""
from __future__ import annotations
import yaml
from typing import Any, Dict, List
from .base import BaseDocumentSkill, ExtractionResult


class CustomSkill(BaseDocumentSkill):
    description = "YAML-defined custom field extractor"

    def __init__(self, config: Dict[str, Any]):
        self.doc_type        = config.get("doc_type", "custom")
        self.required_fields = config.get("required_fields", [])
        self.description     = config.get("description", "Custom skill")
        self._field_defs     = config.get("fields", [])
        self._skill_name     = config.get("name", "custom")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CustomSkill":
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CustomSkill":
        return cls(config)

    def _build_prompt(self, text: str) -> str:
        field_lines = []
        for fd in self._field_defs:
            name = fd.get("name","")
            desc = fd.get("description","")
            ex   = fd.get("example","")
            line = f"  {name}: {desc}"
            if ex:
                line += f" (example: {ex})"
            field_lines.append(line)

        fields_block = "\n".join(field_lines)
        return (
            f"Extract the following fields from the document text below.\n"
            f"Return ONLY key: value pairs, one per line.\n"
            f"If a field is not found write: key: N/A\n\n"
            f"Fields to extract:\n{fields_block}\n\n"
            f"Document text:\n{{text}}"
        )

    def run(self, doc: Any) -> ExtractionResult:
        text   = doc.full_rich_text
        prompt = self._build_prompt(text)
        raw    = self._llm_extract(text, prompt, max_tokens=600)
        fields = self._parse_kv(raw)

        # Supplement with Textract form fields
        for ff in doc.all_forms:
            k = ff.key.strip().lower().replace(" ", "_")
            if k and ff.value and k not in fields:
                fields[k] = ff.value

        found      = sum(1 for f in self.required_fields if fields.get(f))
        confidence = 0.60 + (found / max(len(self.required_fields), 1)) * 0.30

        result = ExtractionResult(
            doc_type=self.doc_type,
            fields=fields,
            confidence=confidence,
            raw_text=text,
        )
        return self._validate(result)
