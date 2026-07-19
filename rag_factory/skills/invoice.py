# -*- coding: utf-8 -*-
"""
Invoice / Receipt Skill
=======================
Uses AWS Textract AnalyzeExpense for structured field extraction,
then LLM for any fields Textract missed.

Extracted fields:
  vendor_name, vendor_address, invoice_number, invoice_date, due_date,
  subtotal, tax_amount, total_amount, currency, payment_terms,
  line_items (table)
"""
from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseDocumentSkill, ExtractionResult

_REQUIRED = ["vendor_name", "invoice_date", "total_amount"]

_EXPENSE_FIELD_MAP = {
    "VENDOR_NAME":     "vendor_name",
    "VENDOR_ADDRESS":  "vendor_address",
    "INVOICE_RECEIPT_ID": "invoice_number",
    "INVOICE_RECEIPT_DATE": "invoice_date",
    "DUE_DATE":        "due_date",
    "SUBTOTAL":        "subtotal",
    "TAX":             "tax_amount",
    "TOTAL":           "total_amount",
    "PAYMENT_TERMS":   "payment_terms",
    "PO_NUMBER":       "po_number",
    "RECEIVER_NAME":   "receiver_name",
}

_LLM_PROMPT = """\
Extract invoice fields from the following document text.
Return ONLY key: value pairs, one per line.
Keys to extract: vendor_name, vendor_address, invoice_number, invoice_date,
due_date, subtotal, tax_amount, total_amount, currency, payment_terms, po_number.
If a field is not found write: key: N/A

Document text:
{text}
"""


class InvoiceSkill(BaseDocumentSkill):
    doc_type        = "invoice"
    required_fields = _REQUIRED
    description     = "Extracts vendor, dates, amounts, and line items from invoices and receipts"

    def run(self, doc: Any) -> ExtractionResult:
        fields: Dict[str, str] = {}
        tables: List[Dict]     = []
        confidence             = 0.0

        # ── 1. Use Textract expense fields if available ───────────────────
        if doc.expense_fields:
            conf_sum, conf_cnt = 0.0, 0
            for ef in doc.expense_fields:
                mapped = _EXPENSE_FIELD_MAP.get(ef.field_type)
                if mapped and ef.value:
                    fields[mapped] = ef.value
                    conf_sum += ef.confidence
                    conf_cnt += 1
                elif ef.field_type == "LINE_ITEM":
                    tables.append({"line_item": ef.value})
            confidence = (conf_sum / conf_cnt / 100.0) if conf_cnt else 0.0

        # ── 2. LLM gap-fill for any missing required fields ───────────────
        text = doc.full_rich_text
        missing = [f for f in _REQUIRED if not fields.get(f)]
        if missing or not fields:
            raw = self._llm_extract(text, _LLM_PROMPT)
            llm_fields = self._parse_kv(raw)
            for k, v in llm_fields.items():
                if k not in fields or not fields[k]:
                    fields[k] = v
            if not confidence:
                confidence = 0.70  # LLM-only confidence

        # ── 3. Table rows from Textract tables ────────────────────────────
        for tbl in doc.all_tables:
            if tbl.rows > 1:
                headers = [c.text for c in tbl.cells if c.row == 0]
                for row_idx in range(1, tbl.rows):
                    row_vals = [c.text for c in tbl.cells if c.row == row_idx]
                    if headers and row_vals:
                        tables.append(dict(zip(headers, row_vals)))

        result = ExtractionResult(
            doc_type=self.doc_type,
            fields=fields,
            tables=tables,
            confidence=confidence,
            raw_text=text,
        )
        return self._validate(result)
