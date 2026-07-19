# -*- coding: utf-8 -*-
"""
Field Validator
===============
Rule-based validation of extracted document fields.

Rules defined per doc_type:
  required      : field must be present and non-empty
  regex         : field value must match pattern
  numeric_range : numeric value must be within [min, max]
  cross_field   : if field_a present, field_b must also be present

ValidationReport:
  valid   : bool
  errors  : List[str]   — hard failures (required missing, format wrong)
  warnings: List[str]   — soft alerts (low confidence, near-expiry)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationReport:
    valid:    bool
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


# ── built-in rule sets ────────────────────────────────────────────────────────
_DATE_RE    = re.compile(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2}|[A-Za-z]+ \d{1,2},?\s*\d{4}")
_AMOUNT_RE  = re.compile(r"[\$£€]?[\d,]+\.?\d*")
_EMAIL_RE   = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_ICD_RE     = re.compile(r"[A-Z]\d{2}\.?\d*")
_CPT_RE     = re.compile(r"\d{5}")

_RULES: Dict[str, List[Dict]] = {
    "invoice": [
        {"type": "required",      "field": "vendor_name"},
        {"type": "required",      "field": "total_amount"},
        {"type": "required",      "field": "invoice_date"},
        {"type": "regex",         "field": "invoice_date",   "pattern": _DATE_RE,   "message": "invoice_date format unrecognised"},
        {"type": "regex",         "field": "total_amount",   "pattern": _AMOUNT_RE, "message": "total_amount does not look like a monetary value", "severity": "warning"},
        {"type": "cross_field",   "field": "due_date",       "requires": "invoice_date", "message": "due_date present but invoice_date missing"},
    ],
    "contract": [
        {"type": "required",      "field": "party_1"},
        {"type": "required",      "field": "party_2"},
        {"type": "required",      "field": "effective_date"},
        {"type": "regex",         "field": "effective_date", "pattern": _DATE_RE,  "message": "effective_date format unrecognised"},
        {"type": "cross_field",   "field": "expiry_date",    "requires": "effective_date", "message": "expiry_date present but effective_date missing"},
    ],
    "medical": [
        {"type": "required",      "field": "patient_name"},
        {"type": "required",      "field": "provider_name"},
        {"type": "required",      "field": "document_date"},
        {"type": "regex",         "field": "document_date",  "pattern": _DATE_RE,  "message": "document_date format unrecognised"},
        {"type": "regex",         "field": "diagnosis_codes","pattern": _ICD_RE,   "message": "diagnosis_codes do not contain valid ICD-10 codes"},
    ],
    "id_document": [
        {"type": "required",      "field": "id_number"},
        {"type": "required",      "field": "last_name"},
        {"type": "required",      "field": "date_of_birth"},
        {"type": "regex",         "field": "date_of_birth",  "pattern": _DATE_RE,  "message": "date_of_birth format unrecognised"},
        {"type": "regex",         "field": "expiry_date",    "pattern": _DATE_RE,  "message": "expiry_date format unrecognised"},
    ],
}


class FieldValidator:
    """Validates ExtractionResult fields against doc_type rule set."""

    def validate(
        self,
        fields:     Dict[str, str],
        doc_type:   str,
        confidence: float = 1.0,
        extra_rules: Optional[List[Dict]] = None,
    ) -> ValidationReport:
        rules  = list(_RULES.get(doc_type, []))
        if extra_rules:
            rules.extend(extra_rules)

        errors:   List[str] = []
        warnings: List[str] = []

        for rule in rules:
            rtype = rule["type"]
            fname = rule["field"]
            val   = fields.get(fname, "").strip()

            if rtype == "required":
                if not val or val.lower() in ("n/a","none","unknown",""):
                    errors.append(f"Required field missing: {fname}")

            elif rtype == "regex":
                if val and val.lower() not in ("n/a","none","unknown"):
                    pattern = rule["pattern"]
                    if not pattern.search(val):
                        msg = rule.get("message", f"{fname}: value '{val}' did not match expected format")
                        if rule.get("severity", "error") == "warning":
                            warnings.append(msg)
                        else:
                            errors.append(msg)

            elif rtype == "numeric_range":
                if val:
                    try:
                        num = float(re.sub(r"[^\d\.]", "", val))
                        mn  = rule.get("min")
                        mx  = rule.get("max")
                        if mn is not None and num < mn:
                            warnings.append(f"{fname}: {num} below minimum {mn}")
                        if mx is not None and num > mx:
                            warnings.append(f"{fname}: {num} exceeds maximum {mx}")
                    except ValueError:
                        warnings.append(f"{fname}: could not parse as number")

            elif rtype == "cross_field":
                if val and val.lower() not in ("n/a","none","unknown"):
                    req = rule["requires"]
                    if not fields.get(req,"").strip():
                        warnings.append(rule.get("message", f"{fname} present but {req} missing"))

        # Confidence warning
        if confidence < 0.60:
            warnings.append(f"Low extraction confidence ({confidence:.0%}) — manual review recommended")

        return ValidationReport(valid=len(errors) == 0, errors=errors, warnings=warnings)


VALIDATOR = FieldValidator()
