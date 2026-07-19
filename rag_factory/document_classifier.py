# -*- coding: utf-8 -*-
"""
Document Classifier
====================
Auto-detects document type from content.

Two-layer approach (same as QueryRouter):
  Layer 1: keyword heuristics (fast, free)
  Layer 2: LLM classification (only when heuristic is inconclusive)

Supported types:
  invoice     — invoices, receipts, bills
  contract    — NDAs, agreements, SOW, leases
  medical     — clinical notes, lab results, prescriptions, claims
  id_document — passports, driver licences, national IDs
  report      — financial reports, research papers, audit reports
  form        — filled-in forms, applications, surveys
  letter      — correspondence, memos
  other       — catch-all

ClassificationResult:
  doc_type    : str
  confidence  : float
  heuristic   : bool
  reason      : str
  extraction_mode : str    ("expense" | "id" | "forms" | "auto")
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    doc_type:        str
    confidence:      float
    heuristic:       bool
    reason:          str
    extraction_mode: str   # maps to Textract / extractor mode

    def to_dict(self):
        return {
            "doc_type":        self.doc_type,
            "confidence":      round(self.confidence, 3),
            "heuristic":       self.heuristic,
            "reason":          self.reason,
            "extraction_mode": self.extraction_mode,
        }


# ── extraction mode map ───────────────────────────────────────────────────────
_DOC_TYPE_TO_MODE = {
    "invoice":     "expense",
    "contract":    "forms",
    "medical":     "forms",
    "id_document": "id",
    "report":      "auto",
    "form":        "forms",
    "letter":      "auto",
    "other":       "auto",
}

# ── heuristic patterns ────────────────────────────────────────────────────────
_INVOICE_P = re.compile(
    r"\b(invoice|receipt|bill to|remit to|subtotal|total due|amount due|"
    r"invoice\s*#|inv\s*#|purchase order|vendor|net\s*30|net\s*60|tax amount)\b", re.I
)
_CONTRACT_P = re.compile(
    r"\b(agreement|contract|terms and conditions|whereas|hereinafter|"
    r"governing law|indemnif|liability|confidential|nda|non.disclosure|"
    r"effective date|termination|intellectual property|sow|statement of work)\b", re.I
)
_MEDICAL_P = re.compile(
    r"\b(patient|diagnosis|icd.?10|cpt code|prescription|physician|"
    r"provider|clinical|lab result|discharge|chief complaint|"
    r"assessment|plan|rx|medication|dosage|healthcare)\b", re.I
)
_ID_P = re.compile(
    r"\b(passport|driver.?s? licen[cs]e|national id|date of birth|dob|"
    r"expiry date|mrz|issuing authority|citizenship|nationality)\b", re.I
)
_REPORT_P = re.compile(
    r"\b(annual report|quarterly|financial statement|audit|revenue|"
    r"executive summary|findings|recommendations|appendix|table of contents)\b", re.I
)
_FORM_P = re.compile(
    r"\b(please fill|check all that apply|signature|date signed|"
    r"applicant|social security|ssn|form\s+\w+|\[\s*\]|\(\s*\))\b", re.I
)
_LETTER_P = re.compile(
    r"\b(dear\s+\w|sincerely|yours truly|regards|to whom it may concern|"
    r"re:|subject:|memo|memorandum)\b", re.I
)


class DocumentClassifier:
    """Auto-classify a document by content and recommend extraction mode."""

    def classify(self, text: str, filename: str = "") -> ClassificationResult:
        """
        Classify document from content + optional filename hint.
        Filename is checked first for strong signals (invoice, contract, ID).
        Content heuristic requires >= 4 keyword matches to avoid false positives
        on general documents that incidentally mention medical/legal terms.
        """
        # Filename strong signal (filename beats content for clear cases)
        if filename:
            fn_result = self.classify_filename(filename)
            if fn_result.confidence >= 0.80:
                return fn_result

        result = self._heuristic(text)
        if result:
            return result
        return self._llm_classify(text)

    def classify_filename(self, filename: str) -> ClassificationResult:
        """Quick filename-based pre-classification (supplement with content)."""
        fname = filename.lower()
        if any(x in fname for x in ("invoice","receipt","bill","inv_","inv-")):
            return ClassificationResult("invoice", 0.80, True,
                "Filename suggests invoice.", "expense")
        if any(x in fname for x in ("contract","agreement","nda","sow","lease")):
            return ClassificationResult("contract", 0.80, True,
                "Filename suggests contract.", "forms")
        if any(x in fname for x in ("passport","license","licence","id_card","national_id")):
            return ClassificationResult("id_document", 0.80, True,
                "Filename suggests ID document.", "id")
        if any(x in fname for x in ("medical","clinical","lab","prescription","discharge")):
            return ClassificationResult("medical", 0.78, True,
                "Filename suggests medical document.", "forms")
        return ClassificationResult("other", 0.40, True,
            "Filename inconclusive.", "auto")

    def _heuristic(self, text: str) -> ClassificationResult | None:
        # Use more text for better signal; general documents need more context
        t      = text[:5000]
        scores = {
            "invoice":     len(_INVOICE_P.findall(t)),
            "contract":    len(_CONTRACT_P.findall(t)),
            "medical":     len(_MEDICAL_P.findall(t)),
            "id_document": len(_ID_P.findall(t)),
            "report":      len(_REPORT_P.findall(t)),
            "form":        len(_FORM_P.findall(t)),
            "letter":      len(_LETTER_P.findall(t)),
        }
        best_type  = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        # Minimum 4 keyword matches required to avoid false positives
        # (general docs incidentally contain medical/legal words)
        if best_score < 4:
            return None   # too weak → LLM

        # Require a clear winner: best must be >= 2x second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1] > 0:
            ratio = best_score / sorted_scores[1]
            if ratio < 2.0:
                return None   # ambiguous → LLM

        confidence = min(0.92, 0.65 + best_score * 0.03)
        mode       = _DOC_TYPE_TO_MODE[best_type]
        return ClassificationResult(
            doc_type=best_type,
            confidence=confidence,
            heuristic=True,
            reason=f"Keyword signal: {best_score} {best_type} terms detected (threshold>=4, ratio>=2x).",
            extraction_mode=mode,
        )

    def _llm_classify(self, text: str) -> ClassificationResult:
        try:
            from rag_factory.components.base import llm_call
            raw = llm_call(
                f"Classify this document.\n\n"
                f"Document text (first 2000 chars):\n{text[:2000]}\n\n"
                "Choose the best document type:\n"
                "  invoice     — invoice, receipt, bill\n"
                "  contract    — NDA, agreement, SoW, lease\n"
                "  medical     — clinical notes, lab results, prescription, claim\n"
                "  id_document — passport, driver licence, national ID\n"
                "  report      — financial/research/audit report\n"
                "  form        — application, survey, filled form\n"
                "  letter      — correspondence, memo\n"
                "  other       — none of the above\n\n"
                "Reply with exactly:\n"
                "DOC_TYPE: <type>\n"
                "CONFIDENCE: <0.0-1.0>\n"
                "REASON: <one sentence>",
                max_tokens=80,
                temperature=0.0,
            )
            doc_type   = "other"
            confidence = 0.60
            reason     = "LLM classified."
            for line in raw.strip().splitlines():
                if line.startswith("DOC_TYPE:"):
                    doc_type = line.split(":",1)[1].strip().lower()
                elif line.startswith("CONFIDENCE:"):
                    try:    confidence = float(line.split(":",1)[1].strip())
                    except: pass
                elif line.startswith("REASON:"):
                    reason = line.split(":",1)[1].strip()
            mode = _DOC_TYPE_TO_MODE.get(doc_type, "auto")
            return ClassificationResult(doc_type, confidence, False, reason, mode)
        except Exception as e:
            return ClassificationResult("other", 0.40, False,
                f"Classifier error: {e}", "auto")


CLASSIFIER = DocumentClassifier()
