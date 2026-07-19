# -*- coding: utf-8 -*-
"""
AI RAG Factory — Query Router
==============================
Analyses a user query and selects the optimal pipeline spec.

Two-layer decision:
  Layer 1 — fast regex/heuristic (no LLM cost):
    • negation words     → simple.yaml + AmbiguityGuard override
    • comparison intent  → production.yaml
    • listing / enumerate → production.yaml
    • temporal keywords  → production.yaml

  Layer 2 — LLM classifier (only when heuristic is inconclusive):
    Returns SIMPLE / COMPLEX / AGENTIC / MULTITENANT
    → maps to simple / production / agentic / multitenant yaml

RoutingResult:
    spec        : chosen YAML filename  (e.g. "production.yaml")
    category    : SIMPLE | COMPLEX | AGENTIC | MULTITENANT
    confidence  : 0.0 – 1.0
    reason      : one sentence explanation shown to user
    heuristic   : True if decided without LLM call (fast path)

Usage:
    from rag_factory.router import QueryRouter
    router = QueryRouter(specs_dir="specs")
    result = router.route("Compare Medicare and Medicaid eligibility")
    print(result.spec, result.reason)
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


# ─── result ───────────────────────────────────────────────────────────────────
@dataclass
class RoutingResult:
    spec:       str
    category:   str
    confidence: float
    reason:     str
    heuristic:  bool = False

    def to_dict(self):
        return {
            "spec":       self.spec,
            "category":   self.category,
            "confidence": self.confidence,
            "reason":     self.reason,
            "heuristic":  self.heuristic,
        }


# ─── heuristic patterns ───────────────────────────────────────────────────────
_COMPARISON = re.compile(
    r"\b(compare|comparison|difference|vs\.?|versus|contrast|distinguish|"
    r"which is better|pros and cons|advantages|disadvantages)\b", re.I
)
_LISTING = re.compile(
    r"\b(list all|enumerate|give me all|what are all|how many)\b", re.I
)
_SUMMARY = re.compile(
    r"\b(summarize|summarise|summarization|give me a summary|provide a summary|"
    r"overview|key points|key takeaways|tl;?dr|what is this (about|document)|"
    r"main (topic|finding|idea|point|theme))\b", re.I
)
_TEMPORAL = re.compile(
    r"\b(latest|recent|new|updated|changed|since|after \d{4}|before \d{4}|"
    r"history of|evolution|trend)\b", re.I
)
_NEGATION = re.compile(
    r"\b(not|never|no\b|without|except|exclude|excluding|"
    r"doesn't|don't|isn't|aren't|won't|cannot|can't)\b", re.I
)
_MULTI_INTENT = re.compile(r"\band\b.{0,60}\band\b", re.I)  # "X and Y and Z"
_STEP_BY_STEP = re.compile(
    r"\b(how to|step.?by.?step|explain how|walk me through|process of|procedure)\b", re.I
)
_AGENTIC = re.compile(
    r"\b(research|find all|investigate|gather|collect all|comprehensive|"
    r"thorough|deep.?dive|multi.?step|iterative|recursively)\b", re.I
)

# ─── spec map ─────────────────────────────────────────────────────────────────
_CATEGORY_TO_SPEC = {
    "SIMPLE":      "simple.yaml",
    "COMPLEX":     "production.yaml",
    "AGENTIC":     "agentic.yaml",
    "MULTITENANT": "multitenant.yaml",
}

# Routing decision table (used in Streamlit to display explanation)
ROUTING_LOGIC = [
    ("Summarize / overview / key points", "simple.yaml",      "Single-pass global retrieval sufficient"),
    ("Comparison / contrast",             "production.yaml",  "Needs query decomposition + hybrid RRF"),
    ("Listing / enumerate",               "production.yaml",  "Needs hybrid retrieval + self-RAG verification"),
    ("Temporal / trend",                  "production.yaml",  "May need incremental index awareness"),
    ("Step-by-step / how-to",             "production.yaml",  "Needs chain-of-thought generation"),
    ("Research / deep-dive",              "agentic.yaml",     "Multi-step ReAct loop, iterative gap detection"),
    ("Negation",                          "simple.yaml",      "AmbiguityGuard rewrites before retrieval"),
    ("Simple / factual",                  "simple.yaml",      "Single-hop dense retrieval sufficient"),
]


# ─── router ───────────────────────────────────────────────────────────────────
class QueryRouter:
    """
    Routes a natural-language query to the most appropriate pipeline spec.
    Falls back to LLM classification when heuristics are inconclusive.
    """

    def __init__(self, specs_dir: str = "specs"):
        self.specs_dir   = specs_dir
        self._available  = self._discover_specs()

    def _discover_specs(self):
        import os
        if not os.path.isdir(self.specs_dir):
            return set(_CATEGORY_TO_SPEC.values())
        return {f for f in os.listdir(self.specs_dir) if f.endswith(".yaml")}

    def _pick(self, category: str) -> str:
        spec = _CATEGORY_TO_SPEC.get(category, "simple.yaml")
        # Graceful fallback if spec file not present
        if spec not in self._available and "simple.yaml" in self._available:
            return "simple.yaml"
        return spec

    # ── public ────────────────────────────────────────────────────────────────
    def route(self, query: str, tenant_id: Optional[str] = None) -> RoutingResult:
        """Return a RoutingResult for the given query."""

        # Tenant context forces multitenant spec
        if tenant_id:
            return RoutingResult(
                spec       = self._pick("MULTITENANT"),
                category   = "MULTITENANT",
                confidence = 1.0,
                reason     = "Tenant context detected — tenant isolation required.",
                heuristic  = True,
            )

        # Layer 1: heuristic fast path
        result = self._heuristic(query)
        if result:
            return result

        # Layer 2: LLM classifier
        return self._llm_classify(query)

    def _heuristic(self, query: str) -> Optional[RoutingResult]:
        q = query.strip()

        if _SUMMARY.search(q):
            return RoutingResult(
                spec="simple.yaml", category="SIMPLE", confidence=0.88, heuristic=True,
                reason="Summary/overview intent — single-pass global retrieval selected.",
            )
        if _AGENTIC.search(q):
            return RoutingResult(
                spec="agentic.yaml", category="AGENTIC", confidence=0.88, heuristic=True,
                reason="Query signals multi-step research — ReAct agentic loop selected.",
            )
        if _COMPARISON.search(q):
            return RoutingResult(
                spec="production.yaml", category="COMPLEX", confidence=0.90, heuristic=True,
                reason="Comparison intent — query decomposition + hybrid RRF selected.",
            )
        if _LISTING.search(q):
            return RoutingResult(
                spec="production.yaml", category="COMPLEX", confidence=0.85, heuristic=True,
                reason="Enumeration intent — hybrid retrieval + self-RAG verification selected.",
            )
        if _STEP_BY_STEP.search(q):
            return RoutingResult(
                spec="production.yaml", category="COMPLEX", confidence=0.82, heuristic=True,
                reason="How-to / step-by-step — chain-of-thought generation selected.",
            )
        if _TEMPORAL.search(q):
            return RoutingResult(
                spec="production.yaml", category="COMPLEX", confidence=0.80, heuristic=True,
                reason="Temporal signal — incremental-aware hybrid retrieval selected.",
            )
        if _NEGATION.search(q) and len(q.split()) <= 20:
            # Short negation query — simple + ambiguity guard handles it
            return RoutingResult(
                spec="simple.yaml", category="SIMPLE", confidence=0.78, heuristic=True,
                reason="Negation detected — AmbiguityGuard will rewrite before retrieval.",
            )
        return None  # inconclusive — fall through to LLM

    def _llm_classify(self, query: str) -> RoutingResult:
        try:
            from .components.base import llm_call
            raw = llm_call(
                f"Classify this RAG query for pipeline routing.\n\n"
                f"Query: {query}\n\n"
                "Choose the best category:\n"
                "  SIMPLE     — factual, single-hop, short factual answer\n"
                "  COMPLEX    — multi-aspect, comparison, needs reasoning or CoT\n"
                "  AGENTIC    — open-ended research, needs iterative retrieval\n\n"
                "Reply with exactly:\n"
                "CATEGORY: <SIMPLE|COMPLEX|AGENTIC>\n"
                "CONFIDENCE: <0.0-1.0>\n"
                "REASON: <one sentence>",
                max_tokens=80,
                temperature=0.0,
            )
            category   = "SIMPLE"
            confidence = 0.70
            reason     = "LLM classified as simple factual query."
            for line in raw.strip().splitlines():
                if line.startswith("CATEGORY:"):
                    category = line.split(":", 1)[1].strip().upper()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
            return RoutingResult(
                spec       = self._pick(category),
                category   = category,
                confidence = confidence,
                reason     = reason,
                heuristic  = False,
            )
        except Exception as exc:
            # Hard fallback — never crash on routing
            return RoutingResult(
                spec="simple.yaml", category="SIMPLE", confidence=0.5,
                reason=f"Router fallback (LLM error: {exc})",
                heuristic=False,
            )

    def explain(self) -> str:
        """Return a human-readable routing decision table."""
        lines = [f"{'Signal':<35} {'Spec':<22} {'Why'}"]
        lines.append("-" * 80)
        for signal, spec, why in ROUTING_LOGIC:
            lines.append(f"{signal:<35} {spec:<22} {why}")
        return "\n".join(lines)
