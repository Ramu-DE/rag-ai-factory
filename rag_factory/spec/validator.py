# -*- coding: utf-8 -*-
"""SpecValidator — pre-flight checks before the assembler runs."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from .pipeline import PipelineSpec
from .manifest import MANIFEST


@dataclass
class ValidationResult:
    valid   : bool
    errors  : List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Valid: {self.valid}"]
        for e in self.errors  : lines.append(f"  ERROR   : {e}")
        for w in self.warnings: lines.append(f"  WARNING : {w}")
        return "\n".join(lines)


class SpecValidator:
    """Validate a PipelineSpec before execution."""

    _LONG_RUNNING = {"iterative_rag", "recursive_rag", "agentic_rag"}

    def validate(self, spec: PipelineSpec) -> ValidationResult:
        errs, warns = [], []

        # All named components must exist in manifest
        for comp_name in spec.active_component_names():
            try:
                MANIFEST.get(comp_name)
            except KeyError as e:
                errs.append(str(e))

        # Long-running agentic modes should use Temporal
        if spec.generation.agentic_mode in self._LONG_RUNNING:
            if not spec.temporal.enabled:
                warns.append(
                    f"agentic_mode='{spec.generation.agentic_mode}' can run >5 min. "
                    "Consider temporal.enabled=true for crash recovery."
                )

        # Streaming incompatibilities
        if spec.generation.streaming:
            if spec.evaluation.enabled:
                warns.append(
                    "evaluation.enabled=true with streaming=true — "
                    "RAGAS scoring requires the full response."
                )

        # Multi-tenant collection without system_guard
        if "tenant" in spec.ingestion.collection_name.lower() and not spec.guards.system_guard:
            warns.append(
                "Collection name suggests multi-tenant but system_guard=false. "
                "FM-S3 (PII leakage) will not be detected."
            )

        # Contextual chunking without generation guard
        if (spec.ingestion.chunker == "contextual_chunking"
                and not spec.guards.generation_guard):
            warns.append(
                "contextual_chunking without generation_guard=true — "
                "FM-G1 (hallucination) will not be checked."
            )

        # Agentic pipeline without evaluation
        if (spec.generation.agentic_mode != "none"
                and not spec.evaluation.enabled):
            warns.append(
                "Agentic pipeline without evaluation.enabled=true — "
                "RAGAS scores won't be tracked."
            )

        return ValidationResult(valid=len(errs) == 0, errors=errs, warnings=warns)


VALIDATOR = SpecValidator()
