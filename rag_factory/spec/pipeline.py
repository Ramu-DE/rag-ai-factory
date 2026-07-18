# -*- coding: utf-8 -*-
"""
PipelineSpec — YAML/dict -> validated pipeline configuration.
One spec file drives one assembled + guarded pipeline.
"""
from __future__ import annotations
import yaml
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class IngestionConfig(BaseModel):
    doc_path        : str
    collection_name : str
    chunker         : str           = "fixed_chunking"
    use_incremental : bool          = False
    tenant_id       : Optional[str] = None


class RetrievalConfig(BaseModel):
    strategy : str            = "dense_retrieval"
    top_k    : int            = Field(5, ge=1, le=50)
    filters  : Dict[str, Any] = Field(default_factory=dict)
    rerank   : bool           = False
    compress : bool           = False


class QueryConfig(BaseModel):
    strategy       : str = "direct"
    max_sub_queries: int = Field(4, ge=1, le=10)


class GenerationConfig(BaseModel):
    agentic_mode: str   = "none"
    streaming   : bool  = False
    use_cache   : bool  = True
    max_tokens  : int   = 1024
    temperature : float = Field(0.1, ge=0.0, le=1.0)


class GuardConfig(BaseModel):
    retrieval_guard : bool = True
    generation_guard: bool = True
    ambiguity_guard : bool = True
    system_guard    : bool = True


class TemporalConfig(BaseModel):
    enabled          : bool = False
    task_queue       : str  = "rag-factory"
    workflow_id_prefix: str = "rag-wf"


class EvaluationConfig(BaseModel):
    enabled     : bool          = False
    ground_truth: Optional[str] = None


class PipelineSpec(BaseModel):
    name       : str
    version    : str             = "1.0.0"
    description: str             = ""

    ingestion  : IngestionConfig
    retrieval  : RetrievalConfig  = Field(default_factory=RetrievalConfig)
    query      : QueryConfig      = Field(default_factory=QueryConfig)
    generation : GenerationConfig = Field(default_factory=GenerationConfig)
    guards     : GuardConfig      = Field(default_factory=GuardConfig)
    temporal   : TemporalConfig   = Field(default_factory=TemporalConfig)
    evaluation : EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _cross_checks(self) -> "PipelineSpec":
        if self.generation.streaming and self.temporal.enabled:
            raise ValueError(
                "streaming=True is incompatible with temporal.enabled=True — "
                "token streams cannot be durably replayed."
            )
        if (self.ingestion.collection_name.startswith("tenant_")
                and not self.ingestion.tenant_id):
            raise ValueError(
                "Collection names starting with 'tenant_' require ingestion.tenant_id."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineSpec":
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineSpec":
        return cls(**d)

    def active_component_names(self) -> List[str]:
        names = [self.ingestion.chunker, self.retrieval.strategy]
        if self.generation.agentic_mode != "none":
            names.append(self.generation.agentic_mode)
        return names

    def active_guards(self) -> List[str]:
        g = self.guards
        return [n for n, on in [
            ("retrieval_guard",  g.retrieval_guard),
            ("generation_guard", g.generation_guard),
            ("ambiguity_guard",  g.ambiguity_guard),
            ("system_guard",     g.system_guard),
        ] if on]

    def to_yaml(self) -> str:
        return yaml.dump(self.model_dump(), default_flow_style=False, allow_unicode=True)
