# -*- coding: utf-8 -*-
"""
BaseComponentSpec — typed contract every RAG component must satisfy.
Modelled after NVIDIA NIM microservice contracts.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ComponentRole(str, Enum):
    CHUNKER   = "chunker"
    RETRIEVER = "retriever"
    QUERY_OPS = "query_ops"
    GENERATOR = "generator"
    GUARD     = "guard"
    EVALUATOR = "evaluator"
    MEMORY    = "memory"
    ROUTER    = "router"


class BaseComponentSpec(BaseModel):
    name         : str
    version      : str            = "1.0.0"
    role         : ComponentRole
    notebook_ref : str
    tier         : int            = Field(..., ge=1, le=9)
    description  : str

    input_schema : List[str]
    output_schema: List[str]

    max_retries  : int            = Field(3,  ge=0, le=10)
    timeout_secs : int            = Field(60, ge=1, le=3600)
    is_async     : bool           = False
    is_streaming : bool           = False

    guards_applied: List[str]     = Field(default_factory=list)
    failure_modes : List[str]     = Field(default_factory=list)
    config        : Dict[str, Any]= Field(default_factory=dict)

    class Config:
        use_enum_values = True

    def satisfies(self, required_outputs: List[str]) -> bool:
        return all(k in self.output_schema for k in required_outputs)

    def compatible_with(self, nxt: "BaseComponentSpec") -> Tuple[bool, List[str]]:
        missing = [k for k in nxt.input_schema if k not in self.output_schema]
        return len(missing) == 0, missing

    def __repr__(self) -> str:
        return f"<{self.role}:{self.name} v{self.version} tier={self.tier}>"
