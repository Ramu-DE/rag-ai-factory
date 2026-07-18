# -*- coding: utf-8 -*-
"""
BaseComponent — abstract runtime class every component wrapper extends.

Each subclass:
  - declares SPEC = <BaseComponentSpec instance>
  - implements run(ctx: dict) -> dict
  - optionally overrides embed() and retrieve()

The assembler calls run(ctx) and merges the returned dict back into ctx.
"""
from __future__ import annotations
import os, json, hashlib, uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import boto3
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Shared AWS / Qdrant helpers
# ---------------------------------------------------------------------------
AWS_REGION      = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL       = "us.anthropic.claude-sonnet-4-6"
EMBEDDING_DIM   = 1024


def get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )


def get_qdrant_client():
    from qdrant_client import QdrantClient
    url = os.getenv("QDRANT_URL", "")
    key = os.getenv("QDRANT_API_KEY", "")
    if url and key:
        return QdrantClient(url=url, api_key=key, timeout=30)
    return QdrantClient(":memory:")


def embed(text: str) -> List[float]:
    bedrock = get_bedrock_client()
    body = json.dumps({"inputText": text, "dimensions": EMBEDDING_DIM, "normalize": True})
    resp = bedrock.invoke_model(modelId=EMBEDDING_MODEL, body=body)
    return json.loads(resp["body"].read())["embedding"]


def chunk_id(key: str) -> str:
    return str(uuid.UUID(hashlib.sha256(key.encode()).hexdigest()[:32]))


def llm_call(prompt: str, system: str = "", max_tokens: int = 1024,
             temperature: float = 0.1) -> str:
    bedrock = get_bedrock_client()
    messages = [{"role": "user", "content": prompt}]
    body: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system
    resp = bedrock.invoke_model(modelId=LLM_MODEL, body=json.dumps(body))
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"]


def dense_search(qdrant, collection_name: str, query_vec: List[float],
                 k: int = 5, filters=None) -> list:
    from qdrant_client.models import Filter
    kwargs = dict(collection_name=collection_name, query=query_vec,
                  limit=k, with_payload=True)
    if filters:
        kwargs["query_filter"] = filters
    return qdrant.query_points(**kwargs).points


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseComponent(ABC):
    """Every component wrapper inherits from this."""

    @property
    @abstractmethod
    def SPEC(self):
        ...

    @abstractmethod
    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the component.
        ctx  — pipeline context dict (keys match SPEC.input_schema)
        Returns dict whose keys include SPEC.output_schema items.
        """
        ...

    def _require(self, ctx: Dict[str, Any], *keys: str) -> None:
        missing = [k for k in keys if k not in ctx]
        if missing:
            raise ValueError(
                f"{self.SPEC.name}: missing required context keys: {missing}"
            )
