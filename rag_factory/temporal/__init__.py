# -*- coding: utf-8 -*-
"""
Temporal workflow + activity definitions for the AI RAG Factory.

Install:  pip install temporalio
Start worker: python -m rag_factory.temporal.worker
"""
from __future__ import annotations
import os
from typing import Any, Dict, List

try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.worker import Worker
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Activity definitions — each wraps one async BaseComponent
# ---------------------------------------------------------------------------

if TEMPORAL_AVAILABLE:
    @activity.defn(name="ingest_document")
    async def ingest_document(params: Dict[str, Any]) -> Dict[str, Any]:
        """Temporal Activity: chunk + embed + upsert a document."""
        from dotenv import load_dotenv
        load_dotenv(override=True)
        from ..components.chunking import ContextualChunking, SemanticChunking, FixedChunking
        chunker_name = params.get("chunker", "fixed_chunking")
        chunker_map  = {
            "fixed_chunking":       FixedChunking,
            "semantic_chunking":    SemanticChunking,
            "contextual_chunking":  ContextualChunking,
        }
        cls = chunker_map.get(chunker_name, FixedChunking)
        return cls().run(params)

    @activity.defn(name="retrieve_chunks")
    async def retrieve_chunks(params: Dict[str, Any]) -> Dict[str, Any]:
        """Temporal Activity: retrieve from Qdrant."""
        from dotenv import load_dotenv
        load_dotenv(override=True)
        from ..components.retrieval import HybridRRFRetrieval, DenseRetrieval
        strategy = params.get("strategy", "dense_retrieval")
        cls = HybridRRFRetrieval if "hybrid" in strategy else DenseRetrieval
        return cls().run(params)

    @activity.defn(name="generate_answer")
    async def generate_answer(params: Dict[str, Any]) -> Dict[str, Any]:
        """Temporal Activity: run agentic generation."""
        from dotenv import load_dotenv
        load_dotenv(override=True)
        from ..components.agentic import (
            SelfRAG, IterativeRAG, RecursiveRAG, CorrectiveRAG
        )
        mode_map = {
            "self_rag":      SelfRAG,
            "iterative_rag": IterativeRAG,
            "recursive_rag": RecursiveRAG,
            "corrective_rag":CorrectiveRAG,
        }
        mode = params.get("agentic_mode", "corrective_rag")
        cls  = mode_map.get(mode, CorrectiveRAG)
        return cls().run(params)

    @activity.defn(name="evaluate_response")
    async def evaluate_response(params: Dict[str, Any]) -> Dict[str, Any]:
        """Temporal Activity: RAGAS evaluation."""
        from dotenv import load_dotenv
        load_dotenv(override=True)
        from ..components.production import EvaluationRAG
        return EvaluationRAG().run(params)

    @activity.defn(name="reindex_stale")
    async def reindex_stale(params: Dict[str, Any]) -> Dict[str, Any]:
        """Temporal Activity: incremental re-index on content-hash mismatch."""
        from dotenv import load_dotenv
        load_dotenv(override=True)
        from ..components.production import IncrementalRAG
        return IncrementalRAG().run(params)

    # -------------------------------------------------------------------------
    # Workflow definitions
    # -------------------------------------------------------------------------
    from datetime import timedelta

    @workflow.defn(name="IngestWorkflow")
    class IngestWorkflow:
        """
        Durable ingestion: extract -> chunk -> embed -> upsert.
        Survives AWS token expiry — retries resume from the failed activity.
        """
        @workflow.run
        async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
            result = await workflow.execute_activity(
                ingest_document,
                params,
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=None,  # uses default: 3 retries, exponential backoff
            )
            return result

    @workflow.defn(name="AgenticQueryWorkflow")
    class AgenticQueryWorkflow:
        """
        Durable agentic query: retrieve -> agentic generation -> evaluate.
        Handles iterative/recursive RAG loops that can exceed 5 minutes.
        """
        @workflow.run
        async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
            # Step 1: Retrieve
            ret = await workflow.execute_activity(
                retrieve_chunks,
                params,
                start_to_close_timeout=timedelta(minutes=2),
            )
            # Step 2: Generate
            gen_params = {**params, **ret}
            gen = await workflow.execute_activity(
                generate_answer,
                gen_params,
                start_to_close_timeout=timedelta(minutes=15),
            )
            # Step 3: Evaluate (if enabled)
            if params.get("evaluate", False) and params.get("ground_truth"):
                eval_params = {**params, **gen}
                scores = await workflow.execute_activity(
                    evaluate_response,
                    eval_params,
                    start_to_close_timeout=timedelta(minutes=5),
                )
                gen.update(scores)
            return gen

    @workflow.defn(name="ScheduledReindexWorkflow")
    class ScheduledReindexWorkflow:
        """
        Cron-style scheduled re-index.
        Run via: temporalio cron schedule every 6 hours.
        """
        @workflow.run
        async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
            return await workflow.execute_activity(
                reindex_stale,
                params,
                start_to_close_timeout=timedelta(minutes=30),
            )

    # -------------------------------------------------------------------------
    # Worker launcher
    # -------------------------------------------------------------------------
    async def run_worker(task_queue: str = "rag-factory"):
        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        client        = await Client.connect(temporal_host)
        worker        = Worker(
            client,
            task_queue=task_queue,
            workflows=[IngestWorkflow, AgenticQueryWorkflow, ScheduledReindexWorkflow],
            activities=[
                ingest_document, retrieve_chunks,
                generate_answer, evaluate_response, reindex_stale,
            ],
        )
        print(f"Temporal worker started on '{task_queue}' -> {temporal_host}")
        await worker.run()

else:
    # Graceful stub when temporalio is not installed
    class IngestWorkflow:          pass   # noqa: E701
    class AgenticQueryWorkflow:    pass   # noqa: E701
    class ScheduledReindexWorkflow:pass   # noqa: E701

    async def run_worker(*args, **kwargs):
        raise ImportError("Install temporalio: pip install temporalio")


__all__ = [
    "TEMPORAL_AVAILABLE",
    "IngestWorkflow", "AgenticQueryWorkflow", "ScheduledReindexWorkflow",
    "run_worker",
]
