# -*- coding: utf-8 -*-
"""
AI RAG Factory — Temporal Worker
================================
Runs the Temporal task queue "rag-factory" for all ingestion, agentic,
and scheduled-reindex workflows.

Quick start:
    # 1. Start Temporal dev server (first time only):
    #    pip install temporalio
    #    temporal server start-dev
    #    # or with Docker: docker run -p 7233:7233 temporalio/auto-setup:latest

    # 2. Start this worker:
    #    cd C:/Users/Administrator/rag-ai-factory
    #    python -m rag_factory.temporal.worker

    # 3. Trigger a workflow (from Python):
    #    from rag_factory.temporal import IngestWorkflow, run_worker
    #    asyncio.run(trigger_ingest("my_collection", "doc_001", "Some text..."))

Environment variables:
    TEMPORAL_HOST       Temporal server address (default: localhost:7233)
    TEMPORAL_NAMESPACE  Temporal namespace      (default: default)
"""
from __future__ import annotations
import asyncio, logging, os, sys

# ─── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
for candidate in [
    os.path.join(os.getcwd(), ".env"),
    os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
    os.path.join(os.path.expanduser("~"), "RAG", ".env"),
]:
    if os.path.isfile(candidate):
        load_dotenv(candidate, override=True)
        break

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag-factory.worker")

# ─── Temporal guard ───────────────────────────────────────────────────────────
try:
    from temporalio.client import Client
    from temporalio.worker import Worker
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False

if not TEMPORAL_AVAILABLE:
    print(
        "\nERROR: temporalio is not installed.\n"
        "       Run:  pip install temporalio\n"
        "       Then re-start this worker.\n"
    )
    sys.exit(1)

# ─── Import activities and workflows from the factory ─────────────────────────
from rag_factory.temporal import (
    ingest_document,
    retrieve_chunks,
    generate_answer,
    evaluate_response,
    reindex_stale,
    IngestWorkflow,
    AgenticQueryWorkflow,
    ScheduledReindexWorkflow,
)

TASK_QUEUE = "rag-factory"
HOST       = os.getenv("TEMPORAL_HOST",      "localhost:7233")
NAMESPACE  = os.getenv("TEMPORAL_NAMESPACE", "default")


async def run():
    log.info("Connecting to Temporal at %s (namespace=%s) ...", HOST, NAMESPACE)
    client = await Client.connect(HOST, namespace=NAMESPACE)
    log.info("Connected.")

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[IngestWorkflow, AgenticQueryWorkflow, ScheduledReindexWorkflow],
        activities=[ingest_document, retrieve_chunks, generate_answer,
                    evaluate_response, reindex_stale],
    )
    log.info("Worker started on task queue '%s'. Press Ctrl+C to stop.", TASK_QUEUE)
    log.info("Open http://localhost:8233 for the Temporal Web UI.")
    await worker.run()


# ─── Helper: trigger workflows from code ──────────────────────────────────────
async def trigger_ingest(
    collection: str,
    doc_id:     str,
    text:       str,
    chunker:    str = "fixed_chunking",
    tenant_id:  str | None = None,
) -> str:
    """Submit an IngestWorkflow and await the result."""
    client = await Client.connect(HOST, namespace=NAMESPACE)
    result = await client.execute_workflow(
        IngestWorkflow.run,
        args=[collection, doc_id, text, chunker, tenant_id],
        id=f"ingest-{doc_id}",
        task_queue=TASK_QUEUE,
    )
    return result


async def trigger_query(
    collection: str,
    query:      str,
    spec:       str = "simple.yaml",
    tenant_id:  str | None = None,
) -> dict:
    """Submit an AgenticQueryWorkflow and await the result."""
    client = await Client.connect(HOST, namespace=NAMESPACE)
    result = await client.execute_workflow(
        AgenticQueryWorkflow.run,
        args=[collection, query, spec, tenant_id],
        id=f"query-{hash(query) & 0xFFFF}",
        task_queue=TASK_QUEUE,
    )
    return result


async def trigger_reindex(collection: str) -> dict:
    """Submit a ScheduledReindexWorkflow and await the result."""
    client = await Client.connect(HOST, namespace=NAMESPACE)
    result = await client.execute_workflow(
        ScheduledReindexWorkflow.run,
        args=[collection],
        id=f"reindex-{collection}",
        task_queue=TASK_QUEUE,
    )
    return result


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Worker stopped.")
