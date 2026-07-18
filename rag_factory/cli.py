# -*- coding: utf-8 -*-
"""
AI RAG Factory CLI
Usage:
  python -m rag_factory.cli run    --spec specs/simple.yaml --query "What is Medicaid?"
  python -m rag_factory.cli ingest --spec specs/simple.yaml --text "path/to/doc.txt"
  python -m rag_factory.cli specs
  python -m rag_factory.cli serve  [--host 0.0.0.0] [--port 8000]
"""
from __future__ import annotations
import argparse, os, sys, time, json

def _load_env():
    from dotenv import load_dotenv
    # Try repo root .env
    for candidate in [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
        os.path.join(os.path.expanduser("~"), "RAG", ".env"),
    ]:
        if os.path.isfile(candidate):
            load_dotenv(candidate, override=True)
            return


def cmd_run(args):
    _load_env()
    from rag_factory.spec      import PipelineSpec, VALIDATOR
    from rag_factory.assembler import Assembler
    from rag_factory.components.base import embed, dense_search, llm_call, get_qdrant_client
    from rag_factory.guards import RetrievalGuard, GenerationGuard, AmbiguityGuard, SystemGuard

    spec_path = args.spec
    if not os.path.isfile(spec_path):
        spec_path = os.path.join("specs", args.spec)
    spec   = PipelineSpec.from_yaml(spec_path)
    result = VALIDATOR.validate(spec)
    if not result.valid:
        print("ERROR: invalid spec\n" + str(result)); sys.exit(1)
    if result.warnings:
        for w in result.warnings: print(f"WARN: {w}")

    print(f"\nPipeline : {spec.name}")
    print(f"Chunker  : {spec.ingestion.chunker}")
    print(f"Retrieval: {spec.retrieval.strategy}")
    print(f"Agentic  : {spec.generation.agentic_mode}")
    print(f"Guards   : {spec.active_guards()}")
    print(f"Query    : {args.query}\n")

    t0     = time.time()
    qdrant = get_qdrant_client()

    # Ambiguity guard
    a_res  = AmbiguityGuard().run({"query": args.query, "retrieved_chunks": [], "tenant_id": "cli"})
    query  = a_res.get("query", args.query)
    if query != args.query:
        print(f"[ambiguity guard] rewritten: {query}")

    # Retrieve
    vec    = embed(query)
    points = dense_search(qdrant, spec.ingestion.collection_name, vec, spec.retrieval.top_k)
    chunks = [p.payload for p in points]

    # System guard
    s_res  = SystemGuard().run({"query": query, "retrieved_chunks": chunks, "tenant_id": "cli"})
    chunks = s_res.get("retrieved_chunks", chunks)
    if s_res.get("guard_log",{}).get("system"):
        for log in s_res["guard_log"]["system"]:
            print(f"[system guard] {log}")

    # Generate
    context = "\n\n".join(c.get("text","") for c in chunks)
    answer  = llm_call(
        f"Answer using only the provided context.\n\nContext:\n{context}\n\nQuestion: {args.query}",
        max_tokens=1024,
    )

    # Generation guard
    g_res  = GenerationGuard().run({"query": args.query, "answer": answer,
                                    "retrieved_chunks": chunks, "tenant_id": "cli"})
    faith  = g_res.get("faithfulness_score","n/a")
    elapsed= int((time.time() - t0) * 1000)

    print("=" * 72)
    print(answer)
    print("=" * 72)
    print(f"\nChunks retrieved : {len(chunks)}")
    print(f"Faithfulness     : {faith}")
    print(f"Elapsed          : {elapsed}ms")


def cmd_ingest(args):
    _load_env()
    from rag_factory.spec      import PipelineSpec
    from rag_factory.components.base import embed, get_qdrant_client, chunk_id
    from qdrant_client.models import VectorParams, Distance, PointStruct
    import re

    spec_path = args.spec
    if not os.path.isfile(spec_path):
        spec_path = os.path.join("specs", args.spec)
    spec  = PipelineSpec.from_yaml(spec_path)
    cname = spec.ingestion.collection_name

    # Read text
    text = args.text
    if os.path.isfile(text):
        with open(text, encoding="utf-8") as f:
            text = f.read()

    # Fixed chunking
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+500])
        start += 450

    qdrant = get_qdrant_client()
    existing = [c.name for c in qdrant.get_collections().collections]
    if cname not in existing:
        qdrant.create_collection(cname, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

    points = []
    for i, ch in enumerate(chunks):
        vec = embed(ch)
        cid = chunk_id(f"{cname}:cli:{i}:{ch[:60]}")
        points.append(PointStruct(id=cid, vector=vec, payload={"text": ch, "chunk_index": i}))
    qdrant.upsert(collection_name=cname, points=points)

    print(f"Ingested {len(points)} chunks into '{cname}'")


def cmd_specs(_args):
    _load_env()
    from rag_factory.spec import PipelineSpec, VALIDATOR
    specs_dir = "specs"
    if not os.path.isdir(specs_dir):
        print("No specs/ directory found"); return
    print(f"{'File':<24} {'Name':<28} {'Chunker':<26} {'Valid'}")
    print("-" * 90)
    for fname in sorted(os.listdir(specs_dir)):
        if fname.endswith(".yaml"):
            try:
                spec = PipelineSpec.from_yaml(os.path.join(specs_dir, fname))
                vr   = VALIDATOR.validate(spec)
                print(f"{fname:<24} {spec.name:<28} {spec.ingestion.chunker:<26} {'PASS' if vr.valid else 'FAIL'}")
            except Exception as e:
                print(f"{fname:<24} ERROR: {e}")


def cmd_serve(args):
    import uvicorn
    from rag_factory.api.main import app
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


def main():
    parser = argparse.ArgumentParser(prog="factory", description="AI RAG Factory CLI")
    sub    = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a query through a pipeline spec")
    p_run.add_argument("--spec",  required=True, help="YAML spec file (e.g. specs/simple.yaml)")
    p_run.add_argument("--query", required=True, help="Query string")
    p_run.set_defaults(func=cmd_run)

    p_ing = sub.add_parser("ingest", help="Ingest a document into a collection")
    p_ing.add_argument("--spec", required=True)
    p_ing.add_argument("--text", required=True, help="Raw text or path to .txt file")
    p_ing.set_defaults(func=cmd_ingest)

    p_spc = sub.add_parser("specs", help="List available pipeline specs")
    p_spc.set_defaults(func=cmd_specs)

    p_srv = sub.add_parser("serve", help="Start the FastAPI server")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
