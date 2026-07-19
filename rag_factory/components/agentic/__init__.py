# -*- coding: utf-8 -*-
"""Tier 4 — Agentic RAG component wrappers."""
from __future__ import annotations
from typing import Any, Dict, List
from ..base import BaseComponent, embed, dense_search, llm_call, get_qdrant_client
from ...spec.manifest import MANIFEST


# ---------------------------------------------------------------------------
# Corrective RAG
# ---------------------------------------------------------------------------
class CorrectiveRAG(BaseComponent):
    SPEC = MANIFEST.get("corrective_rag")

    def __init__(self, relevance_threshold: float = 0.5):
        self.threshold = relevance_threshold

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "retrieved_chunks")
        query  = ctx["query"]
        chunks = ctx["retrieved_chunks"]
        log    = []

        # Score each chunk for relevance
        import re as _re
        good, bad = [], []
        for ch in chunks:
            score_raw = llm_call(
                f"Query: {query}\nPassage: {ch.get('text','')[:300]}\n\n"
                "Reply with a single decimal number 0.0-1.0 for relevance. No explanation.",
                max_tokens=20, temperature=0.0,
            )
            nums = _re.findall(r"[01](?:\.\d+)?|\.\d+", score_raw.strip())
            try:
                score = min(1.0, max(0.0, float(nums[0]))) if nums else 0.0
            except (ValueError, IndexError):
                score = 0.0
            (good if score >= self.threshold else bad).append((score, ch))
            log.append({"text_preview": ch.get("text","")[:60], "score": score})

        if not good:
            # All chunks failed — fallback: use raw query decomposition
            fallback = llm_call(
                f"Answer the following question using your knowledge only "
                f"(no retrieved context was relevant):\n\n{query}",
                max_tokens=512,
            )
            return {
                "answer": fallback,
                "correction_log": log + [{"fallback": True}],
            }

        context = "\n\n".join(c.get("text","") for _, c in good)
        answer  = llm_call(
            f"Answer using only the provided context.\n\nContext:\n{context}"
            f"\n\nQuestion: {query}",
            max_tokens=1024,
        )
        return {"answer": answer, "correction_log": log}


# ---------------------------------------------------------------------------
# Self RAG
# ---------------------------------------------------------------------------
class SelfRAG(BaseComponent):
    SPEC = MANIFEST.get("self_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "retrieved_chunks")
        query  = ctx["query"]
        chunks = ctx["retrieved_chunks"]

        reflection_tokens: List[Dict] = []
        citation_scores:   List[float] = []
        used_chunks:       List[Dict]  = []

        for ch in chunks:
            text = ch.get("text", "")

            # IsRel — is chunk relevant?
            is_rel = llm_call(
                f"Query: {query}\nPassage: {text[:300]}\n\n"
                "Is this passage relevant? Answer YES or NO only.",
                max_tokens=5,
            ).strip().upper().startswith("Y")

            if not is_rel:
                reflection_tokens.append({"chunk": text[:60], "IsRel": False})
                continue

            # IsSup — does answer follow from chunk?
            answer_draft = llm_call(
                f"Using this passage, answer: {query}\n\nPassage: {text}",
                max_tokens=200,
            )
            is_sup = llm_call(
                f"Passage: {text}\nAnswer: {answer_draft}\n\n"
                "Is the answer fully supported by the passage? YES or NO only.",
                max_tokens=5,
            ).strip().upper().startswith("Y")

            sup_score = 1.0 if is_sup else 0.0
            citation_scores.append(sup_score)
            reflection_tokens.append({
                "chunk": text[:60], "IsRel": True,
                "IsSup": is_sup, "support_score": sup_score,
            })
            if is_sup:
                used_chunks.append(ch)

        context = "\n\n".join(c.get("text","") for c in used_chunks) if used_chunks \
                  else "\n\n".join(c.get("text","") for c in chunks[:2])
        answer  = llm_call(
            f"Answer using only supported context.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}",
            max_tokens=1024,
        )
        return {
            "answer":            answer,
            "reflection_tokens": reflection_tokens,
            "citation_scores":   citation_scores,
        }


# ---------------------------------------------------------------------------
# Iterative RAG
# ---------------------------------------------------------------------------
class IterativeRAG(BaseComponent):
    SPEC = MANIFEST.get("iterative_rag")

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        query     = ctx["query"]
        cname     = ctx["collection_name"]
        k         = ctx.get("top_k", 5)
        qdrant    = get_qdrant_client()
        collected = []
        log       = []

        current_query = query
        for iteration in range(self.max_iterations):
            vec    = embed(current_query)
            points = dense_search(qdrant, cname, vec, k)
            new_texts = [p.payload.get("text","") for p in points]
            collected.extend(new_texts)
            log.append({"iteration": iteration + 1, "query": current_query,
                        "chunks_found": len(new_texts)})

            context = "\n".join(collected)
            gap_check = llm_call(
                f"Original question: {query}\n\n"
                f"Information gathered so far:\n{context[:1500]}\n\n"
                "Is there any information still missing to fully answer the question? "
                "If yes, state what is missing as a search query. "
                "If no, reply COMPLETE.",
                max_tokens=100,
            ).strip()

            if "COMPLETE" in gap_check.upper():
                log[-1]["status"] = "complete"
                break
            current_query = gap_check
            log[-1]["status"] = "gap_found"

        answer = llm_call(
            f"Answer fully using the gathered context.\n\n"
            f"Context:\n{chr(10).join(collected[:8])}\n\nQuestion: {query}",
            max_tokens=1024,
        )
        return {"answer": answer, "iteration_log": log}


# ---------------------------------------------------------------------------
# Recursive RAG
# ---------------------------------------------------------------------------
class RecursiveRAG(BaseComponent):
    SPEC = MANIFEST.get("recursive_rag")

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def _decompose_and_answer(self, query: str, cname: str,
                               depth: int, qdrant) -> Dict:
        if depth >= self.max_depth:
            vec    = embed(query)
            points = dense_search(qdrant, cname, vec, 3)
            ctx    = "\n".join(p.payload.get("text","") for p in points)
            return {"query": query, "answer": llm_call(
                f"Context:\n{ctx}\n\nQuestion: {query}", max_tokens=300
            ), "children": []}

        sub_raw = llm_call(
            f"Can this question be answered directly, or must it be broken down?\n"
            f"Question: {query}\n\n"
            "If answerable directly, reply DIRECT.\n"
            "If it needs breakdown, list 2-3 sub-questions (numbered):",
            max_tokens=150,
        )

        if "DIRECT" in sub_raw.upper():
            vec    = embed(query)
            points = dense_search(qdrant, cname, vec, 3)
            ctx    = "\n".join(p.payload.get("text","") for p in points)
            return {"query": query, "answer": llm_call(
                f"Context:\n{ctx}\n\nQuestion: {query}", max_tokens=300
            ), "children": []}

        sub_questions = [
            l.strip().lstrip("0123456789.-) ")
            for l in sub_raw.strip().splitlines() if l.strip()
        ][:3]

        children = [
            self._decompose_and_answer(sq, cname, depth + 1, qdrant)
            for sq in sub_questions
        ]
        child_answers = "\n".join(
            f"Sub-answer: {c['answer']}" for c in children
        )
        synthesis = llm_call(
            f"Synthesise these sub-answers to answer: {query}\n\n{child_answers}",
            max_tokens=512,
        )
        return {"query": query, "answer": synthesis, "children": children}

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query", "collection_name")
        qdrant = get_qdrant_client()
        tree   = self._decompose_and_answer(
            ctx["query"], ctx["collection_name"], 0, qdrant
        )
        return {"answer": tree["answer"], "decomp_tree": tree}


# ---------------------------------------------------------------------------
# Agentic RAG
# ---------------------------------------------------------------------------
class AgenticRAG(BaseComponent):
    SPEC = MANIFEST.get("agentic_rag")

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._require(ctx, "query")
        query      = ctx["query"]
        cname      = ctx.get("collection_name", "")
        qdrant     = get_qdrant_client()
        tool_calls: List[Dict] = []
        agent_trace: List[str] = []
        gathered:   List[str]  = []

        TOOLS = {
            "search_knowledge_base": lambda q: [
                p.payload.get("text","")[:300]
                for p in dense_search(qdrant, cname, embed(q), 3)
            ] if cname else [],
            "summarise": lambda t: llm_call(f"Summarise: {t}", max_tokens=100),
        }

        for step in range(5):
            decision = llm_call(
                f"You are a RAG agent. Query: {query}\n"
                f"Information gathered: {gathered}\n\n"
                f"Available tools: {list(TOOLS.keys())}\n"
                "What should I do next? Format: TOOL:<tool_name>:<tool_input>\n"
                "Or ANSWER:<final answer> if done.",
                max_tokens=200,
            ).strip()

            agent_trace.append(f"Step {step+1}: {decision[:80]}")

            if decision.upper().startswith("ANSWER:"):
                final_answer = decision[7:].strip()
                return {
                    "answer":      final_answer,
                    "tool_calls":  tool_calls,
                    "agent_trace": agent_trace,
                }

            if decision.upper().startswith("TOOL:"):
                parts = decision[5:].split(":", 1)
                if len(parts) == 2:
                    tool_name, tool_input = parts[0].strip(), parts[1].strip()
                    if tool_name in TOOLS:
                        result = TOOLS[tool_name](tool_input)
                        tool_calls.append({"tool": tool_name,
                                           "input": tool_input,
                                           "output": str(result)[:200]})
                        if isinstance(result, list):
                            gathered.extend(result)
                        else:
                            gathered.append(str(result))

        # Fallback: answer with what we have
        context = "\n".join(gathered)
        answer  = llm_call(
            f"Context:\n{context}\n\nQuestion: {query}", max_tokens=512
        )
        return {"answer": answer, "tool_calls": tool_calls, "agent_trace": agent_trace}


__all__ = [
    "CorrectiveRAG", "SelfRAG", "IterativeRAG", "RecursiveRAG", "AgenticRAG",
]
