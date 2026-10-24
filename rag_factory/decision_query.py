# -*- coding: utf-8 -*-
"""
Decision-grade question answering over finalized documents
==========================================================
`/query` is a demo path: dense top-k, an LLM answer, guards that only log.
Business units take decisions on these answers, so this module does what
`/query` does not:

  1. Retrieval restricted to the corpus's CURRENT finalized versions
     (allowed_finalized_ids) and the caller's tenant, so an answer can never
     mix a superseded version or another tenant's document.
  2. Hybrid retrieval: dense vectors PLUS exact lexical matching of
     identifier-like tokens (lot / batch / invoice numbers) through a Qdrant
     full-text index, fused with reciprocal rank fusion. Dense embeddings are
     weak on codes like "VF-24-0917"; lexical matching is exact.
  3. Context blocks that carry identity (type, document, version, page, kind),
     so the model can attribute and cannot blend two documents' fields.
  4. A generation contract: every sentence cites [n]; numbers and identifiers
     are copied verbatim; otherwise the answer is exactly NOT_FOUND. The query
     is never rewritten (negations are kept as asked).
  5. Claim-level verification: each sentence must cite a block; every number
     and identifier in it must occur in the cited text (checked in code);
     then a strict entailment judge checks each claim against its sources.
  6. One regeneration with the verifier's findings, then ABSTAIN. A wrong
     answer is worse than no answer.

LLM and embedding calls go through components.base so tests can stub them.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

NOT_FOUND = "NOT_FOUND"
RRF_K = 60

# Identifier-like tokens: contain a digit and are not just a plain small number.
_ID_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/.]*\d[A-Za-z0-9\-_/.]*|[A-Za-z0-9]*\d[A-Za-z0-9\-_/.]*[A-Za-z][A-Za-z0-9\-_/.]*")
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
_CITE = re.compile(r"\[(\d{1,3})\]")


# --------------------------------------------------------------------------- #
# Small, pure helpers (unit-tested)
# --------------------------------------------------------------------------- #

def identifier_tokens(query: str, limit: int = 4) -> List[str]:
    """Tokens worth an exact lexical lookup: codes with digits, length >= 4."""
    out: List[str] = []
    for m in _ID_TOKEN.finditer(query):
        t = m.group(0).strip(".,;:/")
        if len(t) >= 4 and not re.fullmatch(r"\d{1,3}(?:[.,]\d+)?", t) and t not in out:
            out.append(t)
    return out[:limit]


def rrf(rank_lists: List[List[str]], k: int = RRF_K) -> Dict[str, float]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: Dict[str, float] = {}
    for lst in rank_lists:
        for rank, pid in enumerate(lst, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return scores


def latest_versions_only(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per document, keep only chunks of the highest version present."""
    best: Dict[str, int] = {}
    for c in chunks:
        d = str(c.get("doc_id"))
        best[d] = max(best.get(d, 0), int(c.get("version") or 0))
    return [c for c in chunks if int(c.get("version") or 0) == best[str(c.get("doc_id"))]]


def context_block(n: int, c: Dict[str, Any]) -> str:
    label = " · ".join(str(x) for x in [
        c.get("doc_type") or "document",
        f"document {c.get('doc_id')}",
        f"{c.get('finalized_document_id')} v{c.get('version')}",
        f"page {c['page_num']}" if c.get("page_num") is not None else None,
        c.get("chunk_kind") or "text",
    ] if x)
    body = c.get("window_text") or c.get("text") or ""
    return f"[{n}] ({label})\n{body}"


# Abbreviations whose full stop does not end a sentence (pharmacopoeia, titles, refs).
_ABBREV = re.compile(r"\b(Ph|Eur|USP|JP|BP|Dr|Mr|Ms|Prof|No|Nr|Ref|Fig|approx|vs|e\.g|i\.e|cf|etc|Inc|Ltd|Co|St)\.", re.I)
_PROTECT = "\u2024"  # one-dot leader stands in for a protected full stop


def split_claims(answer: str) -> List[str]:
    """Claims = segments ending in a citation group, else sentences. Headings are not claims.

    The model is told to end every sentence with its citations, so a citation
    group is the most reliable boundary; plain sentence splitting is only the
    fallback, with abbreviations like "Ph. Eur." protected.
    """
    parts: List[str] = []
    for line in answer.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        line = _ABBREV.sub(lambda m: m.group(0)[:-1] + _PROTECT, line)
        for seg in re.split(r"(?:(?<=\][.!?])|(?<=\]))\s+(?=\S)", line):
            seg = seg.strip()
            if not seg:
                continue
            if _CITE.search(seg):
                parts.append(seg)
            else:
                parts.extend(x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", seg) if x.strip())
    parts = [p.replace(_PROTECT, ".") for p in parts]
    claims = []
    for p in parts:
        bare = _CITE.sub("", p).strip(" #*:")
        cited = bool(_CITE.search(p))
        if not cited and ((len(bare) < 12 and not re.search(r"\d", bare)) or p.rstrip().endswith(":")):
            continue  # heading / filler: an uncited short line or a lead-in ending in ':'
        if not bare:
            continue
        if re.fullmatch(r"#+.*", p):
            continue
        claims.append(p)
    return claims


def _norm_num(s: str) -> Optional[float]:
    t = s.replace(" ", "")
    if "," in t and "." in t:
        t = t.replace(",", "") if t.rfind(".") > t.rfind(",") else t.replace(".", "").replace(",", ".")
    elif "," in t:
        t = t.replace(",", ".") if re.fullmatch(r"-?\d+,\d{1,2}", t) else t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def _squash(s: str) -> str:
    return re.sub(r"[\s,]", "", s).lower()


def grounded_tokens(claim: str, sources: List[str]) -> List[str]:
    """Numbers / identifiers in the claim that do NOT occur in any cited source."""
    text = _CITE.sub("", claim)
    src = " \n ".join(sources)
    src_sq = _squash(src)
    src_nums = [n for n in (_norm_num(x) for x in _NUMBER.findall(src)) if n is not None]
    missing: List[str] = []
    for tok in set(identifier_tokens(text, limit=50)):
        if _squash(tok) not in src_sq:
            missing.append(tok)
    for raw in _NUMBER.findall(text):
        if raw in missing:
            continue
        n = _norm_num(raw)
        if n is None:
            continue
        if not any(abs(n - s) <= max(1e-9, abs(s) * 1e-9) for s in src_nums) and _squash(raw) not in src_sq:
            missing.append(raw)
    return sorted(set(missing))


def check_claims(answer: str, blocks: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic checks: a citation per claim, citations exist, numbers grounded."""
    out = []
    for claim in split_claims(answer):
        cites = sorted({int(x) for x in _CITE.findall(claim)})
        problems: List[str] = []
        valid = [c for c in cites if c in blocks]
        if not cites:
            problems.append("no citation")
        if len(valid) != len(cites):
            problems.append(f"cites a source that does not exist: {sorted(set(cites) - set(valid))}")
        if valid:
            missing = grounded_tokens(claim, [blocks[c]["text_for_check"] for c in valid])
            if missing:
                problems.append("not in the cited source: " + ", ".join(missing))
        out.append({"text": claim, "citations": valid, "supported": not problems, "problems": problems})
    return out


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def answer_prompt(question: str, context: str, feedback: Optional[str] = None) -> Tuple[str, str]:
    system = (
        "You answer questions for a regulated pharmaceutical company using ONLY the numbered sources provided. "
        "Rules: (1) Every sentence must end with the citation(s) of the sources that support it, like [1] or [2][3]. "
        "(2) Copy numbers, units, dates and identifiers exactly as they appear in the sources. "
        "(3) Answer the question exactly as asked; keep negations (e.g. 'not', 'except') as written. "
        "(4) If the sources do not contain the answer, reply with exactly NOT_FOUND and nothing else. "
        "(5) If sources disagree, say so and cite each. (6) Never use outside knowledge, never guess, never round. "
        "(7) Be concise: plain sentences or short bullets, no headings."
    )
    user = f"Sources:\n\n{context}\n\nQuestion: {question}"
    if feedback:
        user += ("\n\nYour previous answer had statements that the sources do not support:\n" + feedback +
                 "\nAnswer again following the rules; drop anything you cannot support, or reply NOT_FOUND.")
    return system, user


def judge_prompt(claims: List[Dict[str, Any]], blocks: Dict[int, Dict[str, Any]]) -> str:
    items = []
    for i, c in enumerate(claims):
        srcs = "\n".join(f"  [{n}] {blocks[n]['text_for_check'][:1500]}" for n in c["citations"])
        items.append(f"Claim {i}: {_CITE.sub('', c['text']).strip()}\nSources:\n{srcs}")
    return (
        "You are a strict verifier for a pharmaceutical quality system. For each claim, decide whether its sources "
        "FULLY support it (every fact, number, unit, negation and qualifier). Paraphrase is fine; anything added, "
        "inferred, rounded or contradicted is NOT supported.\n\n" + "\n\n".join(items) +
        '\n\nReply with JSON only: {"verdicts": [{"claim": <index>, "supported": true|false, "reason": "<short>"}]}'
    )


def parse_judge(raw: str, n: int) -> List[Optional[Dict[str, Any]]]:
    m = re.search(r"\{.*\}", raw, re.S)
    out: List[Optional[Dict[str, Any]]] = [None] * n
    if not m:
        return out
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for v in data.get("verdicts", []):
        i = v.get("claim")
        if isinstance(i, int) and 0 <= i < n:
            out[i] = {"supported": bool(v.get("supported")), "reason": str(v.get("reason", ""))[:200]}
    return out


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def _filter(tenant: Optional[str], allowed: Optional[List[str]], extra: Optional[Dict[str, Any]]):
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
    must = []
    if tenant:
        must.append(FieldCondition(key="tenant_id", match=MatchValue(value=tenant)))
    if allowed:
        must.append(FieldCondition(key="finalized_document_id", match=MatchAny(any=list(allowed))))
    for k, v in (extra or {}).items():
        must.append(FieldCondition(key=k, match=MatchValue(value=v)))
    return Filter(must=must) if must else None


def ensure_text_index(qdrant, collection: str, _cache: set = set()) -> bool:
    """Full-text index on `text` for exact identifier matching (idempotent)."""
    if collection in _cache:
        return True
    try:
        from qdrant_client.models import TextIndexParams, TokenizerType
        qdrant.create_payload_index(collection, field_name="text", field_schema=TextIndexParams(
            type="text", tokenizer=TokenizerType.WORD, lowercase=True, min_token_len=2, max_token_len=40))
    except Exception:
        pass  # exists already, or not supported (in-memory client)
    _cache.add(collection)
    return True


def retrieve(qdrant, collection: str, query: str, vec: List[float], top_k: int, flt) -> Dict[str, Any]:
    from qdrant_client.models import FieldCondition, Filter, MatchText
    dense = qdrant.query_points(collection_name=collection, query=vec, limit=max(top_k * 2, 8),
                                with_payload=True, query_filter=flt).points
    by_id: Dict[str, Any] = {str(p.id): p for p in dense}
    dense_ids = [str(p.id) for p in dense]
    dense_score = {str(p.id): float(p.score) for p in dense}
    lexical_lists: List[List[str]] = []
    tokens = identifier_tokens(query)
    if tokens:
        ensure_text_index(qdrant, collection)
    for tok in tokens:
        must = list(flt.must) if flt is not None else []
        must.append(FieldCondition(key="text", match=MatchText(text=tok)))
        try:
            hits, _ = qdrant.scroll(collection_name=collection, scroll_filter=Filter(must=must), limit=top_k,
                                    with_payload=True, with_vectors=False)
        except Exception:
            hits = []
        # Keep only chunks that contain the token verbatim (the word tokenizer splits on '-').
        hits = [h for h in hits if _squash(tok) in _squash(str((h.payload or {}).get("text", "")))]
        for h in hits:
            by_id.setdefault(str(h.id), h)
        if hits:
            lexical_lists.append([str(h.id) for h in hits])
    fused = rrf([dense_ids] + lexical_lists)
    ranked = sorted(fused, key=lambda pid: -fused[pid])
    chunks = []
    for pid in ranked:
        p = by_id[pid]
        c = dict(p.payload or {})
        c["_id"] = pid
        c["_dense_score"] = dense_score.get(pid)
        c["_lexical"] = any(pid in lst for lst in lexical_lists)
        chunks.append(c)
    chunks = latest_versions_only(chunks)[:top_k]
    return {
        "chunks": chunks,
        "tokens": tokens,
        "lexical_hits": sum(len(x) for x in lexical_lists),
        "top_dense_score": max(dense_score.values()) if dense_score else None,
    }


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

def answer_question(*, qdrant, collection: str, question: str, tenant: Optional[str],
                    allowed_finalized_ids: Optional[List[str]], filters: Optional[Dict[str, Any]],
                    top_k: int, min_score: float, embed, llm_call, verify_with_judge: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    flt = _filter(tenant, allowed_finalized_ids, filters)
    r = retrieve(qdrant, collection, question, embed(question), top_k, flt)
    chunks = r["chunks"]
    base = {"question": question, "retrieval": {k: v for k, v in r.items() if k != "chunks"}}

    def done(**kw):
        kw.setdefault("citations", [])
        kw.setdefault("claims", [])
        return {**base, **kw, "elapsed_ms": int((time.time() - t0) * 1000)}

    if not chunks:
        return done(abstained=True, reason="no finalized source matched the question", answer=None)
    top = r["top_dense_score"] or 0.0
    if min_score > 0 and top < min_score and not r["lexical_hits"]:
        return done(abstained=True, reason=f"closest source is too dissimilar (score {top:.2f} < {min_score:.2f})",
                    answer=None, citations=_citations(chunks))

    blocks: Dict[int, Dict[str, Any]] = {}
    for i, c in enumerate(chunks, start=1):
        c["text_for_check"] = (c.get("window_text") or c.get("text") or "")
        blocks[i] = c
    context = "\n\n".join(context_block(i, c) for i, c in blocks.items())

    feedback = None
    claims: List[Dict[str, Any]] = []
    answer = ""
    attempts = 0
    for attempts in (1, 2):
        system, user = answer_prompt(question, context, feedback)
        answer = llm_call(user, system=system, max_tokens=900, temperature=0.0).strip()
        if answer.strip().upper().startswith(NOT_FOUND):
            return done(abstained=True, reason="the finalized records do not contain the answer", answer=None,
                        citations=_citations(chunks), attempts=attempts)
        claims = check_claims(answer, blocks)
        if not claims:
            feedback = "- (the answer made no checkable statement)"
            continue
        if verify_with_judge and all(c["supported"] for c in claims):
            verdicts = parse_judge(llm_call(judge_prompt(claims, blocks), max_tokens=900, temperature=0.0), len(claims))
            for c, v in zip(claims, verdicts):
                if v is None:
                    c["supported"] = False
                    c["problems"].append("verifier gave no verdict")
                elif not v["supported"]:
                    c["supported"] = False
                    c["problems"].append(f"verifier: {v['reason']}")
                else:
                    c["verifier"] = v["reason"]
        bad = [c for c in claims if not c["supported"]]
        if not bad:
            return done(abstained=False, answer=answer, claims=claims, citations=_citations(chunks), attempts=attempts)
        feedback = "\n".join(f"- {_CITE.sub('', c['text']).strip()} ({'; '.join(c['problems'])})" for c in bad)
    return done(abstained=True, reason="not every statement could be verified against the sources",
                answer=None, draft=answer, claims=claims, citations=_citations(chunks), attempts=attempts)


def _citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "n": i,
        "doc_id": c.get("doc_id"),
        "finalized_document_id": c.get("finalized_document_id"),
        "version": c.get("version"),
        "doc_type": c.get("doc_type"),
        "page_num": c.get("page_num"),
        "chunk_kind": c.get("chunk_kind"),
        "score": c.get("_dense_score"),
        "lexical": c.get("_lexical", False),
        "text": (c.get("text") or "")[:1500],
    } for i, c in enumerate(chunks, start=1)]
