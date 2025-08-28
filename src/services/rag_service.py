# src/services/rag_service.py
from __future__ import annotations
import uuid
from typing import Dict, Tuple, Any, List, Optional
from datetime import datetime, timedelta, timezone

from nureongi import build_index, build_embedder, as_retriever, build_rag_chain
from observability.langsmith import set_ls_project, maybe_redact, build_trace_config

EMB = None            # embedder handle
VSTORE = None         # vector store
RETRIEVER = None      # LangChain retriever
_CHAIN_CACHE: Dict[Tuple[str, str], Any] = {}  # (persona, strategy) -> chain


def init_index(
    news_url: str,
    emb_model: str,
    chunk_size: int,
    chunk_overlap: int,
    use_raptor: bool,
    distance: str,
):
    global EMB, VSTORE, RETRIEVER, _CHAIN_CACHE
    EMB = build_embedder(emb_model)
    VSTORE, info = build_index(
        news_url,
        EMB,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        index_mode="summary_only",
        use_raptor=use_raptor,
        distance=distance,
    )
    RETRIEVER = as_retriever(
        VSTORE,
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.25},
    )
    _CHAIN_CACHE.clear()
    return info


# ---------- search caps ----------
def _index_size() -> int:
    try:
        if hasattr(VSTORE, "_collection"):  # Chroma
            c = VSTORE._collection.count()
            if c is not None:
                return int(c)
        if hasattr(VSTORE, "get"):         # LangChain Chroma fallback
            d = VSTORE.get(include=["metadatas"], limit=1_000_000)
            return len(d.get("ids", []))
    except Exception:
        pass
    try:
        if hasattr(VSTORE, "index") and hasattr(VSTORE.index, "ntotal"):  # FAISS
            return int(VSTORE.index.ntotal)
    except Exception:
        pass
    return 0


def _cap_pair(k: int, fetch_k: int) -> Tuple[int, int]:
    n = _index_size()
    if n <= 0:
        return k, fetch_k
    k = max(1, min(int(k), n))
    fetch_k = max(k, min(int(fetch_k), n))
    return k, fetch_k


def set_search_kwargs(k: int, fetch_k: int, lambda_mult: float):
    if RETRIEVER is None:
        raise RuntimeError("Retriever not initialized. Call init_index() first.")
    k, fetch_k = _cap_pair(k, fetch_k)
    RETRIEVER.search_kwargs.update(
        {"k": k, "fetch_k": fetch_k, "lambda_mult": float(lambda_mult)}
    )


# ---------- chains ----------
def get_chain(persona: str, strategy: str):
    if RETRIEVER is None:
        raise RuntimeError("Retriever not initialized. Call init_index() first.")
    key = (persona, strategy or "stuff")
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = build_rag_chain(RETRIEVER, persona=persona, strategy=strategy)
    return _CHAIN_CACHE[key]


# ---------- run: generic (RAPTOR ON by default) ----------
def run_query(
    question: str,
    persona: str,
    strategy: str,
    user_id: str,
    req_meta: Dict,
    service_name: str = "redfin_target-insight",
):
    chain = get_chain(persona, strategy)
    q_for_log = maybe_redact(question)
    trace_cfg = build_trace_config(
        service_name=service_name, user_id=user_id, plan=req_meta.get("plan")
    )
    with set_ls_project(service_name):
        q_input = q_for_log
        if isinstance(q_input, dict):
            q_input = q_input.get("question", str(q_input))
        elif not isinstance(q_input, str):
            q_input = str(q_input)
        answer_text = chain.invoke({"question": q_input}, config=trace_cfg)

    use_raptor_hint = True
    try:
        use_raptor_hint = bool(req_meta.get("pipeline_hints", {}).get("use_raptor", True))
    except Exception:
        pass

    return {
        "version": "v1",
        "data": {
            "answer": {"text": str(answer_text), "format": "markdown"},
            "persona": persona,
            "strategy": strategy,
            "sources": [],
        },
        "meta": {
            "service": service_name,
            "user": {"user_id": user_id or "notuser", "session_id": str(uuid.uuid4())},
            "request": req_meta,
            "pipeline": {
                "index_mode": "summary_only",
                "use_raptor": use_raptor_hint,
                "embedding_model": req_meta.get("plan", {}).get(
                    "emb_model", "BAAI/bge-base-en-v1.5"
                ),
            },
        },
    }


# ---------- news-only helpers (RAPTOR OFF) ----------
def _build_news_filter(
    categories: Optional[List[str]],
    tags: Optional[List[str]],
    recency_days: int
) -> Optional[Dict[str, Any]]:
    ands: List[Dict[str, Any]] = []
    try:
        since_ts = (datetime.now(timezone.utc) - timedelta(days=max(1, int(recency_days)))).timestamp()
    except Exception:
        since_ts = None
    if since_ts and since_ts > 0:
        ands.append({"published_at_ts": {"$gte": since_ts}})
    if categories:
        ands.append({"category": {"$in": categories}})
    if tags:
        ands.append({"tags": {"$in": tags}})
    return {"$and": ands} if ands else None


def run_query_news(
    *,
    question: str,
    persona: str = "news_brief",
    user_id: str = "newsbot",
    req_meta: Optional[Dict[str, Any]] = None,
    service_name: str = "redfin_rag",
    categories: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    recency_days: int = 14,
    k: int = 6,
    fetch_k: int = 12,
    lambda_mult: float = 0.2,
):
    k, fetch_k = _cap_pair(k, fetch_k)
    news_filter = _build_news_filter(categories, tags, recency_days)

    global RETRIEVER, _CHAIN_CACHE
    RETRIEVER = as_retriever(
        VSTORE,
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": float(lambda_mult),
            "filter": news_filter,
        },
    )
    _CHAIN_CACHE.clear()

    meta = dict(req_meta or {})
    meta.update(
        {
            "purpose": "news_publish",
            "news": {"categories": categories, "tags": tags, "recency_days": recency_days},
            "pipeline_hints": {"use_raptor": False, "strict_citation": True},
        }
    )

    return run_query(
        question=question,
        persona=persona,
        strategy="stuff",
        user_id=user_id,
        req_meta=meta,
        service_name=service_name,
    )
