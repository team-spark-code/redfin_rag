# src/services/rag_service.py
from __future__ import annotations

import os
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

# ───────── 기존 공용 파이프라인(일반 RAG) ─────────
from nureongi import build_index, build_embedder, as_retriever, build_rag_chain
from observability.langsmith import set_ls_project, maybe_redact, build_trace_config

# ───────── 뉴스(시맨틱) 전용 인덱스 지원 ─────────
from core import settings
from langchain_huggingface import HuggingFaceEmbeddings
from nureongi.ingestion_news_semantic import build_news_semantic_index  # 뉴스 전용 시맨틱 인덱서
from langchain_community.vectorstores import Chroma

# ====== 공용(일반 RAG) 핸들 ======
EMB = None            # embedder handle
VSTORE = None         # vector store
RETRIEVER = None      # LangChain retriever
_CHAIN_CACHE: Dict[Tuple[str, str], Any] = {}  # (persona, strategy) -> chain

# ====== 뉴스(시맨틱) 전용 핸들 ======
_NEWS_VS = None       # news vector store
_NEWS_CHAIN_CACHE: Dict[Tuple[str, str], Any] = {}  # (persona, strategy) -> chain (뉴스 전용)


# ─────────────────────────────────────────────────────────────────────
#                           기본(일반 RAG)
# ─────────────────────────────────────────────────────────────────────
def init_index(
    news_url: str,
    emb_model: str,
    chunk_size: int,
    chunk_overlap: int,
    use_raptor: bool,
    distance: str,
):
    """
    기존 기본 인덱스 초기화(Recursive + (옵션)RAPTOR).
    - 일반 RAG 라인에서 사용
    """
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


def _index_size(vs=None) -> int:
    """벡터스토어 문서 수(cap 계산용)."""
    target = vs or VSTORE
    if target is None:
        return 0
    try:
        # Chroma(new API)
        if hasattr(target, "_collection"):
            c = target._collection.count()
            if c is not None:
                return int(c)
        # Chroma(LC wrapper)
        if hasattr(target, "get"):
            d = target.get(include=["metadatas"], limit=1_000_000)
            return len(d.get("ids", []))
    except Exception:
        pass
    try:
        # FAISS
        if hasattr(target, "index") and hasattr(target.index, "ntotal"):
            return int(target.index.ntotal)
    except Exception:
        pass
    return 0


def _cap_pair(k: int, fetch_k: int, vs=None) -> Tuple[int, int]:
    """(k, fetch_k)를 인덱스 크기에 맞게 보정."""
    n = _index_size(vs=vs)
    if n <= 0:
        return k, fetch_k
    k = max(1, min(int(k), n))
    fetch_k = max(k, min(int(fetch_k), n))
    return k, fetch_k


def set_search_kwargs(k: int, fetch_k: int, lambda_mult: float):
    """일반 RAG retriever의 검색 파라미터 업데이트."""
    if RETRIEVER is None:
        raise RuntimeError("Retriever not initialized. Call init_index() first.")
    k, fetch_k = _cap_pair(k, fetch_k, vs=VSTORE)
    RETRIEVER.search_kwargs.update(
        {"k": k, "fetch_k": fetch_k, "lambda_mult": float(lambda_mult)}
    )


def get_chain(persona: str, strategy: str):
    """일반 RAG 체인(전역 RETRIEVER 기반, 캐시)."""
    if RETRIEVER is None:
        raise RuntimeError("Retriever not initialized. Call init_index() first.")
    key = (persona, strategy or "stuff")
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = build_rag_chain(RETRIEVER, persona=persona, strategy=strategy)
    return _CHAIN_CACHE[key]


def run_query(
    question: str,
    persona: str,
    strategy: str,
    user_id: str,
    req_meta: Dict,
    service_name: str = None,
):
    """일반 RAG 실행(기존 경로)."""
    service_name = service_name or settings.app.service_name
    chain = get_chain(persona, strategy)
    q_for_log = maybe_redact(question)
    trace_cfg = build_trace_config(
        service_name=service_name, user_id=user_id, plan=req_meta.get("plan")
    )
    with set_ls_project(service_name):
        q_input = q_for_log if isinstance(q_for_log, str) else str(q_for_log)
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
                    "emb_model", settings.rag.emb_model
                ),
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────
#                           뉴스(시맨틱) 전용
# ─────────────────────────────────────────────────────────────────────
def _emb() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.rag.emb_model)


def init_news_index_semantic() -> None:
    """
    서버 부팅 시, 뉴스 전용(시맨틱) 인덱스를 초기화.
    - 별도 컬렉션(settings.news.collection) 사용
    - 기존 일반 인덱스와 분리 운용
    """
    global _NEWS_VS, _NEWS_CHAIN_CACHE
    vs, info = build_news_semantic_index(
        news_url=settings.news.api_url,
        emb_model=settings.rag.emb_model,
        collection_name=settings.news.collection,  # e.g., "news_semantic_v1"
        persist_dir=os.getenv("CHROMA_DIR", "./.chroma"),
        distance="cosine",
    )
    _NEWS_VS = vs
    _NEWS_CHAIN_CACHE.clear()
    print(f"[news] semantic index ready: {info}")


@lru_cache(maxsize=1)
def get_news_vectorstore():
    """
    뉴스 전용 컬렉션을 VectorStore로 붙여서 반환.
    - 부팅 때 init_news_index_semantic()을 안 불러도, 기존 컬렉션이 있으면 cold attach 가능
    """
    global _NEWS_VS
    if _NEWS_VS is not None:
        return _NEWS_VS
    vs = Chroma(
        collection_name=settings.news.collection,
        embedding_function=_emb(),
        persist_directory=os.getenv("CHROMA_DIR", "./.chroma"),
    )
    _NEWS_VS = vs
    return _NEWS_VS


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


def _get_news_chain(persona: str, strategy: str, retriever) -> Any:
    """뉴스 전용 체인(뉴스 retriever 기반, 별도 캐시)."""
    key = (persona, strategy or "stuff")
    if key not in _NEWS_CHAIN_CACHE:
        _NEWS_CHAIN_CACHE[key] = build_rag_chain(retriever, persona=persona, strategy=strategy)
    return _NEWS_CHAIN_CACHE[key]


def run_query_news(
    *,
    question: str,
    persona: str = "news_brief",
    user_id: str = "newsbot",
    req_meta: Optional[Dict[str, Any]] = None,
    service_name: str = None,
    categories: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    recency_days: int = 14,
    k: int = 6,
    fetch_k: int = 12,
    lambda_mult: float = 0.2,
):
    """
    뉴스 라인 전용 실행:
    - 시맨틱 컬렉션(VectorStore)에서 retriever 생성
    - 카테고리/태그/기간 필터를 retriever.search_kwargs.filter로 전달
    - RAPTOR OFF (pipeline_hints)
    """
    service_name = service_name or "redfin_rag"

    news_vs = get_news_vectorstore()
    k, fetch_k = _cap_pair(k, fetch_k, vs=news_vs)
    news_filter = _build_news_filter(categories, tags, recency_days)

    # 뉴스용 retriever를 매 호출 시 생성(필터/파라미터 반영)
    news_retriever = news_vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": float(lambda_mult),
            "filter": news_filter,
        },
    )

    # 뉴스용 체인(뉴스 retriever 기반)
    chain = _get_news_chain(persona=persona, strategy="stuff", retriever=news_retriever)

    meta = dict(req_meta or {})
    meta.update(
        {
            "purpose": "news_publish",
            "news": {"categories": categories, "tags": tags, "recency_days": recency_days},
            "pipeline_hints": {"use_raptor": False, "strict_citation": True},
        }
    )

    q_for_log = maybe_redact(question)
    trace_cfg = build_trace_config(service_name=service_name, user_id=user_id, plan=meta.get("plan"))
    with set_ls_project(service_name):
        q_input = q_for_log if isinstance(q_for_log, str) else str(q_for_log)
        answer_text = chain.invoke({"question": q_input}, config=trace_cfg)

    return {
        "version": "v1",
        "data": {
            "answer": {"text": str(answer_text), "format": "markdown"},
            "persona": persona,
            "strategy": "stuff",
            "sources": [],
        },
        "meta": {
            "service": service_name,
            "user": {"user_id": user_id or "notuser", "session_id": str(uuid.uuid4())},
            "request": meta,
            "pipeline": {
                "index_mode": "semantic_v1",
                "use_raptor": False,
                "embedding_model": settings.rag.emb_model,
                "collection": settings.news.collection,
            },
        },
    }
