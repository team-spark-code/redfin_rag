# src/services/rag_service.py
from __future__ import annotations

import os
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

# ───────── 기존 공용 파이프라인(일반 RAG) ─────────
from nureongi import build_index, build_embedder, as_retriever, build_rag_chain

# ⬇️ 변경: 전역 ENV 스왑(set_ls_project) 제거, 요청 단위 트레이서 주입 방식으로 전환
# - 라우터에서 run_config(= callbacks/tags/metadata 포함)를 넘겨주면 그대로 사용
# - 라우터에서 run_config가 없을 때만 내부 기본값으로 보강
from observability.langsmith import (
    make_tracer_for,          # (백업용) 내부에서 tracer 생성할 때 사용
    maybe_redact,
    build_trace_config,
)

# ====== 공용(일반 RAG) 핸들 ======
EMB = None            # embedder handle
VSTORE = None         # vector store
RETRIEVER = None      # LangChain retriever
_CHAIN_CACHE: Dict[Tuple[str, str], Any] = {}  # (persona, strategy) -> chain


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


def _ensure_run_config(
    service_name: str,
    user_id: str,
    plan: Dict[str, Any] | None,
    run_config: Optional[Dict[str, Any]],
):
    """
    라우터에서 run_config를 주면 그대로 사용.
    없으면 최소한의 tracer/config를 내부에서 보강 생성.
    """
    if run_config:
        return run_config
    tracer = make_tracer_for(service_name)
    cfg = build_trace_config(service_name=service_name, user_id=user_id, plan=plan)
    cfg["callbacks"] = [tracer]
    return cfg


def run_query(
    question: str,
    persona: str,
    strategy: str,
    user_id: str,
    req_meta: Dict,
    service_name: str | None = None,
    run_config: Optional[Dict[str, Any]] = None,   # ⬅️ 추가: 요청 단위 트레이스 설정
):
    """일반 RAG 실행(최종 호출에서 반드시 config=run_config 전달)."""
    from core import settings  # 지연 임포트(순환 참조 방지)

    service_name = service_name or settings.app.service_name
    chain = get_chain(persona, strategy)

    # 입력 마스킹(PII 등)
    q_for_log = maybe_redact(question)

    # 요청 단위 트레이스 설정 준비(라우터 전달 > 내부 기본)
    cfg = _ensure_run_config(
        service_name=service_name,
        user_id=user_id,
        plan=req_meta.get("plan"),
        run_config=run_config,
    )

    # LangChain Runnable 실행 시 **항상** config 전달
    q_input = q_for_log if isinstance(q_for_log, str) else str(q_for_log)
    answer_text = chain.invoke({"question": q_input}, config=cfg)

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
                    "emb_model", getattr(settings.rag, "emb_model", None)
                ),
            },
        },
    }
