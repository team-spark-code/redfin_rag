# services/rag_service.py
from __future__ import annotations
import uuid
from typing import Dict, Tuple, Any

from nureongi import build_index, build_embedder, as_retriever, build_rag_chain
from observability.langsmith import set_ls_project, maybe_redact, build_trace_config

EMB = None            # 임베딩 모델 핸들
VSTORE = None         # 벡터스토어
RETRIEVER = None      # LangChain retriever
_CHAIN_CACHE: Dict[Tuple[str, str], Any] = {}  # (persona, strategy) -> chain

def init_index(news_url: str, emb_model: str,
               chunk_size: int, chunk_overlap: int,
               use_raptor: bool, distance: str):
    """
    인덱스 초기화: 로더→청크→(옵션: 인덱싱 시 RAPTOR)→임베딩→Vector DB
    """
    global EMB, VSTORE, RETRIEVER, _CHAIN_CACHE
    EMB = build_embedder(emb_model)
    VSTORE, info = build_index(
        news_url, EMB,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        index_mode="summary_only",     # 전략에 따라 'hybrid'로 확장 가능
        use_raptor=use_raptor,         # 인덱싱 시 RAPTOR
        distance=distance,             # 'cosine' 등
    )
    RETRIEVER = as_retriever(
        VSTORE,
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.25},
    )
    _CHAIN_CACHE.clear()
    return info

def set_search_kwargs(k: int, fetch_k: int, lambda_mult: float):
    if RETRIEVER is None:
        raise RuntimeError("Retriever not initialized. Call init_index() first.")
    RETRIEVER.search_kwargs.update({"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})

def get_chain(persona: str, strategy: str):
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
    service_name: str = "redfin_target-insight",
):
    chain = get_chain(persona, strategy)
    q_for_log = maybe_redact(question)
    trace_cfg = build_trace_config(service_name=service_name, user_id=user_id, plan=req_meta.get("plan"))
    with set_ls_project(service_name):
        
        # 기존 (문제 발생 가능)
        # answer_text = chain.invoke({"question": q_for_log}, config=trace_cfg)

        # 수정
        q_input = q_for_log
        if isinstance(q_input, dict):
            # 혹시 dict가 들어오면 내부 'question' 키를 우선, 없으면 전체를 문자열화
            q_input = q_input.get("question", str(q_input))
        elif not isinstance(q_input, str):
            q_input = str(q_input)

        answer_text = chain.invoke({"question": q_input}, config=trace_cfg)

    return {
        "version": "v1",
        "data": {
            "answer": {"text": str(answer_text), "format": "markdown"},
            "persona": persona,
            "strategy": strategy,
            "sources": [],   # 필요시 chain에서 반환하도록 확장
        },
        "meta": {
            "service": service_name,
            "user": {"user_id": user_id or "notuser", "session_id": str(uuid.uuid4())},
            "request": req_meta,
            "pipeline": {
                "index_mode": "summary_only",
                "use_raptor": True,  # 쿼리 시 RAPTOR는 chain 내부에서 필수 적용
                "embedding_model": req_meta.get("plan", {}).get("emb_model", "BAAI/bge-base-en-v1.5"),
            },
        },
    }
