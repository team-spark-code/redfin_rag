import uuid
from typing import Dict, Tuple

from nureongi import build_index, build_embedder, as_retriever, build_rag_chain
from observability.langsmith import set_ls_project, maybe_redact, build_trace_config

EMB, VSTORE, RETRIEVER = None, None, None
_CHAIN_CACHE: Dict[Tuple[str, str], object] = {}

def init_index(news_url, emb_model, chunk_size, chunk_overlap, use_raptor, distance):
    global EMB, VSTORE, RETRIEVER, _CHAIN_CACHE
    EMB = build_embedder(emb_model)
    VSTORE, info = build_index(
        news_url, EMB,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        index_mode="summary_only",
        use_raptor=use_raptor,
        distance=distance,
    )
    RETRIEVER = as_retriever(VSTORE, search_type="mmr",
                             search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.25})
    _CHAIN_CACHE.clear()
    return info

def set_search_kwargs(k: int, fetch_k: int, lambda_mult: float):
    # Retriever 파라미터 업데이트 전용 헬퍼
    RETRIEVER.search_kwargs.update({"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})

def get_chain(persona: str, strategy: str):
    key = (persona, strategy)
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

    # LangSmith 프로젝트 컨텍스트 + 입력 마스킹
    q_for_log = maybe_redact(question)
    trace_cfg = build_trace_config(service_name=service_name, user_id=user_id, plan=req_meta.get("plan"))

    with set_ls_project(service_name):
        # Runnable.invoke(config=...) 로 태그/메타 반영
        answer_text = chain.invoke(q_for_log, config=trace_cfg)

    return {
        "version": "v1",
        "data": {
            "answer": {"text": answer_text, "format": "markdown"},
            "persona": persona,
            "strategy": strategy,
            "sources": []
        },
        "meta": {
            "service": service_name,
            "user": {"user_id": user_id or "notuser", "session_id": str(uuid.uuid4())},
            "request": req_meta,
            "pipeline": {
                "index_mode": "summary_only",
                "use_raptor": True,
                "embedding_model": "BAAI/bge-base-en-v1.5"
            }
        }
    }

