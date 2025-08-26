# src/api_rag.py
import os
from dataclasses import asdict
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv

from schemas.query import QueryRequest
from schemas.response import QueryResponseV1
from services import rag_service
from services.strategy import choose_strategy_advanced
from nureongi import PersonaSlug, get_persona_by_alias

load_dotenv(find_dotenv(), override=False)

NEWS_API_URL = os.getenv("NEWS_API_URL", "http://127.0.0.1:8000/news")
EMB_MODEL    = os.getenv("EMB_MODEL", "BAAI/bge-base-en-v1.5")
SERVICE_NAME = "redfin_target-insight"  # ✅ 서비스명 상수

app = FastAPI(title="Redfin Target Insight API", version="0.1.0")

@app.on_event("startup")
def startup_event():
    rag_service.init_index(
        news_url=NEWS_API_URL,
        emb_model=EMB_MODEL,
        chunk_size=500,
        chunk_overlap=120,
        use_raptor=True,
        distance="cosine",
    )

def resolve_persona(user_persona: str | None) -> str:
    if not user_persona:
        return "auto"
    try:
        slug = get_persona_by_alias(user_persona)
        return slug.value if isinstance(slug, PersonaSlug) else str(slug)
    except Exception:
        return user_persona

@app.post(
    "/redfin_target-insight",
    response_model=QueryResponseV1,
    tags=["redfin_target-insight"],                 # ✅ 문서에서 한 눈에
    summary="RAG 답변 생성 (redfin_target-insight)", # ✅ Swagger 요약
    description="사용자 입력 프롬프트를 받아 RAG로 인사이트를 생성합니다."
)
def redfin_target_insight(req: QueryRequest):
    token_budget = int(os.getenv("TOKEN_BUDGET", "3500"))
    plan = choose_strategy_advanced(
        question=req.question, est_context_tokens=None,
        k_hint=req.top_k, doc_count_hint=None, token_budget=token_budget,
    )
    rag_service.set_search_kwargs(k=plan.k, fetch_k=plan.fetch_k, lambda_mult=plan.lambda_mult)

    persona_slug = resolve_persona(req.persona)
    strategy = plan.strategy if req.strategy == "auto" else req.strategy

    req_meta = {
        "service": SERVICE_NAME,        # ✅ 메타에 서비스명 명시
        "question": req.question,
        "top_k": plan.k, "fetch_k": plan.fetch_k, "lambda_mult": plan.lambda_mult,
        "plan": asdict(plan),
    }

    return rag_service.run_query(      # ✅ 서비스명도 함께 전달
        question=req.question,
        persona=persona_slug,
        strategy=strategy,
        user_id=(req.user_id or "notuser"),
        req_meta=req_meta,
        service_name=SERVICE_NAME,     # LangSmith 프로젝트 매핑에 사용
    )
