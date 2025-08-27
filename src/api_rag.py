# src/api_rag.py
import os
from dataclasses import asdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv

from schemas.query import QueryRequest
from schemas.response import QueryResponseV1
from services import rag_service
from services.strategy import choose_strategy_advanced
from nureongi import PersonaSlug, get_persona_by_alias

load_dotenv(find_dotenv(), override=False)

NEWS_API_URL = os.getenv("NEWS_API_URL", "http://192.168.0.123:8000/news/extract")
EMB_MODEL    = os.getenv("EMB_MODEL", "BAAI/bge-base-en-v1.5")
SERVICE_NAME = "redfin_target-insight"  # ✅ 서비스명 상수

app = FastAPI(title="Redfin Target Insight API", version="0.1.0")

# 허용할 오리진들 (필요한 것만 넣어도 되고, 개발 중엔 regex로 대충 풀어도 됨)
ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://localhost:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,      # 또는 allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    allow_credentials=False,
    allow_methods=["*"],                # preflight에서 접근할 메서드 허용
    allow_headers=["*"],                # Content-Type 등 헤더 허용
)

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
        return "ai_industry_professional"  # 'auto' 대신 기본 슬러그로 고정 권장
    spec = get_persona_by_alias(user_persona)
    # spec이 None이면 입력 그대로, 있으면 slug.value 반환
    return spec.slug.value if spec else user_persona

@app.get("/")  # 루트 확인용
def root():
    return {"status": "ok", "see": ["/docs", "/redfin_target-insight"]}

@app.get("/healthz")  # 헬스 체크
def healthz():
    return {"ok": True}

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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_rag:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT","8001")),
        reload=True)

