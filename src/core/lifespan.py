# src/core/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core import settings
from services import rag_service
from observability.mongo_logger import init_mongo, ensure_collections
from services.news_service import publish_from_env

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    init_mongo()
    ensure_collections()
    
    # 진단용 출력
    import os
    print("[diag] NEWS__SEED_ON_STARTUP =", os.getenv("NEWS__SEED_ON_STARTUP"))
    print("[diag] NEWS_SEED_ON_STARTUP  =", os.getenv("NEWS_SEED_ON_STARTUP"))
    print("[diag] settings.news.seed_on_startup =", settings.news.seed_on_startup)
    print("[diag] settings.news.api_url =", settings.news.api_url)

    # 1) 기본 인덱스 (RAG)
    rag_service.init_index(
        news_url=settings.news.api_url,
        emb_model=settings.rag.emb_model,
        chunk_size=500,
        chunk_overlap=120,
        use_raptor=True,
        distance="cosine",
    )

    # 2) 뉴스 전용 시맨틱 인덱스
    try:
        rag_service.init_news_index_semantic()
    except Exception as e:
        print(f"[warn] news semantic index init failed: {e}")

    # 3) 서버 시작 시 뉴스 자동 출간 (옵션 기능)
    #    - .env에 다음을 추가해야 동작합니다:
    #      NEWS_API_URL=https://example.com/feed
    #      NEWS_SEED_ON_STARTUP=true
    #    - NEWS_SEED_ON_STARTUP=false 이거나 설정이 없으면 실행하지 않습니다.
    #    - 나중에 제외하고 싶으면 아래 블록 전체를 주석 처리하세요.
    try:
        seed_flag = str(getattr(settings.news, "seed_on_startup", "false")).lower() in ("1","true","yes","on")
    except Exception:
        seed_flag = False

    if seed_flag and settings.news.api_url:
        try:
            print("[news] seeding from NEWS_API_URL on startup...")
            result = publish_from_env()   # (비동기 함수라면 await 붙여야 함)
            created = len(result.get("created", []))
            skipped = len(result.get("skipped", []))
            errors  = len(result.get("errors", []))
            print(f"[news] seed result: created={created} skipped={skipped} errors={errors}")
        except Exception as e:
            print(f"[news] seed failed: {e}")

    yield
    # shutdown
