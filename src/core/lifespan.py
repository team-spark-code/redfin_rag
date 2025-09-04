# src/core/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core import settings
from observability.mongo_logger import init_mongo, ensure_collections, get_news_collection

# 두 파이프라인 로드
from services import rag_service
from services import news_service
from services.news_service import publish_from_source  # ← Mongo 기반 출간 시드

@asynccontextmanager
async def lifespan(app: FastAPI):
    # === startup ===
    init_mongo()
    ensure_collections()

    # 진단 출력 (Mongo 기반으로 변경된 핵심 값들)
    print("[diag] settings.news.seed_on_startup =", settings.news.seed_on_startup)
    print("[diag] settings.news.ingest_source   =", getattr(settings.news, "ingest_source", "http"))
    print("[diag] settings.news.source_collection =", getattr(settings.news, "source_collection", "extract"))
    print("[diag] settings.news.collection      =", settings.news.collection)
    print("[diag] settings.news.prompt_path     =", settings.news.prompt_path)

    # (1) 일반 RAG 인덱스 초기화: /redfin_target-insight 에서 사용
    #     - ingest_source == "mongo" 이면 api_url 없이도 인덱싱 수행
    #     - 최신 30개 제한은 indexing._load_news_docs(limit=30)에서 적용됨
    try:
        if hasattr(rag_service, "init_index_auto"):
            res_rag = rag_service.init_index_auto()
            print("[rag init_index_auto]", res_rag)
        elif hasattr(rag_service, "init_index"):
            # 구버전 호환: HTTP 모드만 지원
            if not settings.news.api_url:
                print("[rag init_index] skipped: settings.news.api_url is empty (legacy http mode)")
            else:
                res_rag = rag_service.init_index(
                    news_url=settings.news.api_url,
                    emb_model=settings.rag.emb_model,
                    chunk_size=1200,
                    chunk_overlap=120,
                    use_raptor=False,
                    distance="cosine",
                )
                print("[rag init_index] ok:", res_rag if res_rag is not None else "initialized")
        else:
            print("[rag init_index] skipped: function not found")
    except Exception as e:
        print("[warn] rag init_index failed:", e)

    # (2) 서버 시작 시 뉴스 자동 출간(시드)
    #     - 이제 API URL을 사용하지 않고, 항상 Mongo(redfin.extract)에서 읽음
    created = 0
    if bool(getattr(settings.news, "seed_on_startup", False)):
        try:
            print("[news] seeding from Mongo (extract) on startup …")
            result = publish_from_source(mongo_query={}, limit=30)
            created = int(result.get("count", 0))
            # upserts/modified 등도 함께 출력
            print(f"[news] seed result: ok={result.get('ok')} "
                  f"count={result.get('count')} upserts={result.get('upserts')} modified={result.get('modified')}")
        except Exception as e:
            print(f"[news] seed failed: {e}")
    else:
        print("[news] seeding skipped: seed_on_startup=False")

    # (3) 뉴스(출간본) 고정 인덱스 초기화 —— 반드시 '시드 후'에 실행 (문서 없으면 스킵)
    try:
        n_docs = 0
        col = get_news_collection()  # redfin.news_logs
        if col is not None:
            try:
                n_docs = col.count_documents({})
            except Exception:
                n_docs = 0

        if hasattr(news_service, "init_news_index_fixed"):
            if n_docs > 0 or created > 0:
                res_fixed = news_service.init_news_index_fixed(
                    chunk_size=1200,   # 1,200~1,400 권장
                    chunk_overlap=120  # 10%
                )
                print("[news fixed index]", res_fixed)
            else:
                print("[news fixed index] skipped: no docs in collection")
        else:
            print("[news fixed index] skipped: function not found")
    except Exception as e:
        print("[warn] news fixed index init failed:", e)

    # === app running ===
    yield
    # === shutdown ===
