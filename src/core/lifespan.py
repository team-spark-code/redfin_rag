# src/core/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core import settings
from observability.mongo_logger import init_mongo, ensure_collections

# RAG 인덱스 로드는 서비스 기동 시점에 한 번만 수행
from services import rag_service

@asynccontextmanager
def lifespan(app: FastAPI):
    # === startup ===
    init_mongo()
    ensure_collections()

    # 기본 진단 정보 출력
    print("[diag] settings.news.ingest_source   =", getattr(settings.news, "ingest_source", "http"))
    print("[diag] settings.news.source_collection =", getattr(settings.news, "source_collection", "extract"))
    print("[diag] settings.news.api_url          =", getattr(settings.news, "api_url", None))

    # (1) RAG 인덱스 초기화: /redfin_target-insight 에서 사용
    try:
        if hasattr(rag_service, "init_index_auto"):
            res_rag = rag_service.init_index_auto()
            print("[rag init_index_auto]", res_rag)
        elif hasattr(rag_service, "init_index"):
            if not settings.news.api_url:
                print("[rag init_index] skipped: settings.news.api_url is empty (http mode)")
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

    # === app running ===
    yield
    # === shutdown ===
