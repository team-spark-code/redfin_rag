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

    rag_service.init_index(
        news_url=settings.NEWS_API_URL,
        emb_model=settings.EMB_MODEL,
        chunk_size=500,
        chunk_overlap=120,
        use_raptor=True,
        distance="cosine",
    )

    if settings.NEWS_API_URL:
        try:
            result = publish_from_env()
            created = len(result.get("created", []))
            skipped = len(result.get("skipped", []))
            errors = len(result.get("errors", []))
            print(f"[news] seeded from NEWS_API_URL: created={created} skipped={skipped} errors={errors}")
        except Exception as e:
            print(f"[news] seed failed: {e}")

    yield
    # shutdown
