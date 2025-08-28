# src/core/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core import settings
from services import rag_service
from observability.mongo_logger import init_mongo

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    rag_service.init_index(
        news_url=settings.NEWS_API_URL,
        emb_model=settings.EMB_MODEL,
        chunk_size=500,
        chunk_overlap=120,
        use_raptor=True,
        distance="cosine",
    )
    init_mongo()
    yield
    # shutdown (필요 시 정리 작업)
