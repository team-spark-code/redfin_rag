# src/core/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import settings
from core.lifespan import lifespan
from routers.redfin import router as redfin_router
from routers.news import router as news_router

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app.service_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {"status": "ok", "see": ["/docs", "/redfin_target-insight"]}

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    app.include_router(redfin_router)
    app.include_router(news_router)
    
    return app
