# src/core/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import settings
from core.lifespan import lifespan
from routers.redfin import router as redfin_router

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app.service_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],         # 모든 ??리????용
        allow_origin_regex=".*",     # ??규??으로도 모든 ??리????용
        allow_credentials=True,      # 쿠키/??션 ??증????용??려??True
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

    return app
