# src/api_rag.py
import logging
from core.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    from core import settings

    # ───────── 로깅 기본 설정 ─────────
    logging.basicConfig(
        level=logging.INFO,  # DEBUG | INFO | WARNING | ERROR | CRITICAL
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("redfin_rag")

    logger.info("🚀 Starting Redfin RAG API server...")
    logger.info(f"Host={settings.app.host}, Port={settings.app.port}")
    logger.info(f"Service name={settings.app.service_name}")

    uvicorn.run(
        "api_rag:app",
        host=settings.app.host,
        port=settings.app.port,
        workers=4,             # 멀티 워커
        reload=False,          # 멀티 워커 쓸 땐 False 권장
        limit_concurrency=200,
        backlog=2048,
        log_level="info",  # uvicorn 자체 로그 레벨
    )
