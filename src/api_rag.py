# src/api_rag.py
from core.app import create_app

app = create_app()

if __name__ == "__main__":
    import os
    import uvicorn
    from core import settings
    uvicorn.run(
        "api_rag:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
