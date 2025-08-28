import os
from datetime import datetime, timezone
from pymongo import MongoClient

_client = None
_col = None

def init_mongo():
    global _client, _col
    if _client:
        return
    uri = os.getenv("MONGODB_URI", "mongodb://100.97.183.123:27017")
    db_name = os.getenv("MONGO_DB", "redfin")
    col_name = os.getenv("MONGO_COL", "rag_logs")

    _client = MongoClient(uri)
    db = _client[db_name]
    _col = db[col_name]

def log_api_event(envelope: dict, status: int, endpoint: str, error: str = None, extra: dict = None):
    """
    envelope: RAG API 응답 JSON
    status: HTTP 상태 코드
    endpoint: 호출 엔드포인트
    error: 에러 메시지(있을 경우)
    extra: 추가 메타데이터
    """
    if _col is None:
        init_mongo()
    doc = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "status": status,
        "envelope": envelope,
        "error": error,
        "extra": extra or {},
    }
    _col.insert_one(doc)
