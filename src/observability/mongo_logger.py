import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

try:
    from core.settings import settings as _settings
except Exception:
    _settings = None

_client: Optional[MongoClient] = None
_db_name: str = ""


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    if _settings is not None:
        v = getattr(_settings, name, None)
        if v:
            return v
    return os.getenv(name, default)


def _ensure_client() -> Optional[MongoClient]:
    global _client, _db_name
    if _client is not None:
        return _client
    uri = _get_env("MONGO_URI") or _get_env("MONGODB_URI") or "mongodb://127.0.0.1:27017"
    _db_name = _get_env("MONGO_DB", "redfin") or "redfin"
    try:
        _client = MongoClient(uri, connect=False, serverSelectionTimeoutMS=1500)
        _client.admin.command("ping")
        return _client
    except Exception:
        return None


def init_mongo() -> None:
    try:
        _ensure_client()
    except Exception:
        pass


def get_db():
    client = _ensure_client()
    if client is None:
        return None
    try:
        return client.get_database(_db_name)
    except Exception:
        return None


def get_collection(name: str):
    db = get_db()
    if db is None:
        return None
    try:
        return db.get_collection(name)
    except Exception:
        return None


def logs_collection_name() -> str:
    return _get_env("MONGO_COL", "rag_logs") or "rag_logs"


def news_collection_name() -> str:
    return _get_env("NEWS_COL", "news_posts") or "news_posts"


def get_logs_collection():
    return get_collection(logs_collection_name())


def get_news_collection():
    return get_collection(news_collection_name())


def insert_doc(col_name: str, doc: Dict[str, Any]) -> Optional[str]:
    col = get_collection(col_name)
    if col is None:
        return None
    try:
        res = col.insert_one(doc)
        return str(getattr(res, "inserted_id", None))
    except PyMongoError:
        return None


def log_api_event(
    envelope: Dict[str, Any],
    status: int,
    endpoint: str,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    col_name: Optional[str] = None,
) -> None:
    col = get_collection(col_name or logs_collection_name())
    if col is None:
        return
    doc = {
        "created_at": datetime.now(timezone.utc),
        "endpoint": endpoint,
        "status": status,
        "envelope": envelope,
        "error": error,
        "extra": extra or {},
    }
    try:
        col.insert_one(doc)
    except PyMongoError:
        return


def ensure_collections() -> None:
    db = get_db()
    if db is None:
        return
    logs_name = logs_collection_name()
    if logs_name not in db.list_collection_names():
        try:
            db.create_collection(logs_name)
        except Exception:
            pass
    try:
        db[logs_name].create_index([("endpoint", 1), ("created_at", -1)])
        db[logs_name].create_index([("extra.request_meta.user_id", 1), ("created_at", -1)])
        db[logs_name].create_index([("extra.client.ip", 1), ("created_at", -1)])
        db[logs_name].create_index([("extra.duration_ms", -1), ("created_at", -1)])
    except Exception:
        pass
    news_name = news_collection_name()
    if news_name not in db.list_collection_names():
        try:
            db.create_collection(news_name)
        except Exception:
            pass
    try:
        db[news_name].create_index([("post_id", 1)], unique=True, sparse=True)
        db[news_name].create_index([("created_at", -1)])
        db[news_name].create_index([("category", 1), ("created_at", -1)])
        db[news_name].create_index([("tags", 1), ("created_at", -1)])
        db[news_name].create_index([("article_code", 1)], unique=True, sparse=True)
        db[news_name].create_index([("url", 1)], unique=True, sparse=True)
    except Exception:
        pass


def current_mongo_config() -> Dict[str, Any]:
    return {
        "mongo_uri": _get_env("MONGO_URI") or _get_env("MONGODB_URI") or "mongodb://127.0.0.1:27017",
        "db": _db_name or "redfin",
        "logs_collection": logs_collection_name(),
        "news_collection": news_collection_name(),
    }
