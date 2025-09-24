from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from core import settings

_client: Optional[MongoClient] = None


def _ensure_client() -> Optional[MongoClient]:
    global _client
    if _client is not None:
        return _client

    uri = settings.mongo.uri
    to = settings.mongo.timeout_ms
    try:
        _client = MongoClient(
            uri,
            connect=False,  # lazy
            serverSelectionTimeoutMS=to,
            connectTimeoutMS=to,
            socketTimeoutMS=to,
        )
        _client.admin.command("ping")
        print(f"[mongo] connected -> {uri} (db={settings.mongo.db})")
        return _client
    except Exception as e:
        print(f"[mongo] connect failed: {e} (uri={uri})")
        _client = None
        return None


def init_mongo() -> None:
    _ensure_client()


def get_db():
    c = _ensure_client()
    if c is None:
        return None
    return c[settings.mongo.db]


def get_collection(name: str):
    db = get_db()
    if db is None:
        return None
    return db.get_collection(name)


def logs_collection_name() -> str:
    return settings.mongo.logs_collection


def get_logs_collection():
    return get_collection(logs_collection_name())


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
    """Ensure required collections and indexes exist."""
    db = get_db()
    if db is None:
        print("[mongo] unavailable; skip ensure_collections")
        return

    try:
        existing = set(db.list_collection_names())
    except ServerSelectionTimeoutError as e:
        print(f"[mongo] list_collection_names timeout: {e}")
        return
    except Exception as e:
        print(f"[mongo] list_collection_names failed: {e}")
        return

    logs_name = logs_collection_name()

    if logs_name not in existing:
        try:
            db.create_collection(logs_name)
        except Exception:
            pass

    try:
        db[logs_name].create_index([("endpoint", 1), ("created_at", -1)])
        db[logs_name].create_index([("extra.request_meta.user_id", 1), ("created_at", -1)])
        db[logs_name].create_index([("extra.client.ip", 1), ("created_at", -1)])
        db[logs_name].create_index([("extra.duration_ms", -1), ("created_at", -1)])
        db[logs_name].create_index([("extra.retrieval.raptor_applied", 1), ("created_at", -1)])
    except Exception:
        pass
