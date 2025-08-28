# src/routers/news.py
# redfin_news: 출간 + 조회(목록/단건)

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder

from schemas.news import NewsPublishRequest
from services.news_service import (
    generate_news_post,
    publish_from_feed,
    publish_from_env,
)
from observability.mongo_logger import get_news_collection

router = APIRouter(prefix="/redfin_news", tags=["redfin_news"])


@router.post("/publish")
def publish(req: NewsPublishRequest):
    post = generate_news_post(req)
    return jsonable_encoder(post.dict())


@router.post("/publish_feed")
def publish_feed(payload: Dict[str, Any]):
    feed_url: str = payload.get("feed_url")
    item_path: Optional[str] = payload.get("item_path")
    field_map: Optional[Dict[str, str]] = payload.get("field_map")
    default_publish: bool = bool(payload.get("default_publish", True))
    default_top_k: int = int(payload.get("default_top_k", 6))
    if not feed_url:
        raise HTTPException(status_code=400, detail="feed_url is required")
    return publish_from_feed(
        feed_url=feed_url,
        item_path=item_path,
        field_map=field_map,
        default_publish=default_publish,
        default_top_k=default_top_k,
    )


@router.post("/publish_from_env")
def publish_from_env_route():
    return publish_from_env()


@router.get("/posts")
def list_posts(
    limit: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0),
    status: Optional[str] = Query(None, pattern="^(draft|published)$"),
):
    col = get_news_collection()
    if col is None:
        raise HTTPException(status_code=500, detail="news collection unavailable")
    q: Dict[str, Any] = {}
    if status:
        q["status"] = status
    cur = (
        col.find(q, {"_id": 0})
        .sort([("created_at", -1)])
        .skip(skip)
        .limit(limit)
    )
    items = list(cur)
    return jsonable_encoder(items)


@router.get("/posts/{post_id}")
def get_post(post_id: str):
    col = get_news_collection()
    if col is None:
        raise HTTPException(status_code=500, detail="news collection unavailable")
    doc = col.find_one({"post_id": post_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="post not found")
    return jsonable_encoder(doc)
