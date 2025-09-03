# src/routers/news.py
# redfin_news: 출간 + 조회(목록/단건)

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder

# settings 임포트 (프로젝트 구조에 맞춰 선택)
try:
    from core.settings import settings  # 권장
except Exception:
    from core import settings

from schemas.news import NewsPublishRequest
from services.news_service import (
    generate_news_post,
    publish_from_feed,
    publish_from_env,
    init_index,
    init_news_index_semantic,
    init_news_index_fixed,
)
from observability.mongo_logger import get_news_collection

# LangSmith: 요청 단위 프로젝트 분리
from observability.langsmith import make_tracer_explicit, build_trace_config

router = APIRouter(prefix="/redfin_news", tags=["redfin_news"])

def _news_run_cfg(user_id: str = "notuser") -> Dict[str, Any]:
    """뉴스 전용 LangSmith run_config 생성 유틸."""
    tracer = make_tracer_explicit(getattr(settings.news, "langsmith_project", None) or "redfin_news-publish")
    cfg = build_trace_config(service_name="redfin_news", user_id=user_id, plan=None)
    cfg["callbacks"] = [tracer] if tracer else []
    return cfg

@router.post("/publish", status_code=status.HTTP_201_CREATED)
def publish(req: NewsPublishRequest):
    if not (req.title or req.content):
        raise HTTPException(status_code=400, detail="title or content is required")
    run_cfg = _news_run_cfg(user_id="system")
    post = generate_news_post(req, run_config=run_cfg)
    return jsonable_encoder(post.dict())

@router.post("/publish_feed")
def publish_feed_route(payload: Dict[str, Any]):
    feed_url: str = payload.get("feed_url")
    item_path: Optional[str] = payload.get("item_path")
    field_map: Optional[Dict[str, str]] = payload.get("field_map")
    default_publish: bool = str(payload.get("default_publish", True)).lower() in ("1", "true", "yes", "on")
    try:
        default_top_k: int = int(payload.get("default_top_k", 6))
    except Exception:
        default_top_k = 6

    if not feed_url:
        raise HTTPException(status_code=400, detail="feed_url is required")

    run_cfg = _news_run_cfg(user_id="system")
    return publish_from_feed(
        feed_url=feed_url,
        item_path=item_path,
        field_map=field_map,
        default_publish=default_publish,
        default_top_k=default_top_k,
        run_config=run_cfg,
    )

@router.post("/publish_from_env")
def publish_from_env_route():
    try:
        run_cfg = _news_run_cfg(user_id="system")
        return publish_from_env(run_config=run_cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"publish_from_env failed: {e}")

@router.post("/init_index")
def route_init_index():
    """Chroma 컬렉션 생성/열기만 수행."""
    return init_index()

@router.post("/init_index_semantic")
def route_init_index_semantic():
    """시맨틱 청킹으로 인덱스 초기화 (ingest_source에 맞춰 입력 로드)."""
    return init_news_index_semantic()

@router.post("/init_index_fixed")
def route_init_index_fixed(chunk_size: int = 1200, chunk_overlap: int = 120):
    """고정 크기+오버랩 청킹 인덱스 초기화 (ingest_source에 맞춰 입력 로드)."""
    return init_news_index_fixed(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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

    # created_at/updated_at 정렬 키가 없을 수도 있으니 폴백 처리
    try:
        cur = (
            col.find(q, {"_id": 0})
            .sort([("created_at", -1), ("updated_at", -1), ("post_id", -1)])
            .skip(skip)
            .limit(limit)
        )
    except Exception:
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
