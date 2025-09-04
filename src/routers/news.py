# -*- coding: utf-8 -*-
# =============================================================================
# 변경 요약 (2025-09-04)
# - LangSmith 트레이서를 라우터에서 "요청 단위"로 생성하여 services에 run_config로 전달.
# - 프로젝트명은 settings.news.langsmith_project(없으면 settings.news.service_name) 사용.
# - callbacks/tags/metadata 를 포함한 run_config 구조 표준화.
# - 기존 env 기반 프로젝트 스왑(전역 컨텍스트 교체) 제거 → 동시성 안전.
# =============================================================================

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List

from core.settings import settings
from schemas.news import NewsPublishRequest
from services import news_service

# [CHG] LangSmith/Callback 임포트 (LangChain 호환)
try:
    # LangChain 0.3+ 권장: langsmith의 LangChainTracer
    from langchain.callbacks.tracers import LangChainTracer  # type: ignore
    _HAS_TRACER = True
except Exception:
    _HAS_TRACER = False

router = APIRouter(prefix="/redfin_news", tags=["redfin_news"])


# ----------------------- 내부 유틸 -----------------------

def _make_run_config(endpoint: str, extra_tags: Optional[List[str]] = None,
                     extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    요청 단위 LangSmith 설정을 만들어 반환.
    - callbacks: [LangChainTracer(project_name=...)]
    - tags: 엔드포인트/서비스 태그
    - metadata: 서비스/엔드포인트/호출 옵션
    """
    tags = ["news", endpoint]
    if extra_tags:
        tags.extend(extra_tags)

    # 프로젝트명 결정
    project = getattr(settings.news, "langsmith_project", None) \
              or getattr(settings.news, "service_name", None) \
              or "default"

    callbacks = []
    if _HAS_TRACER:
        try:
            # [CHG] 요청 단위 트레이서 생성
            tracer = LangChainTracer(project_name=project)
            callbacks.append(tracer)
        except Exception:
            # 트레이서 생성 실패 시 callbacks 비움(서비스에서 경고 로그 출력)
            callbacks = []

    metadata = {
        "service": getattr(settings.news, "service_name", "redfin_news"),
        "project": project,
        "endpoint": endpoint,
    }
    if extra_meta:
        metadata.update(extra_meta)

    # LangChain RunnableConfig 규격
    return {
        "callbacks": callbacks,    # 없으면 [] (서비스에서 경고 출력)
        "tags": tags,
        "metadata": metadata,
    }


# ----------------------- 라우트 -----------------------

@router.post("/publish", summary="단건 출간")
def publish(req: NewsPublishRequest):
    """
    단건 입력을 받아 뉴스 포스트 생성(LLM 사용 여부는 설정에 따름)
    """
    try:
        run_config = _make_run_config(endpoint="publish")
        post = news_service.generate_news_post(req, feed_meta=None, run_config=run_config)  # [CHG]
        return {"ok": True, "post": post.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish_batch", summary="JSON 배열 일괄 출간")
def publish_batch(
    items: List[Dict[str, Any]],
    default_publish: bool = True,
    default_top_k: int = 6,
):
    try:
        run_config = _make_run_config(endpoint="publish_batch",
                                      extra_meta={"default_publish": default_publish, "default_top_k": default_top_k})
        res = news_service.publish_batch(
            items=items,
            field_map=None,
            default_publish=default_publish,
            default_top_k=default_top_k,
            run_config=run_config,  # [CHG]
        )
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish_from_feed", summary="外부 피드에서 로딩해 출간")
def publish_from_feed(
    feed_url: Optional[str] = None,
    default_publish: bool = True,
    default_top_k: int = 6,
):
    try:
        run_config = _make_run_config(endpoint="publish_from_feed",
                                      extra_meta={"feed_url": feed_url or settings.news.api_url})
        res = news_service.publish_from_feed(
            feed_url=feed_url or settings.news.api_url,
            item_path=None,
            field_map=None,
            default_publish=default_publish,
            default_top_k=default_top_k,
            run_config=run_config,   # [CHG]
        )
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish_from_env", summary="ENV에 설정된 피드로 출간")
def publish_from_env():
    try:
        run_config = _make_run_config(endpoint="publish_from_env")
        res = news_service.publish_from_env(run_config=run_config)  # [CHG]
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish_from_source", summary="Mongo extract에서 읽어 출간(옵션)")
def publish_from_source(limit: int = 30):
    try:
        run_config = _make_run_config(endpoint="publish_from_source",
                                      extra_meta={"limit": limit})
        res = news_service.publish_from_source(mongo_query=None, limit=limit)  # 이 경로는 내부에서 LLM 호출 시 tracer 사용 안함
        # ↑ publish_from_source는 내부에서 _get_llm_for_news() → llm.invoke()를 직접 쓰므로
        #   LangSmith에 기록하려면 거기서도 callbacks를 쓰는 구현이 필요.
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/posts/{post_id}", summary="출간 포스트 조회")
def get_post(post_id: str):
    from observability.mongo_logger import get_news_collection
    col = get_news_collection()
    if col is None:
        raise HTTPException(status_code=500, detail="mongo collection unavailable")
    doc = col.find_one({"post_id": post_id})
    if not doc:
        raise HTTPException(status_code=404, detail="not found")
    # ObjectId 등 직렬화 보정은 FastAPI가 해줌
    return {"ok": True, "post": doc}


@router.get("/posts", summary="최근 포스트 리스트")
def list_posts(limit: int = Query(20, ge=1, le=100)):
    from observability.mongo_logger import get_news_collection
    col = get_news_collection()
    if col is None:
        raise HTTPException(status_code=500, detail="mongo collection unavailable")
    cur = col.find({}).sort([("created_at", -1)]).limit(limit)
    return {"ok": True, "posts": list(cur)}
