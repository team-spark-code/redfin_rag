# src/services/news_service.py
# ----------------------------------------------------------------------
# 변경 요약
# - NewsLoader(기존 nureongi/loaders.py) 직접 사용 → 피드 로딩/매핑/정규화 일원화
# - publish_from_feed/publish_from_env → 내부적으로 publish_from_loader 호출
# - generate_news_post: run_query_news(RAPTOR OFF) 사용, feed 메타는 model_meta.feed에 저장
# - 중복 판단: guid/article_code/url 우선, 없으면 doc_id 보조
# ----------------------------------------------------------------------

import os
import json
import urllib.request  # (직접 JSON 배열을 넣는 publish_batch 용도 보존)
import re
from uuid import uuid4
from typing import Any, Dict, Optional, List

from prompts.news import build_news_prompt
from schemas.news import NewsPublishRequest, NewsPost
from schemas.news_llm import NewsLLMOut
from services import rag_service
from observability.mongo_logger import get_news_collection
from nureongi.loaders import NewsLoader, DEFAULT_FIELD_MAP

# ===== 한국어 보증 유틸 =====
_HANGUL_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")  # (한글 탐지 정규식)

def _has_hangul_ratio(s: str, min_ratio: float = 0.10) -> bool:
    """응답이 한국어인지 간단 비율로 판정"""
    if not s:
        return False
    letters = [ch for ch in s if ch.isalpha() or '\uac00' <= ch <= '\ud7a3' or '\u3131' <= ch <= '\u318E']
    if not letters:
        return False
    hanguls = _HANGUL_RE.findall(s)
    return (len(hanguls) / max(1, len(letters))) >= min_ratio

def _translate_json_to_ko(json_text: str) -> str:
    """LLM이 영어로 내보내면, JSON의 값(value)만 한국어로 번역 (키/구조 보존)"""
    prompt = f"""
다음 JSON 문자열의 **값(value)** 들만 한국어로 번역하라. **키(key)/구조/순서/숫자/URL**은 변경 금지.
반드시 **순수 JSON만** 출력한다. 주석/설명/코드블록 금지.

입력 JSON:
\"\"\"{json_text.strip()}\"\"\"
"""
    resp = rag_service.run_query_news(
        question=prompt,
        persona="news_brief",   # 검색 불필요
        user_id="translator",
        req_meta={"purpose": "json_translate_to_ko"},
        service_name="redfin_rag",
        categories=None, tags=None, recency_days=0, k=1, fetch_k=1, lambda_mult=0.1,
    )
    if hasattr(resp, "dict"):
        out = resp.dict().get("data", {}).get("answer", {}).get("text") or ""
    else:
        out = str(resp)
    return out or json_text
# ============================


def _safe_json_block(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    return json.loads(s)


def _listify(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    return [str(v)]


def _dedupe_key(item: Dict[str, Any]) -> Dict[str, Any]:
    q: Dict[str, Any] = {}
    if item.get("article_code"):
        q["article_code"] = str(item["article_code"])
    elif item.get("article_id"):
        q["article_id"] = str(item["article_id"])
    elif item.get("url"):
        q["url"] = str(item["url"])
    return q


def _exists_in_news(q: Dict[str, Any]) -> bool:
    if not q:
        return False
    col = get_news_collection()
    if col is None:
        return False
    return col.find_one(q) is not None


# -------------------- 단건 생성 --------------------

def generate_news_post(req: NewsPublishRequest, feed_meta: Optional[Dict[str, Any]] = None) -> NewsPost:
    prompt = build_news_prompt(
        title=req.title,
        url=req.url,
        content=req.content,
        categories=req.categories,
        tags=req.tags,
    )

    resp = rag_service.run_query_news(
        question=prompt,
        persona="news_brief",
        user_id="newsbot",
        req_meta={
            "top_k": req.top_k,
            "article_key": getattr(req, "article_key", None),
            "source_url": req.url,
        },
        service_name="redfin_rag",
        categories=req.categories,
        tags=req.tags,
        recency_days=int(os.getenv("NEWS_RECENCY_DAYS", "14")),
        k=min((req.top_k or 6), 8),
        fetch_k=max(min(((req.top_k or 6) * 2), 16), 8),
        lambda_mult=0.2,
    )

    if hasattr(resp, "dict"):
        payload = resp.dict()
        text = payload.get("data", {}).get("answer", {}).get("text") or ""
    elif hasattr(resp, "model_dump"):
        text = resp.model_dump().get("data", {}).get("answer", {}).get("text") or ""
    else:
        text = str(resp)

    # ---- 한국어 보증: 영어 응답이면 JSON 값만 한국어로 번역 ----
    if not _has_hangul_ratio(text, min_ratio=0.10):
        text = _translate_json_to_ko(text)
    # ---------------------------------------------------------

    llm = _parse_llm_output(text)

    title = (llm.title if llm and llm.title else (req.title or "무제"))
    body_md = (llm.body_md if llm and llm.body_md else (req.content or ""))
    tldr_list = (llm.tldr if llm and llm.tldr else _fallback_tldr_from_text(body_md, k=3))
    tags = (llm.tags if llm and llm.tags else (req.tags or []))
    sources = (llm.sources if llm and llm.sources else [])
    hero = (llm.hero_image_url if llm and llm.hero_image_url else None)
    author = (llm.author_name if llm and llm.author_name else getattr(req, "author_name", None))
    category1 = (req.categories[0] if req.categories else None)

    post = NewsPost(
        post_id=str(uuid4()),
        article_id=req.article_id,
        article_code=getattr(req, "article_code", None),
        url=req.url,
        title=title,
        dek=(llm.subtitle if llm and llm.subtitle else None),
        tldr=tldr_list,
        body_md=body_md,
        hero_image_url=hero,
        author_name=author,
        category=category1,
        categories=req.categories or [],
        tags=tags,
        sources=sources,
        status="published" if req.publish else "draft",
        model_meta={
            "persona": "news_brief",
            "strategy": "auto",
            "top_k": req.top_k,
            "feed": feed_meta or {},
            "parsed_ok": bool(llm is not None),
        },
    )

    col = get_news_collection()
    if col is not None:
        try:
            col.insert_one(post.dict())
        except Exception:
            pass

    return post


# -------------------- NewsLoader 기반 배치 출간 --------------------

def _req_from_loader_doc(doc) -> NewsPublishRequest:
    md = dict(doc.metadata or {})
    # loader는 page_content = "title\n\ncontent" 형태; 여기선 전체를 content로 그대로 사용
    content = doc.page_content or ""
    categories: List[str] = []
    if md.get("content_type"):
        categories = [str(md["content_type"])]
    elif md.get("source"):
        categories = [str(md["source"])]

    return NewsPublishRequest(
        article_id=str(md.get("guid")) if md.get("guid") else None,
        article_code=str(md.get("guid") or md.get("doc_id")) if (md.get("guid") or md.get("doc_id")) else None,
        url=md.get("url"),
        title=md.get("title"),
        content=content,
        categories=categories,
        tags=[str(t) for t in (md.get("tags") or [])],
        publish=True,
        top_k=6,
    )


def publish_from_loader(
    feed_url: str,
    field_map: Optional[Dict[str, str]] = None,
    default_publish: bool = True,
    default_top_k: int = 6,
    timeout: int = 15,
) -> Dict[str, Any]:
    loader = NewsLoader(field_map=field_map or DEFAULT_FIELD_MAP)
    docs = loader.load_news_json(feed_url, timeout=timeout)

    created, skipped, errors = [], [], []
    for idx, doc in enumerate(docs):
        try:
            md = dict(doc.metadata or {})
            req = _req_from_loader_doc(doc)

            # 기본값 적용
            if default_publish is not None:
                req.publish = bool(default_publish)
            if default_top_k is not None:
                try:
                    req.top_k = int(default_top_k)
                except Exception:
                    req.top_k = 6

            # 중복 체크: guid/url 우선, 없으면 article_code(doc_id) 보조
            q = _dedupe_key({
                "article_code": req.article_code,
                "article_id": req.article_id,
                "url": req.url,
            }) or ({"article_code": str(md.get("doc_id"))} if md.get("doc_id") else {})
            if q and _exists_in_news(q):
                skipped.append({"index": idx, "reason": "duplicate", "key": q})
                continue

            post = generate_news_post(req, feed_meta=md)
            created.append(post.dict())
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
    return {"created": created, "skipped": skipped, "errors": errors}


# -------------------- 기존 JSON 배열 배치(옵션으로 유지) --------------------

def publish_batch(
    items: List[Dict[str, Any]],
    field_map: Optional[Dict[str, str]] = None,
    default_publish: bool = True,
    default_top_k: int = 6
) -> Dict[str, Any]:
    # 유지: 외부에서 이미 JSON 배열을 주입하는 경우 사용
    created, skipped, errors = [], [], []
    m = field_map or {}
    for idx, it in enumerate(items):
        try:
            # 최소 매핑
            article_code = str(it.get(m.get("article_code", "article_code")) or it.get("guid") or it.get("id") or "")
            article_id = str(it.get(m.get("article_id", "article_id")) or it.get("guid") or "")
            url = it.get(m.get("url", "url")) or it.get("link")
            title = it.get(m.get("title", "title"))
            content = it.get(m.get("content", "content")) or it.get("article_text")
            categories = it.get(m.get("categories", "categories")) or it.get("content_type")
            tags = it.get(m.get("tags", "tags")) or []

            req = NewsPublishRequest(
                article_id=article_id or None,
                article_code=article_code or None,
                url=url,
                title=title,
                content=content,
                categories=[categories] if isinstance(categories, str) else (categories or []),
                tags=[str(t) for t in (tags if isinstance(tags, list) else [tags] if tags else [])],
                publish=bool(it.get(m.get("publish", "publish"), default_publish)),
                top_k=int(it.get(m.get("top_k", "top_k"), default_top_k)),
            )

            q = _dedupe_key({"article_code": req.article_code, "article_id": req.article_id, "url": req.url})
            if q and _exists_in_news(q):
                skipped.append({"index": idx, "reason": "duplicate", "key": q})
                continue

            post = generate_news_post(req, feed_meta=it)
            created.append(post.dict())
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
    return {"created": created, "skipped": skipped, "errors": errors}


# -------------------- 외부 호출 진입점 --------------------

def publish_from_feed(
    feed_url: str,
    item_path: Optional[str] = None,      # NewsLoader가 dict{"items":[...]}도 자동 처리 → 보존하되 미사용
    field_map: Optional[Dict[str, str]] = None,
    default_publish: bool = True,
    default_top_k: int = 6
) -> Dict[str, Any]:
    # 이제는 NewsLoader 경로로 통일
    return publish_from_loader(
        feed_url=feed_url,
        field_map=field_map,
        default_publish=default_publish,
        default_top_k=default_top_k,
        timeout=15,
    )


def publish_from_env() -> Dict[str, Any]:
    feed_url = os.getenv("NEWS_API_URL")
    if not feed_url:
        raise ValueError("NEWS_API_URL is not set")

    # 환경에서 필드 매핑 덮어쓰기: DEFAULT_FIELD_MAP 위에 적용
    field_map_json = os.getenv("NEWS_FEED_FIELD_MAP")
    user_map = json.loads(field_map_json) if field_map_json else {}
    field_map = {**DEFAULT_FIELD_MAP, **user_map} if user_map else DEFAULT_FIELD_MAP

    default_publish = (os.getenv("NEWS_DEFAULT_PUBLISH", "1").lower() in ("1", "true", "yes"))
    try:
        default_top_k = int(os.getenv("NEWS_TOP_K", "6"))
    except Exception:
        default_top_k = 6

    return publish_from_loader(
        feed_url=feed_url,
        field_map=field_map,
        default_publish=default_publish,
        default_top_k=default_top_k,
        timeout=15,
    )

# ------------------- 헬퍼 -------------------

def _fallback_tldr_from_text(text: str, k: int = 3) -> List[str]:
    sents = re.split(r"(?<=[.!?。！？])\s+", (text or "").strip())
    out = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        out.append(s[:90])
        if len(out) >= max(1, k):
            break
    return out

def _parse_llm_output(text: str) -> Optional[NewsLLMOut]:
    try:
        data = _safe_json_block(text)
    except Exception:
        return None
    try:
        if hasattr(NewsLLMOut, "model_validate"):
            return NewsLLMOut.model_validate(data)   # pydantic v2
        return NewsLLMOut.parse_obj(data)            # pydantic v1
    except Exception:
        return None
