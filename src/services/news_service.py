# src/services/news_service.py
# ----------------------------------------------------------------------
# 변경 요약
# - 라우터에서 전달한 run_config(=callbacks/tags/metadata 포함)를
#   체인/LLM 호출에 그대로 전달하도록 시그니처 확장.
# - env 스왑 컨텍스트(langsmith_project) 제거. (동시성 안전)
# - 나머지 퍼블리시 플로우/중복 판정/Loader 연계는 기존 유지.
# ----------------------------------------------------------------------

import os
import json
import urllib.request
import re
from uuid import uuid4
from typing import Any, Dict, Optional, List

from schemas.news import NewsPublishRequest, NewsPost
from schemas.news_llm import NewsLLMOut
from observability.mongo_logger import get_news_collection
from nureongi.loaders import NewsLoader, DEFAULT_FIELD_MAP
from nureongi.vs_news import upsert_news_post_to_chroma
from nureongi.vectorstore import create_chroma_store

# 독립 LLM/체인
from nureongi.llm import build_default_llm
from nureongi.news_chain import build_news_chain

# 템플릿
from core.settings import settings
from nureongi.prompt_loader import load_md_template, render_template

# ===== (시맨틱 인덱스용) 임베딩/청킹 =====
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_core.documents import Document  # 타입 힌트 용도
except Exception:
    HuggingFaceEmbeddings = None
    SemanticChunker = None
    Document = None

# RecursiveCharacterTextSplitter 임포트 (버전에 따른 폴백 처리)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings  # 임베딩 필요


# (옵션) 문자열→불리언/정수 파서
def _as_bool(v: object | None, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _as_int(v: object | None, default: int = 0) -> int:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "":
            return default
        return int(s)
    except Exception:
        return default


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


# -------------------- 초기화 함수(수명주기에서 호출) --------------------

def init_index() -> dict:
    """뉴스용 Chroma 컬렉션 생성/열기."""
    try:
        vs = create_chroma_store(
            collection_name=settings.news.collection,
            persist_dir=settings.news.persist_dir,
            distance="cosine",
        )
        vs.persist()
        return {
            "ok": True,
            "collection": settings.news.collection,
            "persist_dir": settings.news.persist_dir,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def init_news_index_semantic() -> dict:
    """
    뉴스 전용 '시맨틱 인덱스' 초기화:
    - settings.news.api_url에서 기사 로드 → 시맨틱 청킹 → Chroma에 업서트
    """
    try:
        if not settings.news.api_url:
            return {"ok": False, "reason": "settings.news.api_url is empty"}
        if HuggingFaceEmbeddings is None or SemanticChunker is None:
            return {"ok": False, "reason": "semantic stack unavailable (langchain_experimental / huggingface not installed)"}

        loader = NewsLoader(field_map=DEFAULT_FIELD_MAP)
        docs = loader.load_news_json(settings.news.api_url, timeout=15)
        if not docs:
            return {"ok": False, "reason": "no documents loaded", "api_url": settings.news.api_url}

        embed_for_chunk = HuggingFaceEmbeddings(model_name=settings.news.emb_model)
        splitter = SemanticChunker(embeddings=embed_for_chunk, breakpoint_threshold_type="percentile")
        chunks = splitter.split_documents(docs)

        vs = create_chroma_store(
            collection_name=settings.news.collection,
            persist_dir=settings.news.persist_dir,
            distance="cosine",
        )
        vs.vs.add_documents(chunks)
        vs.persist()

        return {
            "ok": True,
            "api_url": settings.news.api_url,
            "collection": settings.news.collection,
            "persist_dir": settings.news.persist_dir,
            "docs": len(docs),
            "chunks": len(chunks),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -------------------- 단건 생성 --------------------

def generate_news_post(
    req: NewsPublishRequest,
    feed_meta: Optional[Dict[str, Any]] = None,
    *,
    run_config: Optional[Dict[str, Any]] = None,   # ⬅️ 추가: 요청 단위 트레이스 설정
) -> NewsPost:
    """
    .md 템플릿을 로드하여 LLM에 전달.
    - retriever 없음(뉴스용 단건 요약/가공)
    - 라우터에서 받은 run_config를 체인 invoke에 그대로 전달
    """
    # 1) 템플릿 로딩/렌더링
    tpl = load_md_template(settings.news.prompt_path or "src/prompts/templates/news_publish_v1.md")

    # 안전장치: 인사이트 템플릿 섞임 차단
    _tpl_str = str(tpl)
    if ("Issue List" in _tpl_str) or ("target-insight" in _tpl_str):
        raise RuntimeError("뉴스 출간에서 인사이트 템플릿이 감지됨. settings.news.prompt_path를 확인하세요.")

    prompt = render_template(
        tpl,
        title=(req.title or ""),
        url=(req.url or ""),
        content=(req.content or ""),
        categories=", ".join(req.categories or []),
        tags=", ".join(req.tags or []),
        meta=json.dumps(feed_meta or {}, ensure_ascii=False),
    )

    # 2) 요약/가공
    if not settings.news.use_llm:
        text = json.dumps({
            "title": req.title or "Untitled",
            "subtitle": None,
            "tldr": [],
            "body_md": req.content or "",
            "tags": req.tags or [],
            "sources": [req.url] if req.url else [],
            "hero_image_url": None,
            "author_name": getattr(req, "author_name", None),
        }, ensure_ascii=False)
    else:
        llm = build_default_llm()
        chain = build_news_chain(llm, settings.news.prompt_path or "src/prompts/templates/news_publish_v1.md")
        inputs = {
            "title": (req.title or ""),
            "url": (req.url or ""),
            "content": (req.content or ""),
            "categories": ", ".join(req.categories or []),
            "tags": ", ".join(req.tags or []),
            "meta": json.dumps(feed_meta or {}, ensure_ascii=False),
        }
        
        # generate_news_post 내부, chain.invoke 전에 (디버그용)
        if run_config is None or not isinstance(run_config, dict) or not run_config.get("callbacks"):
            print("[warn] news run_config.callbacks missing; LangSmith project may fallback to env")
        else:
            # 콜백 클래스/프로젝트명 대략 확인
            try:
                cb = run_config["callbacks"][0]
                print("[diag] news callback tracer =", type(cb).__name__, getattr(cb, "project_name", None))
            except Exception:
                pass
        
        # ⬇️ 중요: 요청 단위 LangSmith 설정 전달
        text = chain.invoke(inputs, config=(run_config or {}))

    llm_out = _parse_llm_output(text)

    title = (llm_out.title if llm_out and llm_out.title else (req.title or "Untitled"))
    body_md = (llm_out.body_md if llm_out and llm_out.body_md else (req.content or ""))
    tldr_list = (llm_out.tldr if llm_out and llm_out.tldr else _fallback_tldr_from_text(body_md, k=3))
    tags = (llm_out.tags if llm_out and llm_out.tags else (req.tags or []))
    sources = (llm_out.sources if llm_out and llm_out.sources else [])
    hero = (llm_out.hero_image_url if llm_out and llm_out.hero_image_url else None)
    author = (llm_out.author_name if llm_out and llm_out.author_name else getattr(req, "author_name", None))
    category1 = (req.categories[0] if req.categories else None)

    post = NewsPost(
        post_id=str(uuid4()),
        article_id=req.article_id,
        article_code=getattr(req, "article_code", None),
        url=req.url,
        title=title,
        dek=(llm_out.subtitle if llm_out and llm_out.subtitle else None),
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
            "parsed_ok": bool(llm_out is not None),
            "service": getattr(settings.news, "service_name", "redfin_news"),
            "ls_project": getattr(settings.news, "langsmith_project", None),
        },
    )
    post.model_meta["urls"] = {"api": f"/redfin_news/posts/{post.post_id}"}

    col = get_news_collection()
    if col is not None:
        try:
            col.insert_one(post.dict())
        except Exception:
            pass

    # 출간 직후 자동 인덱싱
    try:
        if getattr(settings.news, "index_on_publish", False):
            upsert_news_post_to_chroma(post)
    except Exception:
        pass

    return post


# -------------------- NewsLoader 기반 배치 출간 --------------------

def _req_from_loader_doc(doc) -> NewsPublishRequest:
    md = dict(doc.metadata or {})
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
    *,
    run_config: Optional[Dict[str, Any]] = None,   # ⬅️ 추가
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

            # 중복 체크
            q = _dedupe_key({
                "article_code": req.article_code,
                "article_id": req.article_id,
                "url": req.url,
            }) or ({"article_code": str(md.get("doc_id"))} if md.get("doc_id") else {})
            if q and _exists_in_news(q):
                skipped.append({"index": idx, "reason": "duplicate", "key": q})
                continue

            post = generate_news_post(req, feed_meta=md, run_config=run_config)  # ⬅️ 전달
            created.append(post.dict())
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
    return {"created": created, "skipped": skipped, "errors": errors}


# -------------------- 기존 JSON 배열 배치(옵션으로 유지) --------------------

def publish_batch(
    items: List[Dict[str, Any]],
    field_map: Optional[Dict[str, str]] = None,
    default_publish: bool = True,
    default_top_k: int = 6,
    *,
    run_config: Optional[Dict[str, Any]] = None,   # ⬅️ 추가
) -> Dict[str, Any]:
    created, skipped, errors = [], [], []
    m = field_map or {}
    for idx, it in enumerate(items):
        try:
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

            post = generate_news_post(req, feed_meta=it, run_config=run_config)  # ⬅️ 전달
            created.append(post.dict())
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
    return {"created": created, "skipped": skipped, "errors": errors}


# -------------------- 외부 호출 진입점 --------------------

def publish_from_feed(
    feed_url: str,
    item_path: Optional[str] = None,      # 보존(미사용)
    field_map: Optional[Dict[str, str]] = None,
    default_publish: bool = True,
    default_top_k: int = 6,
    *,
    run_config: Optional[Dict[str, Any]] = None,   # ⬅️ 추가
) -> Dict[str, Any]:
    return publish_from_loader(
        feed_url=feed_url,
        field_map=field_map,
        default_publish=default_publish,
        default_top_k=default_top_k,
        timeout=15,
        run_config=run_config,             # ⬅️ 전달
    )


def publish_from_env(*, run_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # ⬅️ 시그니처 변경
    # 1) API URL
    feed_url = settings.news.api_url or os.getenv("NEWS_API_URL")
    if not feed_url:
        raise ValueError("NEWS_API_URL/settings.news.api_url is empty")

    # 2) 필드 매핑
    user_map = getattr(settings.news, "feed_field_map", None)
    if user_map is None:
        field_map_json = os.getenv("NEWS_FEED_FIELD_MAP")
        user_map = json.loads(field_map_json) if field_map_json else None
    field_map = {**DEFAULT_FIELD_MAP, **user_map} if user_map else DEFAULT_FIELD_MAP

    # 3) 퍼블리시/탑K
    default_publish = _as_bool(os.getenv("NEWS_DEFAULT_PUBLISH", None),
                               default=_as_bool(getattr(settings.news, "default_publish", True), True))
    default_top_k = _as_int(os.getenv("NEWS_TOP_K", None),
                            default=_as_int(getattr(settings.news, "top_k", 6), 6))

    # 4) 실행
    return publish_from_loader(
        feed_url=feed_url,
        field_map=field_map,
        default_publish=default_publish,
        default_top_k=default_top_k,
        timeout=15,
        run_config=run_config,             # ⬅️ 전달
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


# ------------------- 고정크기 청킹 + 오버랩 인덱스 초기화 -------------------

def init_news_index_fixed(
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
) -> dict:
    try:
        if not settings.news.api_url:
            return {"ok": False, "reason": "settings.news.api_url is empty"}

        loader = NewsLoader(field_map=DEFAULT_FIELD_MAP)
        docs = loader.load_news_json(settings.news.api_url, timeout=15)
        if not docs:
            return {"ok": False, "reason": "no documents loaded", "api_url": settings.news.api_url}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(model_name=settings.news.emb_model)

        vs = create_chroma_store(
            embedding=embedding,
            collection_name=settings.news.collection,
            persist_dir=settings.news.persist_dir,
            distance="cosine",
        )
        vs.vs.add_documents(chunks)
        vs.persist()

        return {
            "ok": True,
            "mode": "fixed",
            "api_url": settings.news.api_url,
            "collection": settings.news.collection,
            "persist_dir": settings.news.persist_dir,
            "docs": len(docs),
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
