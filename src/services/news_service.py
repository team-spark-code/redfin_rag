# # -*- coding: utf-8 -*- 2025-09-04 10:05
# """
# news_service.py
# - Fix: body_md에 LLM JSON 전체가 문자열로 저장되는 문제 해결
# - Change: LLM 출력 파싱 강화(펜스된 ```json 블록 우선), 실패 시에도 body_md만 추출
# - Change: publish_from_source 가 NewsPost 스키마로 일관 저장(tldr 포함)
# - Cleanup: 중복/불필요 import 제거, 가드 간소화
# (2025-09-04)
# """
# from __future__ import annotations

# import json
# import os
# import re
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict, List, Optional
# from uuid import uuid4

# from pymongo import MongoClient, UpdateOne

# from core.settings import settings
# from nureongi.loaders import NewsLoader, DEFAULT_FIELD_MAP
# from nureongi.prompt_loader import load_md_template, render_template
# from nureongi.vectorstore import create_chroma_store
# from nureongi.vs_news import upsert_news_post_to_chroma
# from observability.mongo_logger import get_news_collection
# from schemas.news import NewsPublishRequest, NewsPost
# from schemas.news_llm import NewsLLMOut

# # LLM/체인
# from nureongi.llm import build_default_llm
# from nureongi.news_chain import build_news_chain

# # Optional chunkers
# try:
#     from langchain_huggingface import HuggingFaceEmbeddings
# except Exception:
#     HuggingFaceEmbeddings = None  # type: ignore

# # Text splitters(버전 대응)
# try:
#     from langchain_text_splitters import RecursiveCharacterTextSplitter
# except Exception:  # pragma: no cover
#     from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore


# # --------------------------------------
# # 작은 유틸
# # --------------------------------------

# def _as_bool(v: object | None, default: bool = False) -> bool:
#     if isinstance(v, bool):
#         return v
#     if v is None:
#         return default
#     return str(v).strip().lower() in {"1", "true", "yes", "on"}


# def _as_int(v: object | None, default: int = 0) -> int:
#     try:
#         return int(str(v).strip()) if v is not None and str(v).strip() != "" else default
#     except Exception:
#         return default


# def _coerce_text(x: Any) -> str:
#     """LLM/체인 반환값을 문자열로 정규화."""
#     try:
#         if hasattr(x, "content"):
#             return str(getattr(x, "content"))
#         if isinstance(x, (dict, list)):
#             return json.dumps(x, ensure_ascii=False)
#         return str(x)
#     except Exception:
#         return str(x)


# _JSON_FENCE_RE = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


# def _safe_json_block(s: str) -> Dict[str, Any]:
#     """문자열에서 JSON 블록만 뽑아 파싱.
#     1) ```json ... ``` 펜스 우선
#     2) 전체 문자열에서 가장 바깥 { ... } 범위 추정
#     실패 시 ValueError 발생
#     """
#     s = (s or "").strip()
#     m = _JSON_FENCE_RE.search(s)
#     if m:
#         return json.loads(m.group(1))
#     i, j = s.find("{"), s.rfind("}")
#     if i != -1 and j != -1 and j > i:
#         return json.loads(s[i : j + 1])
#     # 명백한 JSON 형식이 아닐 때는 실패
#     raise ValueError("no json block")


# def _parse_llm_output_any(x: Any) -> Optional[NewsLLMOut]:
#     """LLM 출력(any)을 NewsLLMOut로 파싱. 실패 시 None."""
#     try:
#         data = x if isinstance(x, dict) else _safe_json_block(_coerce_text(x))
#     except Exception:
#         return None
#     try:
#         # pydantic v2 우선
#         if hasattr(NewsLLMOut, "model_validate"):
#             return NewsLLMOut.model_validate(data)  # type: ignore[attr-defined]
#         return NewsLLMOut.parse_obj(data)  # type: ignore[attr-defined]
#     except Exception:
#         return None


# def _listify(v: Any) -> List[str]:
#     if v is None:
#         return []
#     if isinstance(v, list):
#         return [str(x) for x in v if x is not None]
#     return [str(v)]


# def _dedupe_key(item: Dict[str, Any]) -> Dict[str, Any]:
#     q: Dict[str, Any] = {}
#     if item.get("article_code"):
#         q["article_code"] = str(item["article_code"])  # 우선순위 높음
#     elif item.get("article_id"):
#         q["article_id"] = str(item["article_id"])
#     elif item.get("url"):
#         q["url"] = str(item["url"])
#     return q


# def _exists_in_news(q: Dict[str, Any]) -> bool:
#     if not q:
#         return False
#     col = get_news_collection()
#     return bool(col and col.find_one(q))


# def _fallback_tldr_from_text(text: str, k: int = 3) -> List[str]:
#     sents = re.split(r"(?<=[.!?。！？])\s+", (text or "").strip())
#     out: List[str] = []
#     for s in sents:
#         s = s.strip()
#         if not s:
#             continue
#         out.append(s[:120])
#         if len(out) >= max(1, k):
#             break
#     return out


# # --------------------------------------
# # 인덱스 초기화(요약)
# # --------------------------------------

# def init_index() -> dict:
#     try:
#         vs = create_chroma_store(
#             collection_name=settings.news.collection,
#             persist_dir=settings.news.persist_dir,
#             distance="cosine",
#         )
#         vs.persist()
#         return {"ok": True, "collection": settings.news.collection, "persist_dir": settings.news.persist_dir}
#     except Exception as e:
#         return {"ok": False, "error": str(e)}


# def init_news_index_fixed(*, chunk_size: int = 1200, chunk_overlap: int = 120) -> dict:
#     try:
#         if not settings.news.api_url:
#             return {"ok": False, "reason": "settings.news.api_url is empty"}
#         loader = NewsLoader(field_map=DEFAULT_FIELD_MAP)
#         docs = loader.load_news_json(settings.news.api_url, timeout=15)
#         if not docs:
#             return {"ok": False, "reason": "no documents loaded", "api_url": settings.news.api_url}
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
#         )
#         chunks = splitter.split_documents(docs)
#         if HuggingFaceEmbeddings is None:
#             return {"ok": False, "reason": "huggingface embeddings unavailable"}
#         embedding = HuggingFaceEmbeddings(model_name=settings.news.emb_model)
#         vs = create_chroma_store(
#             embedding=embedding,
#             collection_name=settings.news.collection,
#             persist_dir=settings.news.persist_dir,
#             distance="cosine",
#         )
#         vs.vs.add_documents(chunks)
#         vs.persist()
#         return {
#             "ok": True,
#             "mode": "fixed",
#             "api_url": settings.news.api_url,
#             "collection": settings.news.collection,
#             "persist_dir": settings.news.persist_dir,
#             "docs": len(docs),
#             "chunks": len(chunks),
#             "chunk_size": chunk_size,
#             "chunk_overlap": chunk_overlap,
#         }
#     except Exception as e:
#         return {"ok": False, "error": str(e)}


# # --------------------------------------
# # 단건 생성(핵심)
# # --------------------------------------

# def generate_news_post(
#     req: NewsPublishRequest,
#     feed_meta: Optional[Dict[str, Any]] = None,
#     *,
#     run_config: Optional[Dict[str, Any]] = None,
# ) -> NewsPost:
#     """LLM을 사용해 뉴스 포스트를 생성(혹은 패스스루)하고, NewsPost 스키마로 저장.
#     - body_md에 JSON 전체가 들어가지 않도록 강제 정규화
#     - tldr는 별도 리스트 필드로 저장
#     """
#     # 1) 템플릿 로딩/렌더링(체인 내부 사용용)
#     tpl_path = settings.news.prompt_path or "src/prompts/templates/news_publish_v1.md"
#     tpl = load_md_template(tpl_path)

#     # 2) LLM 실행(or 미사용)
#     llm_out: Optional[NewsLLMOut] = None
#     raw_text: str = ""
#     if settings.news.use_llm:
#         llm = build_default_llm()
#         chain = build_news_chain(llm, tpl_path)
#         inputs = {
#             "title": (req.title or ""),
#             "url": (req.url or ""),
#             "content": (req.content or ""),
#             "categories": ", ".join(req.categories or []),
#             "tags": ", ".join(req.tags or []),
#             "meta": json.dumps(feed_meta or {}, ensure_ascii=False),
#         }
#         try:
#             raw_text = _coerce_text(chain.invoke(inputs, config=(run_config or {})))
#             llm_out = _parse_llm_output_any(raw_text)
#         except Exception:
#             llm_out = None
#             raw_text = _coerce_text(raw_text)
#     else:
#         # LLM 미사용: 그대로 패스스루
#         raw_text = req.content or ""

#     # 3) 결과 정규화: 반드시 body_md(마크다운), tldr(리스트)만 저장
#     if llm_out:
#         title = llm_out.title or (req.title or "Untitled")
#         body_md = llm_out.body_md or (req.content or "")
#         tldr_list = llm_out.tldr or _fallback_tldr_from_text(body_md, k=3)
#         tags = llm_out.tags or (req.tags or [])
#         sources = llm_out.sources or []
#         hero = llm_out.hero_image_url or None
#         subtitle = llm_out.subtitle or None
#         author = llm_out.author_name or getattr(req, "author_name", None)
#     else:
#         # JSON 같아 보이면 body_md로 쓰지 않음(파싱 실패한 JSON 전체가 저장되는 걸 방지)
#         looks_like_json = raw_text.strip().startswith("{") and raw_text.strip().endswith("}")
#         if looks_like_json:
#             try:
#                 data = _safe_json_block(raw_text)
#                 body_md = str(data.get("body_md", "")) or (req.content or "")
#                 tldr_list = _listify(data.get("tldr")) or _fallback_tldr_from_text(body_md, 3)
#             except Exception:
#                 body_md = req.content or ""
#                 tldr_list = _fallback_tldr_from_text(body_md, 3)
#         else:
#             body_md = raw_text or (req.content or "")
#             tldr_list = _fallback_tldr_from_text(body_md, 3)
#         title = req.title or "Untitled"
#         tags, sources, hero, subtitle = (req.tags or []), [], None, None
#         author = getattr(req, "author_name", None)

#     category1 = (req.categories[0] if req.categories else None)

#     post = NewsPost(
#         post_id=str(uuid4()),
#         article_id=req.article_id,
#         article_code=getattr(req, "article_code", None),
#         url=req.url,
#         title=title,
#         dek=subtitle,
#         tldr=tldr_list,
#         body_md=body_md,
#         hero_image_url=hero,
#         author_name=author,
#         category=category1,
#         categories=req.categories or [],
#         tags=tags,
#         sources=sources,
#         status="published" if req.publish else "draft",
#         model_meta={
#             "persona": "news_brief",
#             "strategy": "auto",
#             "top_k": req.top_k,
#             "feed": feed_meta or {},
#             "parsed_ok": bool(llm_out is not None),
#             "service": getattr(settings.news, "service_name", "redfin_news"),
#             "ls_project": getattr(settings.news, "langsmith_project", None),
#         },
#     )
#     post.model_meta["urls"] = {"api": f"/redfin_news/posts/{post.post_id}"}

#     col = get_news_collection()
#     if col is not None:
#         try:
#             col.insert_one(post.dict())  # ✅ body_md/tldr가 개별 필드로 저장됨
#         except Exception:
#             pass

#     # 출간 직후 벡터 인덱싱(옵션)
#     try:
#         if getattr(settings.news, "index_on_publish", False):
#             upsert_news_post_to_chroma(post)
#     except Exception:
#         pass

#     return post


# # --------------------------------------
# # 배치 출간(로더 기반)
# # --------------------------------------

# def _req_from_loader_doc(doc) -> NewsPublishRequest:
#     md = dict(doc.metadata or {})
#     content = doc.page_content or ""
#     categories: List[str] = []
#     if md.get("content_type"):
#         categories = [str(md["content_type"])]
#     elif md.get("source"):
#         categories = [str(md["source"])]
#     return NewsPublishRequest(
#         article_id=str(md.get("guid")) if md.get("guid") else None,
#         article_code=str(md.get("guid") or md.get("doc_id")) if (md.get("guid") or md.get("doc_id")) else None,
#         url=md.get("url"),
#         title=md.get("title"),
#         content=content,
#         categories=categories,
#         tags=[str(t) for t in (md.get("tags") or [])],
#         publish=True,
#         top_k=6,
#     )


# def publish_from_loader(
#     feed_url: str,
#     field_map: Optional[Dict[str, str]] = None,
#     default_publish: bool = True,
#     default_top_k: int = 6,
#     *,
#     run_config: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     loader = NewsLoader(field_map=field_map or DEFAULT_FIELD_MAP)
#     docs = loader.load_news_json(feed_url, timeout=15)

#     created, skipped, errors = [], [], []
#     for idx, doc in enumerate(docs):
#         try:
#             md = dict(doc.metadata or {})
#             req = _req_from_loader_doc(doc)
#             if default_publish is not None:
#                 req.publish = bool(default_publish)
#             if default_top_k is not None:
#                 req.top_k = _as_int(default_top_k, 6)
#             q = _dedupe_key({
#                 "article_code": req.article_code,
#                 "article_id": req.article_id,
#                 "url": req.url,
#             }) or ({"article_code": str(md.get("doc_id"))} if md.get("doc_id") else {})
#             if q and _exists_in_news(q):
#                 skipped.append({"index": idx, "reason": "duplicate", "key": q})
#                 continue
#             post = generate_news_post(req, feed_meta=md, run_config=run_config)
#             created.append(post.dict())
#         except Exception as e:
#             errors.append({"index": idx, "error": str(e)})
#     return {"created": created, "skipped": skipped, "errors": errors}


# # --------------------------------------
# # 기존 JSON 배열 입력(레거시 유지)
# # --------------------------------------

# def publish_batch(
#     items: List[Dict[str, Any]],
#     field_map: Optional[Dict[str, str]] = None,
#     default_publish: bool = True,
#     default_top_k: int = 6,
#     *,
#     run_config: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     created, skipped, errors = [], [], []
#     m = field_map or {}
#     for idx, it in enumerate(items):
#         try:
#             article_code = str(it.get(m.get("article_code", "article_code")) or it.get("guid") or it.get("id") or "")
#             article_id = str(it.get(m.get("article_id", "article_id")) or it.get("guid") or "")
#             url = it.get(m.get("url", "url")) or it.get("link")
#             title = it.get(m.get("title", "title"))

#             # content가 JSON(문자열/객체)일 때 body_md만 추출
#             raw_content = it.get(m.get("content", "content")) or it.get("article_text")
#             content = None
#             try:
#                 data = raw_content if isinstance(raw_content, (dict, list)) else _safe_json_block(str(raw_content))
#                 if isinstance(data, dict) and "body_md" in data:
#                     content = data.get("body_md")
#             except Exception:
#                 pass
#             if content is None:
#                 content = raw_content or ""

#             categories = it.get(m.get("categories", "categories")) or it.get("content_type")
#             tags = it.get(m.get("tags", "tags")) or []

#             req = NewsPublishRequest(
#                 article_id=article_id or None,
#                 article_code=article_code or None,
#                 url=url,
#                 title=title,
#                 content=content,
#                 categories=[categories] if isinstance(categories, str) else (categories or []),
#                 tags=[str(t) for t in (tags if isinstance(tags, list) else [tags] if tags else [])],
#                 publish=bool(it.get(m.get("publish", "publish"), default_publish)),
#                 top_k=_as_int(it.get(m.get("top_k", "top_k"), default_top_k), 6),
#             )

#             q = _dedupe_key({"article_code": req.article_code, "article_id": req.article_id, "url": req.url})
#             if q and _exists_in_news(q):
#                 skipped.append({"index": idx, "reason": "duplicate", "key": q})
#                 continue

#             post = generate_news_post(req, feed_meta=it, run_config=run_config)
#             created.append(post.dict())
#         except Exception as e:
#             errors.append({"index": idx, "error": str(e)})
#     return {"created": created, "skipped": skipped, "errors": errors}


# # --------------------------------------
# # Mongo → 요약/출간(일관 스키마)
# # --------------------------------------

# def _get_llm_for_news():
#     if not getattr(settings.news, "use_llm", True):
#         return None
#     if os.getenv("GOOGLE_API_KEY"):
#         from langchain_google_genai import ChatGoogleGenerativeAI
#         return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), temperature=0)
#     if os.getenv("OPENAI_API_KEY"):
#         from langchain_openai import ChatOpenAI
#         return ChatOpenAI(model=os.getenv("MODEL_LLM", "gpt-4o-mini"), temperature=0)
#     return None


# def _render_body_md(raw_item: Dict[str, Any], llm, prompt_path: str) -> str:
#     title = raw_item.get("title") or ""
#     content = raw_item.get("article_text") or raw_item.get("content") or ""
#     if not llm:
#         return f"# {title}\n\n{content[:1000]}"

#     try:
#         tpl = Path(prompt_path).read_text(encoding="utf-8") if Path(prompt_path).exists() else ""
#     except Exception:
#         tpl = ""

#     system = "당신은 기자입니다. 핵심만 간결히 요약하고, 소제목과 불릿을 사용합니다."
#     user = f"{tpl}\n\n[제목]\n{title}\n\n[본문]\n{content}\n\n위 내용을 간결한 기사 요약(Markdown)으로 작성해 주세요."
#     try:
#         resp = llm.invoke([("system", system), ("user", user)])
#         text = getattr(resp, "content", None) or str(resp)
#         return text.strip()
#     except Exception as e:  # pragma: no cover (네트워크/모델 이슈)
#         return f"# {title}\n\n{content[:1000]}\n\n<!-- LLM 실패: {e} -->"


# def publish_from_source(*, mongo_query: Optional[Dict[str, Any]] = None, limit: int = 30) -> Dict[str, Any]:
#     """Mongo redfin.extract → 요약 → redfin.news_logs(NewsPost 스키마) upsert.
#     - ✨ body_md/tldr가 개별 필드로 저장됨(JSON 문자열 금지)
#     """
#     loader = NewsLoader()
#     docs = loader.load_news_mongo(
#         uri=settings.mongo.uri,
#         db=settings.mongo.db,
#         collection=getattr(settings.news, "source_collection", "extract"),
#         query=mongo_query or {},
#         limit=limit,
#         timeout_ms=getattr(settings.mongo, "timeout_ms", 3000),
#         sort=[("processed_at", -1), ("published", -1)],
#     )
#     if not docs:
#         return {"ok": False, "reason": "no input docs from mongo.extract"}
#     # 🔧 FIX: NameError 방지 — 템플릿 경로 정의
#     tpl_path = getattr(settings.news, "prompt_path", "src/prompts/templates/news_publish_v1.md")

#     llm = _get_llm_for_news()
#     client = MongoClient(settings.mongo.uri, serverSelectionTimeoutMS=settings.mongo.timeout_ms)
#     coll = client[settings.mongo.db][getattr(settings.mongo, "news_collection", "news_logs")]

#     ops: List[UpdateOne] = []
#     count = 0
#     for d in docs:
#         md = d.metadata or {}
#         raw = {
#             "title": md.get("title"),
#             "article_text": d.page_content,  # 원문
#             "source": md.get("source"),
#             "link": md.get("url") or md.get("link"),
#             "guid": md.get("guid"),
#             "published_at": md.get("published"),
#             "tags": md.get("tags") or [],
#             "authors": md.get("authors") or [],
#             "language": md.get("lang"),
#         }
#         body_md = _render_body_md(raw, llm, prompt_path=tpl_path)
#         tldr_list = _fallback_tldr_from_text(body_md, 3)

#         # 일관 스키마로 upsert
#         doc = {
#             "type": "post",
#             "title": raw["title"],
#             "dek": None,
#             "tldr": tldr_list,
#             "body_md": body_md,
#             "hero_image_url": None,
#             "author_name": None,
#             "category": None,
#             "categories": [],
#             "tags": raw["tags"],
#             "sources": [],
#             "status": "published",
#             "model_meta": {
#                 "persona": "news_brief",
#                 "strategy": "auto",
#                 "feed": {k: v for k, v in md.items() if k not in {"article_text"}},
#                 "service": getattr(settings.news, "service_name", "redfin_news"),
#                 "ls_project": getattr(settings.news, "langsmith_project", None),
#             },
#             "source": raw["source"],
#             "link": raw["link"],
#             "guid": raw["guid"],
#             "published_at": md.get("published") or md.get("published_at"),
#             "lang": raw["language"],
#             "created_at": datetime.utcnow(),
#             "updated_at": datetime.utcnow(),
#         }
#         filt = {"$or": [{"guid": raw["guid"]}, {"link": raw["link"]}]}
#         ops.append(UpdateOne(filt, {"$set": doc}, upsert=True))
#         count += 1

#     if not ops:
#         return {"ok": True, "count": 0, "upserts": 0, "modified": 0}

#     res = coll.bulk_write(ops, ordered=False)
#     upserts = len(res.upserted_ids or {})
#     modified = res.modified_count
#     return {"ok": True, "count": count, "upserts": upserts, "modified": modified}


# # --------------------------------------
# # 외부 진입점(라우터에서 사용)
# # --------------------------------------

# def publish_from_feed(
#     feed_url: str,
#     item_path: Optional[str] = None,  # 보존(미사용)
#     field_map: Optional[Dict[str, str]] = None,
#     default_publish: bool = True,
#     default_top_k: int = 6,
#     *,
#     run_config: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     return publish_from_loader(
#         feed_url=feed_url,
#         field_map=field_map,
#         default_publish=default_publish,
#         default_top_k=default_top_k,
#         run_config=run_config,
#     )


# def publish_from_env(*, run_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     feed_url = settings.news.api_url or os.getenv("NEWS_API_URL")
#     if not feed_url:
#         raise ValueError("NEWS_API_URL/settings.news.api_url is empty")

#     user_map = getattr(settings.news, "feed_field_map", None)
#     if user_map is None:
#         field_map_json = os.getenv("NEWS_FEED_FIELD_MAP")
#         user_map = json.loads(field_map_json) if field_map_json else None
#     field_map = {**DEFAULT_FIELD_MAP, **user_map} if user_map else DEFAULT_FIELD_MAP

#     default_publish = _as_bool(os.getenv("NEWS_DEFAULT_PUBLISH"),
#                                default=_as_bool(getattr(settings.news, "default_publish", True), True))
#     default_top_k = _as_int(os.getenv("NEWS_TOP_K"),
#                             default=_as_int(getattr(settings.news, "top_k", 6), 6))

#     return publish_from_loader(
#         feed_url=feed_url,
#         field_map=field_map,
#         default_publish=default_publish,
#         default_top_k=default_top_k,
#         run_config=run_config,
#     )


# # -*- coding: utf-8 -*-
# # =============================================================================
# # 변경 요약 (2025-09-04)
# # 1) [CHG] 파서 강화:
# #    - _coerce_text / _safe_json_block / _parse_llm_output_any 추가
# #    - 문자열/AIMessage/dict 어떤 결과라도 일관되게 파싱
# # 2) [CHG] generate_news_post():
# #    - LLM 미사용시 불필요한 JSON 합성 제거
# #    - 체인 결과를 원시 텍스트화 → JSON이면 NewsLLMOut로 파싱
# #    - 파싱 실패 시에도 body_md만 안전 추출(원문/마크다운 유지)
# # 3) [CHG] publish_batch():
# #    - 입력 content가 JSON(문자열/객체)일 때 body_md만 뽑아 사용
# # 4) 기능/기존 로직은 유지(중복판정, 인덱싱, 컬렉션 저장 등)
# # =============================================================================
# from __future__ import annotations
# import os
# import json
# import urllib.request
# import re
# from uuid import uuid4
# from typing import Any, Dict, Optional, List
# from datetime import datetime
# from pymongo import MongoClient, UpdateOne

# from schemas.news import NewsPublishRequest, NewsPost
# from schemas.news_llm import NewsLLMOut   # [CHG] 관대한 스키마로 교체됨
# from observability.mongo_logger import get_news_collection
# from nureongi.loaders import NewsLoader, DEFAULT_FIELD_MAP
# from nureongi.vs_news import upsert_news_post_to_chroma
# from nureongi.vectorstore import create_chroma_store

# # 독립 LLM/체인
# from nureongi.llm import build_default_llm
# from nureongi.news_chain import build_news_chain

# # 템플릿/세팅
# from core.settings import settings
# from nureongi.prompt_loader import load_md_template, render_template

# # ===== (시맨틱 인덱스용) 임베딩/청킹 =====
# try:
#     from langchain_huggingface import HuggingFaceEmbeddings
#     from langchain_experimental.text_splitter import SemanticChunker
#     from langchain_core.documents import Document  # 타입 힌트 용도
# except Exception:
#     HuggingFaceEmbeddings = None
#     SemanticChunker = None
#     Document = None

# # RecursiveCharacterTextSplitter 임포트 (버전에 따른 폴백 처리)
# try:
#     from langchain_text_splitters import RecursiveCharacterTextSplitter
# except Exception:
#     from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_huggingface import HuggingFaceEmbeddings  # 임베딩 필요


# # ------------------- 유틸/형변환 -------------------

# def _as_bool(v: object | None, default: bool = False) -> bool:
#     if isinstance(v, bool):
#         return v
#     if v is None:
#         return default
#     return str(v).strip().lower() in ("1", "true", "yes", "on")

# def _as_int(v: object | None, default: int = 0) -> int:
#     try:
#         if v is None:
#             return default
#         s = str(v).strip()
#         if s == "":
#             return default
#         return int(s)
#     except Exception:
#         return default

# # [CHG] LLM/체인 반환을 항상 str로 정규화
# def _coerce_text(x: Any) -> str:
#     try:
#         if hasattr(x, "content"):
#             return str(getattr(x, "content"))
#         if isinstance(x, (dict, list)):
#             return json.dumps(x, ensure_ascii=False)
#         return str(x)
#     except Exception:
#         return str(x)

# # [CHG] 문자열에서 JSON 블록만 잘라 파싱
# def _safe_json_block(s: str) -> Dict[str, Any]:
#     s = (s or "").strip()
#     i, j = s.find("{"), s.rfind("}")
#     if i != -1 and j != -1 and j > i:
#         s = s[i : j + 1]
#     return json.loads(s)

# # [CHG] 어떤 결과든 NewsLLMOut로 파싱 시도
# def _parse_llm_output_any(x: Any) -> Optional[NewsLLMOut]:
#     try:
#         if isinstance(x, dict):
#             data = x
#         else:
#             s = _coerce_text(x)
#             data = _safe_json_block(s)
#     except Exception:
#         return None

#     try:
#         if hasattr(NewsLLMOut, "model_validate"):   # pydantic v2
#             return NewsLLMOut.model_validate(data)
#         return NewsLLMOut.parse_obj(data)           # pydantic v1
#     except Exception:
#         return None

# def _listify(v: Any) -> List[str]:
#     if v is None:
#         return []
#     if isinstance(v, list):
#         return [str(x) for x in v if x is not None]
#     return [str(v)]

# def _dedupe_key(item: Dict[str, Any]) -> Dict[str, Any]:
#     q: Dict[str, Any] = {}
#     if item.get("article_code"):
#         q["article_code"] = str(item["article_code"])
#     elif item.get("article_id"):
#         q["article_id"] = str(item["article_id"])
#     elif item.get("url"):
#         q["url"] = str(item["url"])
#     return q

# def _exists_in_news(q: Dict[str, Any]) -> bool:
#     if not q:
#         return False
#     col = get_news_collection()
#     if col is None:
#         return False
#     return col.find_one(q) is not None


# # -------------------- 초기화 함수(수명주기에서 호출) --------------------

# def init_index() -> dict:
#     """뉴스용 Chroma 컬렉션 생성/열기."""
#     try:
#         vs = create_chroma_store(
#             collection_name=settings.news.collection,
#             persist_dir=settings.news.persist_dir,
#             distance="cosine",
#         )
#         vs.persist()
#         return {
#             "ok": True,
#             "collection": settings.news.collection,
#             "persist_dir": settings.news.persist_dir,
#         }
#     except Exception as e:
#         return {"ok": False, "error": str(e)}

# def init_news_index_semantic() -> dict:
#     """
#     뉴스 전용 '시맨틱 인덱스' 초기화:
#     - settings.news.api_url에서 기사 로드 → 시맨틱 청킹 → Chroma 업서트
#     """
#     try:
#         if not settings.news.api_url:
#             return {"ok": False, "reason": "settings.news.api_url is empty"}
#         if HuggingFaceEmbeddings is None or SemanticChunker is None:
#             return {"ok": False, "reason": "semantic stack unavailable (langchain_experimental / huggingface not installed)"}

#         loader = NewsLoader(field_map=DEFAULT_FIELD_MAP)
#         docs = loader.load_news_json(settings.news.api_url, timeout=15)
#         if not docs:
#             return {"ok": False, "reason": "no documents loaded", "api_url": settings.news.api_url}

#         embed_for_chunk = HuggingFaceEmbeddings(model_name=settings.news.emb_model)
#         splitter = SemanticChunker(embeddings=embed_for_chunk, breakpoint_threshold_type="percentile")
#         chunks = splitter.split_documents(docs)

#         vs = create_chroma_store(
#             collection_name=settings.news.collection,
#             persist_dir=settings.news.persist_dir,
#             distance="cosine",
#         )
#         vs.vs.add_documents(chunks)
#         vs.persist()

#         return {
#             "ok": True,
#             "api_url": settings.news.api_url,
#             "collection": settings.news.collection,
#             "persist_dir": settings.news.persist_dir,
#             "docs": len(docs),
#             "chunks": len(chunks),
#         }
#     except Exception as e:
#         return {"ok": False, "error": str(e)}


# # -------------------- 단건 생성 --------------------

# def generate_news_post(
#     req: NewsPublishRequest,
#     feed_meta: Optional[Dict[str, Any]] = None,
#     *,
#     run_config: Optional[Dict[str, Any]] = None,   # 요청 단위 트레이스 설정
# ) -> NewsPost:
#     """
#     .md 템플릿을 로드하여 LLM에 전달.
#     - retriever 없음(뉴스 단건 요약/가공)
#     - 라우터에서 받은 run_config를 체인 invoke에 그대로 전달
#     """
#     # 1) 템플릿 로딩/렌더링
#     tpl = load_md_template(settings.news.prompt_path or "src/prompts/templates/news_publish_v1.md")

#     # 안전장치: 인사이트 템플릿 섞임 차단
#     _tpl_str = str(tpl)
#     if ("Issue List" in _tpl_str) or ("target-insight" in _tpl_str):
#         raise RuntimeError("뉴스 출간에서 인사이트 템플릿이 감지됨. settings.news.prompt_path를 확인하세요.")

#     prompt = render_template(
#         tpl,
#         title=(req.title or ""),
#         url=(req.url or ""),
#         content=(req.content or ""),
#         categories=", ".join(req.categories or []),
#         tags=", ".join(req.tags or []),
#         meta=json.dumps(feed_meta or {}, ensure_ascii=False),
#     )

#     # 2) 요약/가공
#     # [CHG] LLM 미사용시 불필요한 JSON 합성 제거
#     if not settings.news.use_llm:
#         raw_text = req.content or ""
#         llm_out = None
#     else:
#         llm = build_default_llm()
#         chain = build_news_chain(llm, settings.news.prompt_path or "src/prompts/templates/news_publish_v1.md")
#         inputs = {
#             "title": (req.title or ""),
#             "url": (req.url or ""),
#             "content": (req.content or ""),
#             "categories": ", ".join(req.categories or []),
#             "tags": ", ".join(req.tags or []),
#             "meta": json.dumps(feed_meta or {}, ensure_ascii=False),
#         }

#         if run_config is None or not isinstance(run_config, dict) or not run_config.get("callbacks"):
#             print("[warn] news run_config.callbacks missing; LangSmith project may fallback to env")
#         else:
#             try:
#                 cb = run_config["callbacks"][0]
#                 print("[diag] news callback tracer =", type(cb).__name__, getattr(cb, "project_name", None))
#             except Exception:
#                 pass

#         # [CHG] 결과를 먼저 원시 텍스트로 정규화 → JSON 파싱 시도
#         raw_text = _coerce_text(chain.invoke(inputs, config=(run_config or {})))
#         llm_out = _parse_llm_output_any(raw_text)

#     # [CHG] 최종 필드 결정: JSON 구조가 있으면 우선, 없으면 raw_text/req.content를 마크다운 본문으로 사용
#     if llm_out:
#         title = llm_out.title or (req.title or "Untitled")
#         body_md = llm_out.body_md or (req.content or "")
#         tldr_list = llm_out.tldr or _fallback_tldr_from_text(body_md, k=3)
#         tags = llm_out.tags or (req.tags or [])
#         sources = llm_out.sources or []
#         hero = llm_out.hero_image_url or None
#         subtitle = llm_out.subtitle or None
#         author = llm_out.author_name or getattr(req, "author_name", None)
#     else:
#         title = req.title or "Untitled"
#         body_md = req.content or ""
#         # raw_text가 JSON이면 body_md만 추출 시도
#         try:
#             maybe = _safe_json_block(raw_text)
#             if isinstance(maybe, dict) and "body_md" in maybe:
#                 body_md = str(maybe["body_md"] or body_md)
#         except Exception:
#             if raw_text.strip():
#                 body_md = raw_text
#         tldr_list = _fallback_tldr_from_text(body_md, k=3)
#         tags, sources, hero, subtitle = (req.tags or []), [], None, None
#         author = getattr(req, "author_name", None)

#     category1 = (req.categories[0] if req.categories else None)

#     post = NewsPost(
#         post_id=str(uuid4()),
#         article_id=req.article_id,
#         article_code=getattr(req, "article_code", None),
#         url=req.url,
#         title=title,
#         dek=subtitle,
#         tldr=tldr_list,
#         body_md=body_md,
#         hero_image_url=hero,
#         author_name=author,
#         category=category1,
#         categories=req.categories or [],
#         tags=tags,
#         sources=sources,
#         status="published" if req.publish else "draft",
#         model_meta={
#             "persona": "news_brief",
#             "strategy": "auto",
#             "top_k": req.top_k,
#             "feed": feed_meta or {},
#             "parsed_ok": bool(llm_out is not None),
#             "service": getattr(settings.news, "service_name", "redfin_news"),
#             "ls_project": getattr(settings.news, "langsmith_project", None),
#         },
#     )
#     post.model_meta["urls"] = {"api": f"/redfin_news/posts/{post.post_id}"}

#     col = get_news_collection()
#     if col is not None:
#         try:
#             col.insert_one(post.dict())
#         except Exception:
#             pass

#     # 출간 직후 자동 인덱싱
#     try:
#         if getattr(settings.news, "index_on_publish", False):
#             upsert_news_post_to_chroma(post)
#     except Exception:
#         pass

#     return post


# # -------------------- NewsLoader 기반 배치 출간 --------------------

# def _req_from_loader_doc(doc) -> NewsPublishRequest:
#     md = dict(doc.metadata or {})
#     content = doc.page_content or ""
#     categories: List[str] = []
#     if md.get("content_type"):
#         categories = [str(md["content_type"])]
#     elif md.get("source"):
#         categories = [str(md["source"])]
#     return NewsPublishRequest(
#         article_id=str(md.get("guid")) if md.get("guid") else None,
#         article_code=str(md.get("guid") or md.get("doc_id")) if (md.get("guid") or md.get("doc_id")) else None,
#         url=md.get("url"),
#         title=md.get("title"),
#         content=content,
#         categories=categories,
#         tags=[str(t) for t in (md.get("tags") or [])],
#         publish=True,
#         top_k=6,
#     )

# def publish_from_loader(
#     feed_url: str,
#     field_map: Optional[Dict[str, str]] = None,
#     default_publish: bool = True,
#     default_top_k: int = 6,
#     *,
#     run_config: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     loader = NewsLoader(field_map=field_map or DEFAULT_FIELD_MAP)
#     docs = loader.load_news_json(feed_url, timeout=timeout if (timeout := 15) else 15)

#     created, skipped, errors = [], [], []
#     for idx, doc in enumerate(docs):
#         try:
#             md = dict(doc.metadata or {})
#             req = _req_from_loader_doc(doc)

#             if default_publish is not None:
#                 req.publish = bool(default_publish)
#             if default_top_k is not None:
#                 try:
#                     req.top_k = int(default_top_k)
#                 except Exception:
#                     req.top_k = 6

#             q = _dedupe_key({
#                 "article_code": req.article_code,
#                 "article_id": req.article_id,
#                 "url": req.url,
#             }) or ({"article_code": str(md.get("doc_id"))} if md.get("doc_id") else {})
#             if q and _exists_in_news(q):
#                 skipped.append({"index": idx, "reason": "duplicate", "key": q})
#                 continue

#             post = generate_news_post(req, feed_meta=md, run_config=run_config)
#             created.append(post.dict())
#         except Exception as e:
#             errors.append({"index": idx, "error": str(e)})
#     return {"created": created, "skipped": skipped, "errors": errors}


# # -------------------- 기존 JSON 배열 배치(옵션으로 유지) --------------------

# def publish_batch(
#     items: List[Dict[str, Any]],
#     field_map: Optional[Dict[str, str]] = None,
#     default_publish: bool = True,
#     default_top_k: int = 6,
#     *,
#     run_config: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     created, skipped, errors = [], [], []
#     m = field_map or {}
#     for idx, it in enumerate(items):
#         try:
#             article_code = str(it.get(m.get("article_code", "article_code")) or it.get("guid") or it.get("id") or "")
#             article_id = str(it.get(m.get("article_id", "article_id")) or it.get("guid") or "")
#             url = it.get(m.get("url", "url")) or it.get("link")
#             title = it.get(m.get("title", "title"))

#             # [CHG] content가 JSON(문자열/객체)일 때 body_md만 추출
#             raw_content = it.get(m.get("content", "content")) or it.get("article_text")
#             content = None
#             try:
#                 if isinstance(raw_content, (dict, list)):
#                     data = raw_content
#                 else:
#                     data = _safe_json_block(str(raw_content))
#                 if isinstance(data, dict) and "body_md" in data:
#                     content = data.get("body_md")
#             except Exception:
#                 pass
#             if content is None:
#                 content = raw_content or ""

#             categories = it.get(m.get("categories", "categories")) or it.get("content_type")
#             tags = it.get(m.get("tags", "tags")) or []

#             req = NewsPublishRequest(
#                 article_id=article_id or None,
#                 article_code=article_code or None,
#                 url=url,
#                 title=title,
#                 content=content,
#                 categories=[categories] if isinstance(categories, str) else (categories or []),
#                 tags=[str(t) for t in (tags if isinstance(tags, list) else [tags] if tags else [])],
#                 publish=bool(it.get(m.get("publish", "publish"), default_publish)),
#                 top_k=int(it.get(m.get("top_k", "top_k"), default_top_k)),
#             )

#             q = _dedupe_key({"article_code": req.article_code, "article_id": req.article_id, "url": req.url})
#             if q and _exists_in_news(q):
#                 skipped.append({"index": idx, "reason": "duplicate", "key": q})
#                 continue

#             post = generate_news_post(req, feed_meta=it, run_config=run_config)
#             created.append(post.dict())
#         except Exception as e:
#             errors.append({"index": idx, "error": str(e)})
#     return {"created": created, "skipped": skipped, "errors": errors}


# # -------------------- 외부 호출 진입점 --------------------

# def publish_from_feed(
#     feed_url: str,
#     item_path: Optional[str] = None,      # 보존(미사용)
#     field_map: Optional[Dict[str, str]] = None,
#     default_publish: bool = True,
#     default_top_k: int = 6,
#     *,
#     run_config: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     return publish_from_loader(
#         feed_url=feed_url,
#         field_map=field_map,
#         default_publish=default_publish,
#         default_top_k=default_top_k,
#         timeout=15,
#         run_config=run_config,
#     )

# def publish_from_env(*, run_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     feed_url = settings.news.api_url or os.getenv("NEWS_API_URL")
#     if not feed_url:
#         raise ValueError("NEWS_API_URL/settings.news.api_url is empty")

#     user_map = getattr(settings.news, "feed_field_map", None)
#     if user_map is None:
#         field_map_json = os.getenv("NEWS_FEED_FIELD_MAP")
#         user_map = json.loads(field_map_json) if field_map_json else None
#     field_map = {**DEFAULT_FIELD_MAP, **user_map} if user_map else DEFAULT_FIELD_MAP

#     default_publish = _as_bool(os.getenv("NEWS_DEFAULT_PUBLISH", None),
#                                default=_as_bool(getattr(settings.news, "default_publish", True), True))
#     default_top_k = _as_int(os.getenv("NEWS_TOP_K", None),
#                             default=_as_int(getattr(settings.news, "top_k", 6), 6))

#     return publish_from_loader(
#         feed_url=feed_url,
#         field_map=field_map,
#         default_publish=default_publish,
#         default_top_k=default_top_k,
#         timeout=15,
#         run_config=run_config,
#     )


# # ------------------- 헬퍼 -------------------

# def _fallback_tldr_from_text(text: str, k: int = 3) -> List[str]:
#     sents = re.split(r"(?<=[.!?。！？])\s+", (text or "").strip())
#     out = []
#     for s in sents:
#         s = s.strip()
#         if not s:
#             continue
#         out.append(s[:90])
#         if len(out) >= max(1, k):
#             break
#     return out


# # ------------------- 고정크기 청킹 + 오버랩 인덱스 초기화 -------------------

# def init_news_index_fixed(
#     *,
#     chunk_size: int = 1200,
#     chunk_overlap: int = 120,
# ) -> dict:
#     try:
#         if not settings.news.api_url:
#             return {"ok": False, "reason": "settings.news.api_url is empty"}

#         loader = NewsLoader(field_map=DEFAULT_FIELD_MAP)
#         docs = loader.load_news_json(settings.news.api_url, timeout=15)
#         if not docs:
#             return {"ok": False, "reason": "no documents loaded", "api_url": settings.news.api_url}

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
#         )
#         chunks = splitter.split_documents(docs)

#         embedding = HuggingFaceEmbeddings(model_name=settings.news.emb_model)

#         vs = create_chroma_store(
#             embedding=embedding,
#             collection_name=settings.news.collection,
#             persist_dir=settings.news.persist_dir,
#             distance="cosine",
#         )
#         vs.vs.add_documents(chunks)
#         vs.persist()

#         return {
#             "ok": True,
#             "mode": "fixed",
#             "api_url": settings.news.api_url,
#             "collection": settings.news.collection,
#             "persist_dir": settings.news.persist_dir,
#             "docs": len(docs),
#             "chunks": len(chunks),
#             "chunk_size": chunk_size,
#             "chunk_overlap": chunk_overlap,
#         }
#     except Exception as e:
#         return {"ok": False, "error": str(e)}


# # ------------------- Mongo → 뉴스 요약/출간 (옵션) -------------------

# def _get_llm_for_news():
#     """출간용 LLM. NEWS__USE_LLM=false면 None 반환해서 패스스루 저장."""
#     from core import settings
#     if not getattr(settings.news, "use_llm", True):
#         return None
#     import os
#     if os.getenv("GOOGLE_API_KEY"):
#         from langchain_google_genai import ChatGoogleGenerativeAI
#         return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL","gemini-2.0-flash"), temperature=0)
#     if os.getenv("OPENAI_API_KEY"):
#         from langchain_openai import ChatOpenAI
#         return ChatOpenAI(model=os.getenv("MODEL_LLM","gpt-4o-mini"), temperature=0)
#     return None

# def _render_body_md(raw_item: Dict[str, Any], llm, prompt_path: str) -> str:
#     """템플릿+LLM로 요약 본문 생성. LLM이 없으면 패스스루(기사 본문 일부/제목만)."""
#     title = raw_item.get("title") or ""
#     content = raw_item.get("article_text") or raw_item.get("content") or ""
#     if not llm:
#         return f"# {title}\n\n{content[:800]}"

#     from pathlib import Path
#     tpl = ""
#     try:
#         p = Path(prompt_path)
#         tpl = p.read_text(encoding="utf-8") if p.exists() else ""
#     except Exception:
#         tpl = ""

#     system = "당신은 기자입니다. 핵심만 간결히 요약하고, 소제목과 불릿을 사용합니다."
#     user = f"{tpl}\n\n[제목]\n{title}\n\n[본문]\n{content}\n\n위 내용을 간결한 기사 요약(Markdown)으로 작성해 주세요."
#     try:
#         resp = llm.invoke([("system", system), ("user", user)])
#         text = getattr(resp, "content", None) or str(resp)
#         return text.strip()
#     except Exception as e:
#         return f"# {title}\n\n{content[:800]}\n\n<!-- LLM 실패: {e} -->"

# def publish_from_source(*, mongo_query: Optional[Dict[str, Any]] = None, limit: int = 30) -> Dict[str, Any]:
#     """
#     Mongo redfin.extract 에서 기사 로드 → (선택)요약 → redfin.news_logs에 upsert 저장.
#     - retriever 사용 없음
#     - api_url 의존 없음
#     """
#     from core import settings
#     from nureongi.loaders import NewsLoader

#     loader = NewsLoader()
#     docs = loader.load_news_mongo(
#         uri=settings.mongo.uri,
#         db=settings.mongo.db,
#         collection=getattr(settings.news, "source_collection", "extract"),
#         query=mongo_query or {},
#         limit=limit,
#         timeout_ms=getattr(settings.mongo, "timeout_ms", 3000),
#         sort=[("processed_at", -1), ("published", -1)],
#     )
#     if not docs:
#         return {"ok": False, "reason": "no input docs from mongo.extract"}

#     llm = _get_llm_for_news()
#     posts: List[Dict[str, Any]] = []
#     for d in docs:
#         md = d.metadata or {}
#         raw = {
#             "title": md.get("title"),
#             "article_text": d.page_content,
#             "source": md.get("source"),
#             "link": md.get("url") or md.get("link"),
#             "guid": md.get("guid"),
#             "published_at": md.get("published_at"),
#             "tags": md.get("tags") or [],
#             "authors": md.get("authors") or [],
#             "language": md.get("lang"),
#         }
#         body_md = _render_body_md(raw, llm, prompt_path=getattr(settings.news, "prompt_path", "src/prompts/templates/news_publish_v1.md"))
#         posts.append({
#             "type": "post",
#             "title": raw["title"],
#             "body_md": body_md,
#             "source": raw["source"],
#             "link": raw["link"],
#             "guid": raw["guid"],
#             "published_at": raw["published_at"],
#             "tags": raw["tags"],
#             "authors": raw["authors"],
#             "lang": raw["language"],
#             "created_at": datetime.utcnow(),
#             "updated_at": datetime.utcnow(),
#         })

#     client = MongoClient(settings.mongo.uri, serverSelectionTimeoutMS=settings.mongo.timeout_ms)
#     coll = client[settings.mongo.db][getattr(settings.mongo, "news_collection", "news_logs")]

#     bulk: List[UpdateOne] = []
#     for p in posts:
#         filt = {"$or": [{"guid": p.get("guid")}, {"link": p.get("link")}]}
#         bulk.append(UpdateOne(filt, {"$set": p}, upsert=True))

#     if bulk:
#         res = coll.bulk_write(bulk, ordered=False)
#         upserts = len(res.upserted_ids or {})
#         modified = res.modified_count
#     else:
#         upserts = modified = 0

#     return {"ok": True, "count": len(posts), "upserts": upserts, "modified": modified}

######################################################################################################################

# src/services/news_service.py
# ----------------------------------------------------------------------
# 변경 요약
# - 라우터에서 전달한 run_config(=callbacks/tags/metadata 포함)를
#   체인/LLM 호출에 그대로 전달하도록 시그니처 확장.
# - env 스왑 컨텍스트(langsmith_project) 제거. (동시성 안전)
# - 나머지 퍼블리시 플로우/중복 판정/Loader 연계는 기존 유지.
# ----------------------------------------------------------------------
from __future__ import annotations
import os
import json
import urllib.request
import re
from uuid import uuid4
from typing import Any, Dict, Optional, List
from datetime import datetime
from pymongo import MongoClient, UpdateOne

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

def _get_llm_for_news():
    """출간용 LLM. NEWS__USE_LLM=false면 None 반환해서 패스스루 저장."""
    from core import settings
    if not getattr(settings.news, "use_llm", True):
        return None
    # 우선순위: GOOGLE → OPENAI
    import os
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL","gemini-2.0-flash"), temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("MODEL_LLM","gpt-4o-mini"), temperature=0)
    # 키가 없으면 패스스루
    return None

def _render_body_md(raw_item: Dict[str, Any], llm, prompt_path: str) -> str:
    """템플릿+LLM로 요약 본문 생성. LLM이 없으면 패스스루(기사 본문 일부/제목만)."""
    title = raw_item.get("title") or ""
    content = raw_item.get("article_text") or raw_item.get("content") or ""
    if not llm:
        # 패스스루: 제목 + 본문 앞부분 800자
        return f"# {title}\n\n{content[:800]}"

    # 템플릿 로딩
    from pathlib import Path
    tpl = ""
    try:
        p = Path(prompt_path)
        tpl = p.read_text(encoding="utf-8") if p.exists() else ""
    except Exception:
        tpl = ""

    # 간단 프롬프트
    system = "당신은 기자입니다. 핵심만 간결히 요약하고, 소제목과 불릿을 사용합니다."
    user = f"{tpl}\n\n[제목]\n{title}\n\n[본문]\n{content}\n\n위 내용을 간결한 기사 요약(Markdown)으로 작성해 주세요."
    try:
        resp = llm.invoke([("system", system), ("user", user)])
        text = getattr(resp, "content", None) or str(resp)
        return text.strip()
    except Exception as e:
        return f"# {title}\n\n{content[:800]}\n\n<!-- LLM 실패: {e} -->"

def publish_from_source(*, mongo_query: Optional[Dict[str, Any]] = None, limit: int = 30) -> Dict[str, Any]:
    """
    Mongo redfin.extract 에서 기사 로드 → (선택)요약 → redfin.news_logs에 upsert 저장.
    - retriever 사용 없음
    - api_url 의존 없음
    """
    from core import settings
    from nureongi.loaders import NewsLoader

    # 1) 입력 로드 (Mongo 최신 우선, 최대 limit)
    loader = NewsLoader()
    docs = loader.load_news_mongo(
        uri=settings.mongo.uri,
        db=settings.mongo.db,
        collection=getattr(settings.news, "source_collection", "extract"),
        query=mongo_query or {},
        limit=limit,
        timeout_ms=getattr(settings.mongo, "timeout_ms", 3000),
        sort=[("processed_at", -1), ("published", -1)],
    )
    if not docs:
        return {"ok": False, "reason": "no input docs from mongo.extract"}

    # 2) 요약/본문 생성
    llm = _get_llm_for_news()
    posts: List[Dict[str, Any]] = []
    for d in docs:
        md = d.metadata or {}
        raw = {
            "title": md.get("title"),
            "article_text": d.page_content,  # loaders에서 title+content로 page_content 구성한 경우, 요약 함수에서 다시 잘라 씀
            "source": md.get("source"),
            "link": md.get("url") or md.get("link"),
            "guid": md.get("guid"),
            "published_at": md.get("published_at"),
            "tags": md.get("tags") or [],
            "authors": md.get("authors") or [],
            "language": md.get("lang"),
        }
        body_md = _render_body_md(raw, llm, prompt_path=getattr(settings.news, "prompt_path", "src/prompts/templates/news_publish_v1.md"))
        posts.append({
            "type": "post",
            "title": raw["title"],
            "body_md": body_md,
            "source": raw["source"],
            "link": raw["link"],
            "guid": raw["guid"],
            "published_at": raw["published_at"],
            "tags": raw["tags"],
            "authors": raw["authors"],
            "lang": raw["language"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        })

    # 3) Mongo 저장 (upsert by guid/link)
    client = MongoClient(settings.mongo.uri, serverSelectionTimeoutMS=settings.mongo.timeout_ms)
    coll = client[settings.mongo.db][getattr(settings.mongo, "news_collection", "news_logs")]

    bulk: List[UpdateOne] = []
    for p in posts:
        filt = {"$or": [{"guid": p.get("guid")}, {"link": p.get("link")}]}
        bulk.append(UpdateOne(filt, {"$set": p}, upsert=True))

    if bulk:
        res = coll.bulk_write(bulk, ordered=False)
        upserts = len(res.upserted_ids or {})
        modified = res.modified_count
    else:
        upserts = modified = 0

    return {"ok": True, "count": len(posts), "upserts": upserts, "modified": modified}
