# nureongi/loaders.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Iterable, Mapping
import re, html, hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.documents import Document
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson import ObjectId

# 기본 매핑: 너희 JSON 구조에 맞춤
DEFAULT_FIELD_MAP: Dict[str, str] = {
    "id": "guid",               # 고유 식별자
    "title": "title",           # 제목
    "content": "article_text",  # 본문 ← 핵심!
    "url": "link",              # 원문 링크
    "source": "source",         # 출처
    "published_at": "processed_at",  # 공개/처리 시각 (있으면 사용)
    "authors": "authors",       # 작성자 배열 (없을 수 있음)
    "tags": "tags",             # 태그
    "lang": "language",         # 언어 ("ENGLISH" 등)
}

def _to_epoch_any(x) -> Optional[float]:
    """str(ISO8601) 또는 datetime 모두 epoch로 변환"""
    if x is None:
        return None
    try:
        # datetime 인스턴스
        if hasattr(x, "timestamp"):
            return float(x.timestamp())
        # 문자열
        s = str(x).replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None

def _to_epoch(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None

def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return html.unescape(s)

def _prefer(*vals) -> str:
    for v in vals:
        if v:
            return str(v)
    return ""

def _norm_lang(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    x = str(x).strip().lower()
    # 간단 표준화
    if x in ("en", "english"):
        return "en"
    if x in ("ko", "korean", "kr"):
        return "ko"
    if x in ("ja", "japanese"):
        return "ja"
    if x in ("zh", "chinese", "cn", "zh-cn", "zh-tw"):
        return "zh"
    return x

def _doc_id(md: Dict[str, Any]) -> str:
    # guid > link > title 해시
    gid = (md or {}).get("guid") or (md or {}).get("url")
    if gid:
        return str(gid)
    title = (md or {}).get("title") or ""
    return "title:" + hashlib.md5(title.encode("utf-8")).hexdigest()

def _session_with_retry(total=3, backoff=0.3) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total, backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST")
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

class NewsLoader:
    """뉴스 API(JSON array) → LangChain Document 리스트 변환기
    - field_map으로 외부 스키마를 내부 표준 키로 매핑
    - transform 콜백으로 레코드 후처리 가능
    """
    def __init__(
        self,
        session: Optional[requests.Session] = None,
        field_map: Optional[Dict[str, str]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.session = session or _session_with_retry()
        self.field_map = field_map or DEFAULT_FIELD_MAP
        self.transform = transform

    def _map_field(self, item: Dict[str, Any], key: str) -> Any:
        src = self.field_map.get(key, key)
        return item.get(src)

    def load_news_json(self, url: str, timeout: int = 10) -> List[Document]:
        docs: List[Document] = []
        try:
            r = self.session.get(url, timeout=timeout)
            r.raise_for_status()
            items = r.json()
            # API가 {"items":[...]} 형태라면 보정
            if isinstance(items, dict) and "items" in items:
                items = items["items"]
            if not isinstance(items, list):
                items = []
        except Exception as e:
            print(f"⚠️ 뉴스 API 로드 실패: {e}")
            items = []

        for it in items:
            if self.transform:
                it = self.transform(it)

            title = _prefer(self._map_field(it, "title"), "")
            raw_content = _prefer(self._map_field(it, "content"), "")
            content = _strip_html(raw_content)

            if not content and not title:
                continue  # 내용이 전혀 없으면 스킵

            # 메타데이터 구성
            md = {
                "type": "news",
                "source": self._map_field(it, "source"),
                "url": self._map_field(it, "url"),
                "guid": self._map_field(it, "id"),
                "published_at": self._map_field(it, "published_at"),
                "published_at_ts": _to_epoch(self._map_field(it, "published_at")),
                "authors": self._map_field(it, "authors"),
                "tags": self._map_field(it, "tags"),
                "title": title,
                "lang": _norm_lang(self._map_field(it, "lang")),
                # 원본 보조 필드도 보관(있으면)
                "content_type": it.get("content_type"),
                "readability_score": it.get("readability_score"),
                "text_length": it.get("text_length"),
                "key_entities": it.get("key_entities"),
            }

            page = f"{title}\n\n{content}".strip() if title else content
            docs.append(Document(page_content=page, metadata=md))

        # doc_id 부여 (업서트/표시/중복제거용)
        out: List[Document] = []
        for d in docs:
            md = dict(d.metadata or {})
            md["doc_id"] = _doc_id(md)
            out.append(Document(page_content=d.page_content, metadata=md))
        return out
    
    
    def _to_documents(self, items: List[Dict[str, Any]]) -> List[Document]:
        """HTTP/Mongo에서 읽은 dict 목록을 공통 Document[]로 통일"""
        docs: List[Document] = []
        for it in items:
            it = dict(it or {})
            if self.transform:
                it = self.transform(it)

            title = _prefer(self._map_field(it, "title"), "")
            raw_content = _prefer(self._map_field(it, "content"), "")
            content = _strip_html(raw_content)

            if not content and not title:
                continue

            # published/processed 값은 어떤 키가 오든 하나로 수렴
            published_any = (
                self._map_field(it, "published_at")
                or it.get("processed_at")
                or it.get("published")
            )

            md = {
                "type": "news",
                "source": self._map_field(it, "source") or it.get("source"),
                "url": self._map_field(it, "url") or it.get("link"),
                "guid": self._map_field(it, "id") or str(it.get("_id") or ""),
                "published_at": published_any,
                "published_at_ts": _to_epoch_any(published_any),
                "authors": self._map_field(it, "authors") or it.get("authors"),
                "tags": self._map_field(it, "tags") or it.get("tags", []),
                "title": title,
                "lang": _norm_lang(self._map_field(it, "lang") or it.get("language")),
                # 원본 보조 필드(있으면 유지)
                "content_type": it.get("content_type"),
                "readability_score": it.get("readability_score"),
                "text_length": it.get("text_length"),
                "key_entities": it.get("key_entities"),
            }

            page = f"{title}\n\n{content}".strip() if title else content
            docs.append(Document(page_content=page, metadata=md))

        # 업서트/표시/중복제거 용 doc_id
        out: List[Document] = []
        for d in docs:
            md = dict(d.metadata or {})
            md["doc_id"] = _doc_id(md)
            out.append(Document(page_content=d.page_content, metadata=md))
        return out

    def load_news_mongo(
        self,
        uri: str,
        db: str,
        collection: str,
        query: Optional[Mapping[str, Any]] = None,
        limit: Optional[int] = None,
        timeout_ms: Optional[int] = 3000,
        projection: Optional[Mapping[str, int]] = None,
        sort: Optional[Iterable] = None,
    ) -> List[Document]:
        """MongoDB redfin.extract에서 기사 읽어 Document[]로 변환"""
        items: List[Dict[str, Any]] = []
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
            coll = client[db][collection]

            q = dict(query or {})
            cur = coll.find(q, projection)
            if sort:
                cur = cur.sort(list(sort))
            if limit and limit > 0:
                cur = cur.limit(int(limit))

            for doc in cur:
                if "_id" in doc and isinstance(doc["_id"], ObjectId):
                    doc["_id"] = str(doc["_id"])
                items.append(doc)

        except Exception as e:
            print(f"⚠️ Mongo 로드 실패: {e}")
            items = []

        return self._to_documents(items)

    def load_from_source(
        self,
        settings,
        http_url: Optional[str] = None,
        mongo_query: Optional[Mapping[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Document]:
        """환경설정에 따라 자동으로 HTTP/Mongo에서 로드"""
        src = (getattr(settings.news, "ingest_source", None) or "http").lower()
        if src == "mongo":
            return self.load_news_mongo(
                uri=settings.mongo.uri,
                db=settings.mongo.db,
                collection=getattr(settings.news, "source_collection", "extract"),
                query=mongo_query,
                limit=limit,
                timeout_ms=getattr(settings.mongo, "timeout_ms", 3000),
                # 최신 우선: processed_at / published 둘 중 있는 값 기준으로 내림차순
                sort=[("processed_at", -1), ("published", -1)]
            )
        # 기본 HTTP
        url = http_url or getattr(settings.news, "api_url", None)
        items = []
        if url:
            # 기존 load_news_json을 재사용하되, 반환을 _to_documents 형식과 맞추려면
            # load_news_json 내부 로직을 건드리지 않고 그대로 사용합니다.
            return self.load_news_json(url)
        return []
