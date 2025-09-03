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
    
def load_news_from_mongo(
    mongo_uri: str,
    db_name: str,
    collection: str,
    *,
    query: Optional[Mapping[str, Any]] = None,
    projection: Optional[Mapping[str, int]] = None,
    limit: Optional[int] = None,
    recency_days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    MongoDB에서 기사 원문을 읽어, 기존 HTTP JSON과 동일한 스키마로 정규화하여 반환.
    반환 스키마 예시:
    {
        "guid": str,              # 고유키
        "title": str,
        "content": str,           # 기사 본문 (기존 article_text -> content 매핑)
        "url": str | None,
        "category": str | None,
        "published_at": str | None   # ISO8601
    }
    """
    _query = dict(query or {})

    # 최근 N일 필터가 있다면 간단히 published_at/created_at 계열로 가정 필터
    if recency_days and recency_days > 0:
        threshold = datetime.utcnow() - timedelta(days=recency_days)
        # 컬럼명이 프로젝트마다 다를 수 있음: published_at, pubDate, created_at 중 하나를 우선 시도
        _query.setdefault("$or", [
            {"published_at": {"$gte": threshold}},
            {"pubDate": {"$gte": threshold}},
            {"created_at": {"$gte": threshold}},
        ])

    _proj = dict(projection or {})
    # 가져오고 싶은 대표 필드들 (없어도 동작하게 유연화)
    if not _proj:
        _proj = {
            "_id": 1,
            "title": 1,
            "article_text": 1,  # 중요: 본문
            "url": 1,
            "link": 1,          # url 대체 후보
            "category": 1,
            "categories": 1,
            "published_at": 1,
            "pubDate": 1,
            "created_at": 1,
        }

    client = MongoClient(mongo_uri)
    try:
        cur = (client[mdb_name := db_name][collection]
               .find(_query, _proj)
               .sort([("_id", -1)]))
        if limit:
            cur = cur.limit(int(limit))

        items: List[Dict[str, Any]] = []
        for doc in cur:
            _id = doc.get("_id")
            guid = str(_id) if isinstance(_id, ObjectId) else (str(_id) if _id else None)

            # 본문 후보: article_text > content > text
            content = (
                doc.get("article_text")
                or doc.get("content")
                or doc.get("text")
                or ""
            )

            # URL 후보: url > link
            url = doc.get("url") or doc.get("link")

            # 카테고리 후보: category > categories[0]
            category = doc.get("category")
            if not category:
                cats = doc.get("categories")
                if isinstance(cats, list) and cats:
                    category = cats[0]

            # 발행일 후보
            published = doc.get("published_at") or doc.get("pubDate") or doc.get("created_at")
            if isinstance(published, datetime):
                published_iso = published.isoformat()
            else:
                published_iso = str(published) if published else None

            items.append({
                "guid": guid,
                "title": doc.get("title") or "",
                "content": content,
                "url": url,
                "category": category,
                "published_at": published_iso,
            })
        return items
    finally:
        client.close()    

