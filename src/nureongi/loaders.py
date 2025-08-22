# nureongi/loaders.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import re, html, hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain.schema import Document

def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return html.unescape(s)

def _doc_id(md: Dict[str, Any]) -> str:
    link = (md or {}).get("link") or ""
    title = (md or {}).get("title") or ""
    if link:
        return link
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
    """뉴스 API(JSON array) → LangChain Document 리스트 변환기"""
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or _session_with_retry()

    def load_news_json(self, url: str, timeout: int = 10) -> List[Document]:
        docs: List[Document] = []
        try:
            r = self.session.get(url, timeout=timeout)
            r.raise_for_status()
            items = r.json()
            if not isinstance(items, list):
                items = []
        except Exception as e:
            print(f"⚠️ 뉴스 API 로드 실패: {e}")
            items = []

        for it in items:
            title = it.get("title") or ""
            summary = _strip_html(it.get("summary") or "")
            body = f"{title}\n\n{summary}".strip()
            md = {
                "type": "news",
                "source": it.get("source"),
                "link": it.get("link"),
                "published": it.get("published"),
                "authors": it.get("authors"),
                "tags": it.get("tags"),
                "title": title,
            }
            if body:
                docs.append(Document(page_content=body, metadata=md))

        # doc_id 부여 (기사 단위 dedup/평가/표시에 필요)
        out: List[Document] = []
        for d in docs:
            md = dict(d.metadata or {})
            md["doc_id"] = _doc_id(md)
            out.append(Document(page_content=d.page_content, metadata=md))
        return out
