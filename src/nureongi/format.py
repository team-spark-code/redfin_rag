# nureongi/format.py
from langchain.schema import Document
from typing import List

def format_docs(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        src = md.get("link") or md.get("source") or ""
        title = md.get("title") or ""
        lines.append(f"[{i}] {title}\n{d.page_content}\n(Source: {src})")
    return "\n\n---\n\n".join(lines)

# alias (이름 유지 호환)
format_ctx = format_docs
