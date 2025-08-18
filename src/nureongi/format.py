from typing import List
from langchain.schema import Document


# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(f"<document>{doc.page_content}</document>" for doc in docs)


# 컨텍스트 포맷터(출처 강제)
def build_cite(md: dict) -> str:
    title = md.get("title") or md.get("source") or "문서"
    ministry = md.get("ministry") or md.get("publisher") or "기관"
    year = (md.get("publish_date") or md.get("date") or "")[:4]
    page = md.get("page") or md.get("page_number") or "?"
    return f"[{ministry}/{title}({year}):{page}]"

def format_ctx(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        cite = build_cite(d.metadata or {})
        blocks.append(f"{d.page_content}\nCITE: {cite}")
    return "\n\n---\n\n".join(blocks)
