# [신규 파일] src/nureongi/vs_news.py
from __future__ import annotations
from typing import Dict, Any
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from core.settings import settings

def upsert_news_post_to_chroma(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    출간된 '우리 기사'를 Chroma 컬렉션에 upsert (retriever는 쓰지 않음).
    id=post_id, text=title + 2줄 개행 + body_md, metadata=기본 필드
    """
    emb = HuggingFaceEmbeddings(model_name=settings.news.emb_model)
    vs = Chroma(
        collection_name=settings.news.vector_collection_posts,
        embedding_function=emb,
        persist_directory=settings.news.persist_dir,
    )

    title = (post.get("title") or "").strip()
    body = (post.get("body_md") or post.get("content") or "").strip()
    content = f"{title}\n\n{body}".strip()

    md = {
        "post_id": post.get("post_id"),
        "url": post.get("url"),
        "author": post.get("author_name"),
        "tags": post.get("tags") or [],
        "categories": post.get("categories") or [],
        "status": post.get("status"),
        "published_at": post.get("created_at") or datetime.utcnow().isoformat(),
        "source": "our_news_post",
    }

    try:
        vs.delete(ids=[post.get("post_id")])  # 동일 id 있으면 교체
    except Exception:
        pass

    vs.add_texts(texts=[content], metadatas=[md], ids=[post.get("post_id")])
    return {"collection": settings.news.vector_collection_posts, "ok": True}

# ---- (미래용: 리트리버는 주간/월간 리포트에서만 켜세요. 지금은 주석으로만 남김) ----
# def get_news_posts_retriever(k: int = 8, fetch_k: int = 32, lambda_mult: float = 0.25):
#     emb = HuggingFaceEmbeddings(model_name=settings.news.emb_model)
#     vs = Chroma(
#         collection_name=settings.news.vector_collection_posts,
#         embedding_function=emb,
#         persist_directory=settings.news.persist_dir,
#     )
#     return vs.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
#     )
