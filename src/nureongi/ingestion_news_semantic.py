from __future__ import annotations
from typing import Optional, List
from pathlib import Path
import os

from langchain_core.documents import Document

# 시맨틱 청킹
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# 기존 자산 재사용
from .loaders import NewsLoader
from .vectorstore import create_chroma_store, VSReturn  # 기존 Chroma 래퍼 재사용

def build_news_semantic_index(
    news_url: str,
    *,
    emb_model: str = "BAAI/bge-base-en-v1.5",
    collection_name: Optional[str] = None,
    persist_dir: Optional[str | Path] = None,
    distance: str = "cosine",
) -> tuple[object, dict]:
    """
    뉴스 전용: 시맨틱 청킹 → 임베딩 → Chroma 인덱싱.
    기존 indexing.py와 독립적으로 동작 (회귀 리스크 최소화).
    """
    # 1) 문서 로드 (기존 loaders 재사용)
    loader = NewsLoader()
    docs: List[Document] = loader.load_news_json(news_url)
    if not docs:
        raise SystemExit("뉴스 문서가 0건입니다. NEWS_API_URL 또는 API 응답을 확인하세요.")

    # 2) 시맨틱 청킹
    emb_for_chunk = HuggingFaceEmbeddings(model_name=emb_model)
    splitter = SemanticChunker(embeddings=emb_for_chunk, breakpoint_threshold_type="percentile")
    chunks = splitter.split_documents(docs)

    # 3) 벡터스토어 생성 (기존 create_chroma_store 재사용)
    coll = collection_name or os.getenv("NEWS_COLLECTION", "news_semantic_v1")
    pdir = str(persist_dir or os.getenv("CHROMA_DIR", "./.chroma"))
    vsr: VSReturn = create_chroma_store(
        embedding=emb_for_chunk,   # 같은 임베딩으로 인덱싱
        collection_name=coll,
        docs=chunks,
        persist_dir=pdir,
        distance=distance,
    )

    info = {"backend": vsr.backend, **vsr.details, "n_docs": len(chunks)}
    return vsr.vectorstore, info
