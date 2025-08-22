# nureongi/indexing.py
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .loaders import NewsLoader
from .vectorstore import auto_qdrant_faiss, VSReturn

@dataclass
class TextChunker:
    chunk_size: int = 500
    chunk_overlap: int = 120
    index_mode: str = "summary_only"  # "title+summary" | "summary_only"

    def _apply_index_mode(self, docs: List[Document]) -> List[Document]:
        if self.index_mode == "title+summary":
            return docs
        out = []
        for d in docs:
            lines = (d.page_content or "").splitlines()
            new_content = "\n".join(lines[1:]).lstrip() if lines else d.page_content
            out.append(Document(page_content=new_content, metadata=d.metadata))
        return out

    def split(self, docs: List[Document]) -> List[Document]:
        docs2 = self._apply_index_mode(docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return splitter.split_documents(docs2)

def build_index(
    news_url: str,
    emb,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 120,
    index_mode: str = "summary_only",
    use_raptor: bool = True,    # RAPTOR 요약 레벨 사용 여부
    raptor_levels: int = 2,
    collection_name: Optional[str] = None,
    faiss_dir: Optional[Path] = None,
    distance: str = "cosine",
):
    loader = NewsLoader()
    news_docs = loader.load_news_json(news_url)
    if not news_docs:
        raise SystemExit("뉴스 문서가 0건입니다. NEWS_API_URL 또는 API 응답을 확인하세요.")

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, index_mode=index_mode)
    chunks = chunker.split(news_docs)

    # RAPTOR 요약 레벨 추가
    if use_raptor:
        try:
            from .raptor import RaptorSummarizer
            # RAPTOR는 LLM이 필요하므로 외부에서 주입하는 대신 간단히 비활성도 가능
            raise_if_no_llm = False  # 필요하면 True로 바꿔서 강제
            # texts = [d.page_content for d in chunks]
            # raptor = RaptorSummarizer(emb=emb, llm=llm)  # llm 인자는 서비스/테스트 단에서 주입 권장
            # levels = raptor.summarize_levels(texts, max_levels=raptor_levels)
            # chunks += raptor.as_documents(levels)
        except Exception as e:
            print(f"[WARN] RAPTOR disabled or not available: {e}")

    coll = collection_name or os.getenv("QDRANT_COLLECTION", "redfin_news")
    faiss_dir = faiss_dir or Path(os.getenv("FAISS_DIR", "./faiss_bge_base_raptor"))
    vsr: VSReturn = auto_qdrant_faiss(
        embedding=emb,
        collection_name=coll,
        docs=chunks,
        faiss_dir=faiss_dir,
        distance=distance,
    )
    return vsr.vectorstore, {"backend": vsr.backend, **vsr.details, "n_docs": len(chunks)}
