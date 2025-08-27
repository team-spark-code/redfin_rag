# nureongi/indexing.py
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .loaders import NewsLoader
from .vectorstore import create_chroma_store, VSReturn   # <- 내부 구현이 Chroma 전용으로 바뀌어 있음
from .raptor import raptor_build_and_compress, RaptorParams


# 인덱싱 시 요약에 사용할 LLM (chain.py와 동일한 우선순위)
def _default_llm():
    """
    환경변수에 따라 기본 LLM 객체를 생성하는 헬퍼 함수.
    - GOOGLE_API_KEY가 있으면 Gemini(ChatGoogleGenerativeAI) 사용
    - OPENAI_API_KEY가 있으면 OpenAI(ChatOpenAI) 사용
    - 두 키가 모두 없으면 RuntimeError 발생
    """
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL","gemini-2.0-flash"), temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("MODEL_LLM","gpt-4o-mini"), temperature=0)
    raise RuntimeError("No LLM key found. Set GOOGLE_API_KEY or OPENAI_API_KEY.")


@dataclass
class TextChunker:
    """
    뉴스 문서를 청크 단위로 나누는 도우미 클래스.
    - chunk_size: 각 청크 최대 길이
    - chunk_overlap: 청크 간 중첩 길이
    - index_mode:
        - "title+summary": 문서 전체를 청킹
        - "summary_only": 첫 줄(제목/요약)을 제외하고 본문만 청킹
    """
    chunk_size: int = 500
    chunk_overlap: int = 120
    index_mode: str = "summary_only"  # "title+summary" | "summary_only"

    def _apply_index_mode(self, docs: List[Document]) -> List[Document]:
        """
        index_mode 규칙을 적용해 문서 내용을 가공.
        - summary_only 모드일 경우: 첫 줄(제목/요약)을 제외한 본문만 남김
        """
        if self.index_mode == "title+summary":
            return docs
        # summary_only: 1행(제목/요약)을 제외하고 본문 중심으로 인덱싱
        out: List[Document] = []
        for d in docs:
            lines = (d.page_content or "").splitlines()
            new_content = "\n".join(lines[1:]).lstrip() if lines else d.page_content
            out.append(Document(page_content=new_content, metadata=d.metadata))
        return out

    def split(self, docs: List[Document]) -> List[Document]:
        """
        문서 리스트를 RecursiveCharacterTextSplitter로 청킹.
        index_mode 규칙 적용 후 chunk_size/overlap 기준으로 분할.
        """
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
    use_raptor: bool = True,            # 인덱싱 단계 RAPTOR 적용 여부(권장 True)
    raptor_levels: int = 2,             # RAPTOR 트리 깊이(2~3 권장)
    collection_name: Optional[str] = None,
    chroma_dir: Optional[Path] = None,  # 영속 디렉터리(.env: CHROMA_DIR)
    distance: str = "cosine",           # "cosine" | "l2" | "dot(ip)"
):
    """
    뉴스 JSON을 불러와 청킹/요약/벡터스토어 인덱싱을 수행하는 엔트리포인트 함수.
    
    단계:
    1) NewsLoader로 뉴스 문서 로드
    2) TextChunker로 청크 분할
    3) (옵션) RAPTOR 요약 적용 → 문서 수 축소
    4) Chroma 기반 벡터스토어 생성 및 영속화
    
    Args:
        news_url: 뉴스 JSON 파일 경로 혹은 URL
        emb: HuggingFaceEmbeddings 인스턴스
        chunk_size: 청크 최대 길이
        chunk_overlap: 청크 간 중첩
        index_mode: 'title+summary' 또는 'summary_only'
        use_raptor: RAPTOR 요약 여부
        raptor_levels: RAPTOR 트리 깊이
        collection_name: Chroma 컬렉션 이름
        chroma_dir: 영속 디렉터리 경로
        distance: 벡터 거리 측정 방식 ("cosine"/"l2"/"ip")

    Returns:
        (vectorstore, info)
        - vectorstore: 생성된 Chroma 객체
        - info: backend, 경로, 문서 수 등의 메타정보
    """
    # 1) 로드
    loader = NewsLoader()
    news_docs = loader.load_news_json(news_url)
    if not news_docs:
        raise SystemExit("뉴스 문서가 0건입니다. NEWS_API_URL 또는 API 응답을 확인하세요.")

    # 2) 청크
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, index_mode=index_mode)
    chunks = chunker.split(news_docs)

    # 3) 인덱싱 시 RAPTOR 요약(권장)
    if use_raptor:
        try:
            llm = _default_llm()
            # 환경변수 노브 반영
            target_k = int(os.getenv("RAPTOR_TARGET_K", "24"))
            per_cluster_topm = int(os.getenv("RAPTOR_PER_CLUSTER_TOPM", "6"))
            n_comp = int(os.getenv("RAPTOR_N_COMPONENTS", "10"))
            n_nei_env = os.getenv("RAPTOR_N_NEIGHBORS")
            n_nei = int(n_nei_env) if n_nei_env else None
            maxtok = int(os.getenv("RAPTOR_SUMMARY_MAXTOK", "256"))

            params = RaptorParams(
                max_levels=raptor_levels,
                target_k=target_k,
                per_cluster_topm=per_cluster_topm,
                n_components=n_comp,
                n_neighbors=n_nei,
                summary_maxtok=maxtok,
            )
            # 트리 요약 → 요약 문서만 인덱싱
            chunks = raptor_build_and_compress(chunks, llm=llm, params=params)
        except Exception as e:
            print(f"[WARN] RAPTOR indexing failed, using raw chunks: {e}")

    # 4) Chroma 벡터스토어 구축
    coll = collection_name or os.getenv("CHROMA_COLLECTION", "redfin_vectordb")
    # persist 디렉터리는 vectorstore 쪽에서 CHROMA_DIR 기본값을 사용함
    vsr: VSReturn = create_chroma_store(
        embedding=emb,
        collection_name=coll,
        docs=chunks,
        # 아래 인자들은 Chroma 전용 구현에서 무시되지만, 인터페이스 유지 차원에서 남겨둠
        persist_dir=chroma_dir or os.getenv("CHROMA_DIR", "./.chroma"),
        distance=distance,
    )

    info = {"backend": vsr.backend, **vsr.details, "n_docs": len(chunks)}
    return vsr.vectorstore, info
