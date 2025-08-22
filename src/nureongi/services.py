# nureongi/services.py
import os 
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from .loaders import NewsLoader
from .pipeline import TextChunker, VectorIndexer, RagChain
from .raptor import RaptorSummarizer
from .persona import get_persona_by_alias, render_prompt_args
from .vectorstore import auto_qdrant_faiss, as_retriever

class VectorIndexer:
    def __init__(self, embedding, distance: str = "cosine"):
        self.embedding = embedding
        self.distance = DistanceStrategy.COSINE if distance == "cosine" else DistanceStrategy.L2

    def build(self, docs: List[Document], save_dir: Optional[Path] = None) -> FAISS:
        vs = FAISS.from_documents(docs, embedding=self.embedding, distance_strategy=self.distance)
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            vs.save_local(str(save_dir))
        return vs


def build_rag_chain(llm, vectorstore, k: int = 8, fetch_k: int = 60, lambda_mult: float = 0.25,
                    persona: str = "ai_industry_professional"):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )

    def _format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    # 페르소나 템플릿 선택 (없으면 안전한 기본 프롬프트로 폴백)
    spec = get_persona_by_alias(persona)
    if spec is None:
        from langchain_core.prompts import PromptTemplate
        base = PromptTemplate.from_template(
            "Use the following CONTEXT to answer the QUESTION concisely.\n"
            "If not found, say you don't know.\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}\n"
        )
        def _format_payload(inp):
            return base.format(question=inp["question"], context=inp["context"])
    else:
        def _format_payload(inp):
            args = render_prompt_args(
                question=inp["question"],
                context=inp["context"],
                now=datetime.now().strftime("%Y-%m-%d %H:%M"),
                role=spec.label
            )
            return spec.template.format(**args)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(_format_payload)
        | llm
        | StrOutputParser()
    )
    return chain

class RedfinApps:
    def __init__(self, emb, llm, chunk_size=500, chunk_overlap=120, index_mode="title+summary"):
        self.emb = emb
        self.llm = llm
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, index_mode=index_mode)
        self.loader = NewsLoader()

    def build_index(self, news_url: str, use_raptor: bool = True, raptor_levels: int = 2,
                    save_dir: Optional[Path] = None,
                    collection_name: Optional[str] = None,
                    distance: str = "cosine"):
        news_docs = self.loader.load_news_json(news_url)
        chunks = self.chunker.split(news_docs)

        if use_raptor:
            from .raptor import RaptorSummarizer
            texts = [d.page_content for d in chunks]
            raptor = RaptorSummarizer(emb=self.emb, llm=self.llm)
            levels = raptor.summarize_levels(texts, max_levels=raptor_levels)
            raptor_docs = raptor.as_documents(levels)
            chunks = chunks + raptor_docs

        coll = collection_name or os.getenv("QDRANT_COLLECTION", "redfin_news")
        faiss_dir = save_dir or Path(os.getenv("FAISS_DIR", "./faiss_bge_base_raptor"))
        vsr = auto_qdrant_faiss(
            embedding=self.emb,
            collection_name=coll,
            docs=chunks,
            faiss_dir=faiss_dir,
            distance=distance,               # 기본 cosine
            content_payload_key="page_content",
            # qdrant_url=os.getenv("QDRANT_URL"),       # 필요 시 오버라이드
            # qdrant_api_key=os.getenv("QDRANT_API_KEY")
        )
        print(f"[INFO] VectorStore backend={vsr.backend} details={vsr.details}")
        return vsr.vectorstore

class RedfinNewsService:
    """
    기능 ① redfin_news: 스크랩 기사 기반 요약 기사 출간(클러스터 요약)
    """
    def __init__(self, emb, llm):
        self.base = RedfinApps(emb=emb, llm=llm, index_mode="summary_only")  # 제목 누수 방지

    def build(self, news_url: str, use_raptor: bool = True, levels: int = 2, save_dir: Optional[Path] = None) -> FAISS:
        return self.base.build_index(news_url, use_raptor=use_raptor, raptor_levels=levels, save_dir=save_dir)

    def generate_issue_briefs(self, vectorstore: FAISS, llm=None, k: int = 12) -> List[str]:
        """
        인덱스 내 RAPTOR 요약(raptor_summary)만 추려 상위 이슈 브리프 생성
        """
        llm = llm or self.base.llm
        docs = vectorstore.similarity_search("Top story clusters and their key takeaways.", k=k)
        summaries = [d.page_content for d in docs if (d.metadata or {}).get("type") == "raptor_summary"]
        if not summaries:
            summaries = [d.page_content for d in docs]
        prompt = PromptTemplate.from_template(
            "Merge the following cluster summaries into a publishable daily brief (<=10 bullets):\n\n{context}"
        )
        chain = prompt | (llm) | StrOutputParser()
        return [chain.invoke({"context": "\n\n---\n\n".join(summaries)})]

class RedfinTargetInsightService:
    """
    기능 ② redfin_target-insight: 사용자 프롬프트에 컨텍스트 기반 요약 답변 제공
    """
    def __init__(self, emb, llm):
        self.base = RedfinApps(emb=emb, llm=llm, index_mode="summary_only")

    def build(self, news_url: str, use_raptor: bool = True, levels: int = 2, save_dir: Optional[Path] = None) -> FAISS:
        return self.base.build_index(news_url, use_raptor=use_raptor, raptor_levels=levels, save_dir=save_dir)

    def make_chain(self, vectorstore: FAISS, persona: str = "ai_industry_professional"):
        return build_rag_chain(self.base.llm, vectorstore, k=8, fetch_k=60, lambda_mult=0.25, persona=persona)

