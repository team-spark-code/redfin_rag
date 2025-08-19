#!/usr/bin/env python
# coding: utf-8
"""
RAPTOR-style RAG (no external helpers)
- 대용량 PDF 다건 병렬 로딩 + 캐싱
- 청크 분할 → 임베딩(OpenAI) → RAPTOR식 클러스터 요약(레벨 업)
- leaf + 요약 전체를 FAISS 벡터스토어로 구축
- Retriever(MMR) + 프롬프트 + LLM으로 질의응답
"""

import os
import json
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ===== 환경변수 로드 =====
from dotenv import load_dotenv
load_dotenv()  # OPENAI_API_KEY, LANGSMITH_* 등

# ===== LangSmith (선택) =====
from langsmith import Client
LS_CLIENT = None
try:
    LS_CLIENT = Client()
except Exception:
    LS_CLIENT = None

# ===== LangChain / 모델 =====
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ===== RAPTOR 구성 요소 =====
import numpy as np
import pandas as pd
import umap
from sklearn.mixture import GaussianMixture


# =========================================================
# 0) PDF 로딩 & 캐싱 (nureongi.* 대체: 자체 구현)
# =========================================================
def find_pdf_files(root: Path, pattern: str = "**/*.pdf") -> List[Path]:
    """루트 경로에서 재귀적으로 PDF 파일을 찾는다."""
    return sorted([p for p in root.glob(pattern) if p.is_file()])

def load_single_pdf(path: Path, max_pages: Optional[int] = None) -> List[Document]:
    """단일 PDF를 로드하여 LangChain Document 리스트로 반환."""
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    # 페이지 수 제한
    if max_pages is not None:
        docs = [d for d in docs if (d.metadata or {}).get("page", 0) < max_pages]
    # 최소 메타데이터 정리
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata.setdefault("source", str(path.name))
    return docs

def load_cache(cache_path: Optional[Path]) -> Dict[str, List[Document]]:
    """피클 캐시 로드. 실패 시 빈 캐시."""
    if not cache_path:
        return {}
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_path: Optional[Path], cache: Dict[str, List[Document]]) -> None:
    """피클 캐시 저장."""
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

def load_all_pdfs(
    root: Path,
    pattern: str = "**/*.pdf",
    workers: int = 8,
    max_pages: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> List[Document]:
    """다수 PDF 병렬 로드 + 캐시 반영."""
    pdf_files = find_pdf_files(root, pattern)
    cache = load_cache(cache_path)

    todo = []
    for p in pdf_files:
        key = str(p.resolve())
        if key not in cache:
            todo.append(p)

    documents: List[Document] = []
    # 캐시에 있는 문서 먼저 적재
    for k in cache:
        documents.extend(cache[k])

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(load_single_pdf, p, max_pages): p for p in todo}
            for fut in as_completed(futures):
                p = futures[fut]
                key = str(p.resolve())
                try:
                    docs = fut.result()
                    cache[key] = docs
                    documents.extend(docs)
                except Exception as e:
                    print(f"❌ 로드 실패: {p} -> {e}")
        save_cache(cache_path, cache)
    return documents


# =========================================================
# 1) 텍스트 분할
# =========================================================
def split_docs(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 120) -> List[Document]:
    """RecursiveCharacterTextSplitter로 문서를 청크로 분할."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


# =========================================================
# 2) 임베딩 & LLM
# =========================================================
def make_embeddings(model_name: Optional[str] = None) -> OpenAIEmbeddings:
    """OpenAI 임베딩 생성."""
    model = model_name or os.getenv("MODEL_EMB", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, disallowed_special=())

def make_llm(model_name: Optional[str] = None, temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 챗 LLM 생성."""
    model = model_name or os.getenv("MODEL_LLM", "gpt-4.1-mini")
    return ChatOpenAI(model=model, temperature=temperature)


# =========================================================
# 3) RAPTOR: UMAP + GMM(BIC) 계층 클러스터 요약
# =========================================================
RANDOM_SEED = 42

def umap_global(emb: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int(max(2, (len(emb) - 1) ** 0.5))
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED).fit_transform(emb)

def umap_local(emb: np.ndarray, dim: int, n_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED).fit_transform(emb)

def optimal_k_bic(emb: np.ndarray, kmax: int = 50) -> int:
    kmax = min(kmax, len(emb))
    if kmax <= 1:
        return 1
    ks = np.arange(1, kmax + 1)
    bics = []
    for k in ks:
        gm = GaussianMixture(n_components=k, random_state=RANDOM_SEED).fit(emb)
        bics.append(gm.bic(emb))
    return int(ks[int(np.argmin(bics))])

def gmm_soft_labels(emb: np.ndarray, threshold: float = 0.1) -> (List[np.ndarray], int):
    k = optimal_k_bic(emb)
    gm = GaussianMixture(n_components=k, random_state=RANDOM_SEED).fit(emb)
    probs = gm.predict_proba(emb)
    labels = [np.where(p > threshold)[0] for p in probs]
    return labels, k

def hier_cluster_labels(emb: np.ndarray, dim: int = 10, thr: float = 0.1) -> List[np.ndarray]:
    """전역 → 로컬 계층형 클러스터 라벨."""
    if len(emb) <= dim + 1:
        return [np.array([0]) for _ in range(len(emb))]

    reduced_g = umap_global(emb, dim)
    g_labels, g_k = gmm_soft_labels(reduced_g, thr)

    all_local = [np.array([]) for _ in range(len(emb))]
    total = 0
    for gi in range(g_k):
        idxs = np.array([gi in labs for labs in g_labels])
        sub = emb[idxs]
        if len(sub) == 0:
            continue
        if len(sub) <= dim + 1:
            l_labels = [np.array([0]) for _ in range(len(sub))]
            l_k = 1
        else:
            reduced_l = umap_local(sub, dim)
            l_labels, l_k = gmm_soft_labels(reduced_l, thr)

        sub_indices = np.where(idxs)[0]
        for j in range(l_k):
            members = np.array([j in labs for labs in l_labels])
            for orig_idx in sub_indices[members]:
                all_local[int(orig_idx)] = np.append(all_local[int(orig_idx)], j + total)
        total += l_k
    return all_local

def summarize_by_cluster(texts: List[str], labels: List[np.ndarray], llm: ChatOpenAI, level: int) -> pd.DataFrame:
    """같은 클러스터 묶음을 요약."""
    rows = []
    for t, labs in zip(texts, labels):
        for lab in labs:
            rows.append({"text": t, "cluster": int(lab)})
    df = pd.DataFrame(rows)
    clusters = sorted(df["cluster"].unique().tolist())

    prompt = PromptTemplate.from_template(
        "다음 텍스트 묶음의 핵심을 한국어로 5문장 이내로 간결히 요약하세요. 숫자/고유명사는 유지하세요.\n\n{context}"
    )
    chain = prompt | llm | StrOutputParser()

    summaries = []
    for c in clusters:
        block = "\n\n---\n\n".join(df[df.cluster == c]["text"].tolist())
        summaries.append(chain.invoke({"context": block}))
    return pd.DataFrame({"cluster": clusters, "level": level, "summary": summaries})

def raptor_levels(texts: List[str], emb_model: OpenAIEmbeddings, llm: ChatOpenAI,
                  max_levels: int = 3, dim: int = 10, thr: float = 0.1) -> Dict[int, pd.DataFrame]:
    """RAPTOR: leaf 요약 → 요약을 다시 클러스터링/요약을 반복."""
    out: Dict[int, pd.DataFrame] = {}
    level = 1
    cur = texts
    while level <= max_levels and len(cur) > 1:
        emb = np.array(emb_model.embed_documents(cur))
        labs = hier_cluster_labels(emb, dim=dim, thr=thr)
        df = summarize_by_cluster(cur, labs, llm, level)
        out[level] = df
        cur = df["summary"].tolist()
        level += 1
    return out


# =========================================================
# 4) VectorStore (FAISS)
# =========================================================
def build_faiss(leaf_texts: List[str], raptor: Dict[int, pd.DataFrame], emb: OpenAIEmbeddings,
                index_dir: str = "faiss_index") -> FAISS:
    """leaf + 각 레벨 요약을 모두 인덱싱하여 FAISS 저장."""
    all_texts = list(leaf_texts)
    for lvl in sorted(raptor.keys()):
        all_texts.extend(raptor[lvl]["summary"].tolist())
    vs = FAISS.from_texts(texts=all_texts, embedding=emb)
    # 영속화
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(index_dir)
    return vs


# =========================================================
# 5) Retriever + Prompt + Chain
# =========================================================
def build_retriever(vectorstore: FAISS):
    """MMR 검색기로 리트리버 구성."""
    return vectorstore.as_retriever(search_type="mmr",
                                    search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.25})

def format_ctx(docs: List[Document]) -> str:
    """컨텍스트 블록 포맷 + 간단 출처 표기."""
    blocks = []
    for d in docs:
        md = d.metadata or {}
        title = md.get("title") or md.get("source") or "문서"
        page = md.get("page") or md.get("page_number") or "?"
        blocks.append(f"{d.page_content}\nCITE: [{title}:{page}]")
    return "\n\n---\n\n".join(blocks)

def build_rag_chain(retriever, llm: ChatOpenAI):
    """프롬프트 + LLM으로 RAG 체인 구성."""
    prompt = PromptTemplate.from_template(
        """역할: 공식 문서 기반 RAG 보조원.
규칙:
- 컨텍스트에 있는 내용만 사용. 추측 금지.
- 각 문장 끝에 CITE 표기 유지.
- 불필요한 수식어 없이 간결·정확.

출력 형식:
- 주제(1문장)
- 핵심(3~5 불릿, 각 항목 끝에 CITE)
- 시사점(0~3 불릿, 가능 시 CITE)

[질문]
{question}

[컨텍스트]
{context}

한국어로 답변."""
    )
    chain = (
        {"context": retriever | (lambda ds: format_ctx(ds)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# =========================================================
# 6) main
# =========================================================
def main():
    # ----- 경로/파라미터 -----
    data_root = Path(os.getenv("DATA_ROOT", "dataset"))   # PDF 루트 디렉터리
    cache_path = Path(os.getenv("PDF_CACHE", "cache/pdf_cache.pkl"))
    index_dir  = os.getenv("FAISS_DIR", "faiss_index")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_ovlp = int(os.getenv("CHUNK_OVERLAP", "120"))
    max_pages  = int(os.getenv("MAX_PAGES", "0")) or None  # 0이면 무제한

    # ----- 1) PDF 로드 -----
    documents = load_all_pdfs(
        root=data_root, pattern="**/*.pdf", workers=8, max_pages=max_pages, cache_path=cache_path
    )
    if not documents:
        print("⚠️ dataset 폴더에서 PDF를 찾지 못했습니다.")
        return
    print(f"[INFO] 로드된 원문 문서 수: {len(documents)}")

    # ----- 2) 분할 -----
    chunked = split_docs(documents, chunk_size=chunk_size, chunk_overlap=chunk_ovlp)
    print(f"[INFO] 청크 수: {len(chunked)}")

    # ----- 3) 모델 -----
    emb = make_embeddings()
    llm = make_llm()

    # ----- 4) RAPTOR 요약 레벨 생성 -----
    leaf_texts = [d.page_content for d in chunked]
    raptor = raptor_levels(leaf_texts, emb, llm, max_levels=3, dim=10, thr=0.1)
    print(f"[INFO] RAPTOR 레벨 생성: {list(raptor.keys())}")

    # ----- 5) FAISS 인덱스 구축 -----
    vectorstore = build_faiss(leaf_texts, raptor, emb, index_dir=index_dir)
    retriever = build_retriever(vectorstore)

    # ----- 6) RAG 체인 -----
    chain = build_rag_chain(retriever, llm)

    # ----- 7) 예시 질의 -----
    examples = [
        "문서 전체에서 가장 중요한 정책 포인트를 요약해줘.",
        "삼성전자의 생성형 AI 명칭과 발표일을 알려줘.",
    ]
    for q in examples:
        print(f"\n[Q] {q}")
        ans = chain.invoke(q)
        print(f"[A]\n{ans}")

    # ----- 8) LangSmith 로깅(선택) -----
    if LS_CLIENT:
        try:
            LS_CLIENT.create_run(
                name="raptor-rag-sample",
                inputs={"examples": examples},
                outputs={"answer_snippet": ans[:1000]},
                tags=["RAG", "RAPTOR", "FAISS"],
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
