#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# 패키지 설치 안내(가상환경에서 실행 권장)
# -----------------------------------------------------------------------------
# python -m pip install -U pip setuptools wheel
# python -m pip install -U umap-learn scikit-learn numpy pandas requests
# python -m pip install -U langchain langchain-openai langchain-community langsmith
# python -m pip install -U pymupdf faiss-cpu
# (Windows/Apple Silicon에서 faiss 설치 이슈 시)
# python -m pip install "faiss-cpu==1.8.0.post3"
#
# python -m pip install -U langchain-google-genai google-generativeai sentence-transformers
#
# * OpenAI 임베딩/LLM을 쓰므로 .env에 OPENAI_API_KEY 필수
# * 로컬 임베딩을 쓰고 싶으면 HuggingFace 계열로 교체 가능(옵션)
# =============================================================================

# ===== 0. 환경 설정 =====
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

from typing import List, Dict, Any, Optional
import os, warnings, json, re, html
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

warnings.filterwarnings("ignore")

# OpenAI/LangSmith 환경값(없으면 None 대입 금지)
_openai = os.getenv("OPENAI_API_KEY")
if not _openai:
    raise RuntimeError("OPENAI_API_KEY가 필요합니다(.env 설정).")
os.environ["OPENAI_API_KEY"] = _openai

if os.getenv("LANGSMITH_TRACING_V2") is None:
    os.environ["LANGSMITH_TRACING_V2"] = "true"
for k in ["HUGGINGFACEHUB_API_TOKEN","LANGSMITH_ENDPOINT","LANGSMITH_API_KEY","LANGSMITH_PROJECT"]:
    v = os.getenv(k)
    if v:
        os.environ[k] = v

# 데이터 경로/설정
DATA_ROOT = Path(os.getenv("DATA_ROOT", Path(__file__).resolve().parent / "dataset"))
PDF_CACHE = Path(os.getenv("PDF_CACHE", Path(__file__).resolve().parent / "cache/pdf_cache.pkl"))
FAISS_DIR = Path(os.getenv("FAISS_DIR", Path(__file__).resolve().parent / "faiss_index"))
NEWS_API_URL = os.getenv("NEWS_API_URL", "http://192.168.0.123:8000/news")

# ===== 0-1. 라이브러리 임포트 =====
from langsmith import Client
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 누렁이 유틸(그대로 사용)
from nureongi import (
    find_pdf_files, load_single_pdf, load_cache, save_cache,
    build_routed_rag_chain, as_retriever
)

# ===== 0-2. RAPTOR 의존 =====
import numpy as np
import pandas as pd
import umap
from sklearn.mixture import GaussianMixture

LS_CLIENT = None
try:
    LS_CLIENT = Client()
except Exception:
    pass


# =============================================================================
# 1) 데이터 로드: PDF(병렬+캐시) + 뉴스(JSON)
# =============================================================================

def load_all_pdfs(
    root: Path,
    pattern: str = "**/*.pdf",
    workers: int = 8,
    max_pages: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> List[Document]:
    """대용량 PDF를 병렬 로드하고, 실패/성공 모두 캐시에 반영."""
    pdf_files = find_pdf_files(root, pattern)
    cache = load_cache(cache_path)

    todo = []
    for p in pdf_files:
        key = str(p.resolve())
        if key not in cache:
            todo.append(p)

    documents: List[Document] = []
    for k in cache:  # 캐시 적재분
        documents.extend(cache[k])

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(load_single_pdf, p, max_pages): p for p in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading PDFs"):
                p = futures[fut]
                key = str(p.resolve())
                try:
                    docs = fut.result()
                    cache[key] = docs
                    documents.extend(docs)
                except Exception as e:
                    print(f"❌ 실패: {p} -> {e}")
        save_cache(cache_path, cache)

    return documents

def _strip_html(s: str) -> str:
    """간단 HTML 제거 및 엔티티 언이스케이프."""
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return html.unescape(s)

def load_news_json(url: str) -> List[Document]:
    """뉴스 JSON(서비스 스크랩 API)을 Document로 변환."""
    docs: List[Document] = []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        items = resp.json()
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
    return docs


# =============================================================================
# 2) 텍스트 분할
# =============================================================================
def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 120) -> List[Document]:
    """PDF/뉴스 혼합 문서를 동일 규칙으로 청크 분할."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


# =============================================================================
# 3) 임베딩 & LLM
# =============================================================================
# ----------------------------- 모델 선택기 -----------------------------
def select_llm(backend: Optional[str] = None, temperature: float = 0.0):
    """
    LLM_BACKEND: openai | gemini | ollama(로컬, 선택)
    - openai  → ChatOpenAI (OPENAI_API_KEY 필요)
    - gemini  → ChatGoogleGenerativeAI (GOOGLE_API_KEY 필요)
    - ollama  → 로컬 LLM (옵션)  ※ 필요 시: from langchain_community.chat_models import ChatOllama
    """
    backend = (backend or os.getenv("LLM_BACKEND", "openai")).lower()
    if backend == "openai":
        return ChatOpenAI(model=os.getenv("MODEL_LLM", "gpt-4.1-mini"), temperature=temperature)
    elif backend == "gemini":
        # 예: GEMINI_MODEL=gemini-1.5-flash 또는 gemini-1.5-pro
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY가 필요합니다(.env).")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    elif backend == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"), temperature=temperature)
    else:
        raise ValueError(f"지원하지 않는 LLM_BACKEND: {backend}")

def select_embeddings(backend: Optional[str] = None):
    """
    EMB_BACKEND: openai | google | hf
    - openai → OpenAIEmbeddings (OPENAI_API_KEY)
    - google → GoogleGenerativeAIEmbeddings (GOOGLE_API_KEY, text-embedding-004)
    - hf     → HuggingFaceBgeEmbeddings (로컬, BAAI/bge-m3 권장)
    """
    backend = (backend or os.getenv("EMB_BACKEND", "openai")).lower()
    if backend == "openai":
        return OpenAIEmbeddings(model=os.getenv("MODEL_EMB", "text-embedding-3-small"))
    elif backend == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY가 필요합니다(.env).")
        return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMB", "text-embedding-004"))
    elif backend == "hf":
        return HuggingFaceBgeEmbeddings(
            model_name=os.getenv("EMB_MODEL", "BAAI/bge-m3"),
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages:",
            embed_instruction="Represent this sentence for retrieving relevant documents:",
        )
    else:
        raise ValueError(f"지원하지 않는 EMB_BACKEND: {backend}")
# ----------------------------------------------------------------------

# 위 설정한 모델 선택기 불러오기
def make_embeddings():
    """백엔드 스위치형 임베딩 선택기 래퍼"""
    return select_embeddings()

def make_llm(temp: float = 0.0):
    """백엔드 스위치형 LLM 선택기 래퍼"""
    return select_llm(temperature=temp)


# =============================================================================
# 4) RAPTOR: UMAP+GMM(BIC) 계층 클러스터 요약(레벨 업)
# =============================================================================
RANDOM_SEED = 42

def _umap(emb: np.ndarray, dim: int, n_neighbors: int, metric: str = "cosine"):
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED).fit_transform(emb)

def _optimal_k_bic(emb: np.ndarray, kmax: int = 50) -> int:
    kmax = min(kmax, len(emb))
    if kmax <= 1:
        return 1
    ks = np.arange(1, kmax + 1)
    bics = []
    for k in ks:
        gm = GaussianMixture(n_components=k, random_state=RANDOM_SEED).fit(emb)
        bics.append(gm.bic(emb))
    return int(ks[int(np.argmin(bics))])

def _gmm_soft_labels(emb: np.ndarray, threshold: float = 0.1) -> (List[np.ndarray], int):
    k = _optimal_k_bic(emb)
    gm = GaussianMixture(n_components=k, random_state=RANDOM_SEED).fit(emb)
    probs = gm.predict_proba(emb)
    labels = [np.where(p > threshold)[0] for p in probs]
    return labels, k

def hier_cluster_labels(emb: np.ndarray, dim: int = 10, thr: float = 0.1) -> List[np.ndarray]:
    """전역 → 로컬 계층 라벨(간결 구현)."""
    if len(emb) <= dim + 1:
        return [np.array([0]) for _ in range(len(emb))]
    # 전역 축소(이웃은 대략 sqrt(N))
    reduced_g = _umap(emb, dim=dim, n_neighbors=max(2, int((len(emb)-1)**0.5)))
    g_labels, g_k = _gmm_soft_labels(reduced_g, thr)

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
            reduced_l = _umap(sub, dim=dim, n_neighbors=10)
            l_labels, l_k = _gmm_soft_labels(reduced_l, thr)
        sub_indices = np.where(idxs)[0]
        for j in range(l_k):
            members = np.array([j in labs for labs in l_labels])
            for orig_idx in sub_indices[members]:
                all_local[int(orig_idx)] = np.append(all_local[int(orig_idx)], j + total)
        total += l_k
    return all_local

def summarize_by_cluster(texts: List[str], labels: List[np.ndarray], llm: ChatOpenAI, level: int) -> pd.DataFrame:
    """클러스터 묶음을 LLM으로 요약."""
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
    """RAPTOR: leaf 요약 → 요약을 다시 클러스터링/요약 반복."""
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


# =============================================================================
# 5) VectorStore(FAISS) 구축: leaf + 요약 전체
# =============================================================================
def build_faiss_with_metadata(leaf_docs: List[Document], raptor: Dict[int, pd.DataFrame],
                              emb: OpenAIEmbeddings, index_dir: Path) -> FAISS:
    """leaf Document + 레벨 요약을 Document로 구성해 메타데이터 보존 후 인덱싱."""
    all_docs: List[Document] = []
    all_docs.extend(leaf_docs)
    for lvl in sorted(raptor.keys()):
        for s in raptor[lvl]["summary"].tolist():
            all_docs.append(Document(page_content=s, metadata={"type": "raptor_summary", "level": lvl}))
    vs = FAISS.from_documents(all_docs, embedding=emb)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    return vs


# =============================================================================
# 6) Retriever + 프롬프트 + 체인(누렁이 build_routed_rag_chain 활용)
# =============================================================================
def build_chain(vectorstore: FAISS, llm: ChatOpenAI):
    """MMR 리트리버 + 누렁이 라우팅 체인."""
    retriever = vectorstore.as_retriever(search_type="mmr",
                                         search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.25})
    # 누렁이 체인은 내부적으로 페르소나/프롬프트 라우팅을 처리
    chain = build_routed_rag_chain(retriever)
    return chain


# =============================================================================
# 7) main
# =============================================================================
def main():
    print(f"[DEBUG] CWD={Path.cwd()}")
    print(f"[DEBUG] DATA_ROOT={DATA_ROOT.resolve()}  FAISS_DIR={FAISS_DIR.resolve()}")
    print(f"[DEBUG] NEWS_API_URL={NEWS_API_URL}")

    # --- PDF 로드 ---
    pdf_docs = load_all_pdfs(
        root=DATA_ROOT,
        pattern="**/*.pdf",
        workers=8,
        max_pages=None,
        cache_path=PDF_CACHE,
    )
    print(f"[INFO] PDF 문서(원본) 수: {len(pdf_docs)}")

    # --- 뉴스 JSON 로드 ---
    news_docs = load_news_json(NEWS_API_URL)
    print(f"[INFO] 뉴스 문서 수: {len(news_docs)}")

    if not pdf_docs and not news_docs:
        raise SystemExit("로드된 문서가 없습니다(PDF/뉴스 모두 0). 경로/API를 확인하세요.")

    # --- 합치기 & 청크 분할 ---
    all_raw_docs = pdf_docs + news_docs
    chunked_docs = split_documents(all_raw_docs, chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
                                   chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")))
    print(f"[INFO] 청크 수: {len(chunked_docs)}")
    if not chunked_docs:
        raise SystemExit("청크가 생성되지 않았습니다. 문서의 인코딩/로더를 확인하세요.")

    # --- 임베딩/LLM ---
    embeddings = make_embeddings()
    llm = make_llm(temp=0.0)

    # --- RAPTOR 레벨 생성 ---
    leaf_texts = [d.page_content for d in chunked_docs]
    raptor = raptor_levels(leaf_texts, embeddings, llm, max_levels=int(os.getenv("RAPTOR_LEVELS", "3")), dim=10, thr=0.1)
    print(f"[INFO] RAPTOR 레벨: {list(raptor.keys()) or []}")

    # 임베딩 백엔드별로 인덱스 디렉토리 분리
    emb_tag = os.getenv("EMB_BACKEND", "openai").lower()
    faiss_dir_effective = FAISS_DIR.parent / f"{FAISS_DIR.name}_{emb_tag}"
    
    # --- VectorStore(FAISS) 구축 ---
    vectorstore = build_faiss_with_metadata(chunked_docs, raptor, embeddings, faiss_dir_effective)
    print(f"[INFO] VectorStore 구축 완료: {faiss_dir_effective.resolve()}")

    # --- 체인 구성(누렁이 라우팅 체인) ---
    chain = build_chain(vectorstore, llm)

    # --- 예시 질의 ---
    examples = [
        {"q": "문서 전체에서 핵심 정책 포인트를 요약해줘.", "persona": "auto"},
        {"q": "방금 들어온 AI 뉴스의 요지를 요약하고, 기업/연구 관점 시사점을 정리해줘.", "persona": "news_brief"},
        {"q": "삼성전자의 생성형 AI 명칭과 발표일은?", "persona": "auto"},
    ]
    for ex in examples:
        print(f"\n[Q] {ex['q']}")
        ans = chain.invoke({"question": ex["q"], "persona": ex["persona"]})
        print(f"[A]\n{ans}")

    # --- LangSmith(선택) ---
    if LS_CLIENT:
        try:
            LS_CLIENT.create_run(
                name="raptor-rag-run",
                inputs={"examples": [e["q"] for e in examples]},
                outputs={"last_answer": ans[:1000] if isinstance(ans, str) else str(ans)[:1000]},
                tags=["RAG", "RAPTOR", "FAISS", "news-json"],
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
