# pip install -U umap-learn langchain langchain-openai langchain-community sentence-transformers qdrant-client
from dotenv import load_dotenv ; load_dotenv()
from typing import List, Dict, Any, Iterable, Optional
import os, warnings, pickle, json
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# pip install pymupdf faiss-gpu
from langsmith import Client
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from nureongi import (
    find_pdf_files, load_single_pdf, load_all_pdfs, load_cache, save_cache,
    build_routed_rag_chain, choose_prompt, build_routed_rag_chain,
    auto_qdrant_faiss, as_retriever
)

"""
대용량 PDF를 로딩하려면 어떻게 해야 할까?
1. PDF가 많으면 로딩이 느려질 수 있음
    - 병렬 처리 (concurrent.futures.ThreadPoolExecutor)
    - 문서별 로드 예외 처리 (깨진 PDF 무시)
    - 전처리 후 캐싱 (한번 읽은 문서는 pickle로 저장)
"""

# ===== 0. 환경 설정 =====
warnings.filterwarnings("ignore")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["LANGSMITH_TRACING_V2"] = os.getenv("LANGSMITH_TRACING_V2", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "nureongi-rag")

client = Client()

# ===== 1. 문서 로드 =====
documents = load_all_pdfs(
    root=Path("dataset"),
    pattern="**/*.pdf",
    workers=8,
    max_pages=None,
    cache_path=Path("cache/pdf_cache.pkl")
)

# ===== 2. 텍스트 분할 =====
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
docs = splitter.split_documents(documents)

# ===== 3. 임베딩 생성 =====
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ===== 4. 벡터스토어 구성 (Qdrant → 실패 시 FAISS 폴백) =====
# - 기존 FAISS.from_embeddings() 대신 자동 폴백 모듈 사용
# - vectors는 FAISS 폴백 경로에서만 활용되어 재임베딩 방지
vectors = []
batch_size = 100    # 100개씩 나눠서 요청
for i in tqdm(range(0, len(docs), batch_size)):
    batch = docs[i:i+batch_size]
    vectors.extend(embeddings.embed_documents([d.page_content for d in batch]))

vsr = auto_qdrant_faiss(
    embedding=embeddings,
    collection_name="gov_docs",     # 컬렉션/인덱스 식별자
    docs=docs,                      # split 후 Document 리스트
    vectors=vectors,                # Qdrant 경로에선 무시, FAISS 폴백에서만 사용
    faiss_dir="faiss_index",        # 폴백 시 인덱스 영속화 경로
    # qdrant_url="https://<your-endpoint>",   # 필요 시 직접 지정
    # qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    quiet=False
)
vectorstore = vsr.vectorstore
print(f"[VectorBackend] {vsr.backend} / details={vsr.details}")

# ===== 5. Retriever 설정 (MMR + 기본 k 조정) =====
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.25},
)

# ===== 6. 컨텍스트 포맷터 추가 (출처 태그 강제) =====
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

# ===== 7. Frontier Lab 스타일 프롬프트 =====
prompt = PromptTemplate.from_template(
    """역할: 정부 공식문서 기반 RAG 보조원.
규칙:
- 컨텍스트에 있는 내용만 사용, 추측 금지.
- 각 불릿/문장 끝에 반드시 CITE 태그에 있는 출처 표기 포함.
- 불필요한 수식어 배제, 간결·정확.

출력 형식:
- 주제(1문장)
- 핵심(3~5 불릿, 각 불릿 말미에 출처)
- 시사점(2~3 불릿, 가능하면 출처)
- 근거 없음/쟁점(있으면 0~2 불릿)

[질문]
{question}

[컨텍스트]
{context}

한국어로만 답변."""
)

# ===== 8. RAG 체인 (LLM 파라미터 확정 + 컨텍스트 포맷터 적용) =====
rag_chain = build_routed_rag_chain(retriever)
print(rag_chain.invoke({"question": "행안부 재난관리기금 집행 지침 최신 개정 포인트 요약", "persona": "auto"}))
print(rag_chain.invoke({"question": "타 부처 유사 제도 비교와 정책 제언", "persona": "gov-policy"}))
print(rag_chain.invoke({"question": "최신 선행연구와 시사점 정리", "persona": "academic_researcher"}))

# ===== 9. 실행 및 LangSmith 메타 로깅 (간단) =====
query = "이 문서의 핵심 내용을 요약해줘"
result = rag_chain.invoke({"question": query, "persona": "auto"})
print(result)

# 선택: 실행 요약을 LangSmith에 별도 런으로 남김
try:
    client.create_run(
        name="document-summary-run",
        inputs={"query": query},
        outputs={"answer": result[:1000]},
        tags=["RAG", "summary", "frontier-lab"],
    )
except Exception:
    pass