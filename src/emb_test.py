#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
영어 전용 RAG 비교 실험(임베딩 3~4종)
- 기본: BGE-small, BGE-base, OpenAI text-embedding-3-small
- 옵션: voyage-3.5-lite (최근 좋은 성능 유료 모델) 추가
- 목적: 동일 조건에서 순수 임베딩 검색력(리트리벌) 비교

예)
  python emb_test.py --topk 10 --max_q 300 \
    --models BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5 text-embedding-3-small
  python emb_test.py --models voyage-3.5-lite BAAI/bge-base-en-v1.5 text-embedding-3-small
"""
import os, time, math, json, hashlib, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from dotenv import load_dotenv, find_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings

import re, html
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.embeddings import Embeddings  # 최신
except ImportError:
    from langchain.embeddings.base import Embeddings

# # test.py의 로더/청킹 재사용(동일 폴더에 test.py 존재 가정)
# from test import load_news_json, split_documents

load_dotenv(find_dotenv(), override=False)

# ---------------------- 임베딩 어댑터들 ----------------------
VOYAGE_API = "https://api.voyageai.com/v1/embeddings"

def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return html.unescape(s)

def load_news_json(url: str):
    docs = []
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        items = r.json()
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
            "type":"news","source":it.get("source"),"link":it.get("link"),
            "published":it.get("published"),"authors":it.get("authors"),
            "tags":it.get("tags"),"title":title,
        }
        if body:
            docs.append(Document(page_content=body, metadata=md))
    return docs

def split_documents(documents, chunk_size=500, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

class VoyageEmbeddings(Embeddings):
    """LangChain Embeddings 규약 준수 + callable 호환(__call__)"""
    def __init__(self, model: str = "voyage-3.5-lite", api_key: str | None = None, timeout: int = 30):
        self.model = model
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError("VOYAGE_API_KEY가 필요합니다(.env)")
        self.timeout = timeout

    def _post(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}
        r = requests.post(VOYAGE_API, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Voyage API 실패: {r.status_code} {r.text[:200]}")
        data = r.json().get("data", [])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out, B = [], 128
        for i in range(0, len(texts), B):
            out.extend(self._post(texts[i:i+B]))
        X = np.array(out, dtype=np.float32)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # 코사인 공정성: L2 정규화
        return X.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    # 일부 VectorStore 구현이 callable을 기대하는 경우 대비
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)
    
def make_bge(name: str) -> HuggingFaceBgeEmbeddings:
    """BGE 영문 전용 인스트럭션 + L2 normalize."""
    return HuggingFaceBgeEmbeddings(
        model_name=name,
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="Represent this sentence for searching relevant passages:",
        embed_instruction="Represent this document for retrieval:"
    )

def make_openai(name: str) -> OpenAIEmbeddings:
    """OpenAI 임베딩. LangChain FAISS에서 cosine 사용으로 정규화 불필요."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 필요합니다(.env)")
    return OpenAIEmbeddings(model=name)

def make_embedder(model_name: str):
    if model_name.startswith("voyage"):
        return VoyageEmbeddings(model=model_name)
    elif model_name.startswith("text-embedding-3"):
        return make_openai(model_name)
    elif model_name.startswith("BAAI/bge-"):
        return make_bge(model_name)
    else:
        raise ValueError(f"알 수 없는 모델명: {model_name}")

# ---------------------- 공통 유틸 ----------------------
def _doc_id(md: Dict[str, Any]) -> str:
    link = (md or {}).get("link") or ""
    title = (md or {}).get("title") or ""
    if link:
        return link
    return "title:" + hashlib.md5(title.encode("utf-8")).hexdigest()

def _assign_doc_ids(docs: List[Document]) -> List[Document]:
    out = []
    for d in docs:
        md = dict(d.metadata or {})
        md["doc_id"] = _doc_id(md)
        out.append(Document(page_content=d.page_content, metadata=md))
    return out

def _build_queries_from_news(news_docs: List[Document]) -> List[Tuple[str, str]]:
    """Q=제목, GT=해당 문서 doc_id."""
    pairs = []
    for d in news_docs:
        title = (d.metadata or {}).get("title") or ""
        if not title.strip():
            continue
        pairs.append((title.strip(), d.metadata["doc_id"]))
    return pairs

def _dedup_rank_by_doc(results: List[Tuple[Document, float]]) -> List[Tuple[str, float]]:
    """청크 결과를 문서(doc_id) 레벨로 상향."""
    seen, out = set(), []
    for d, s in results:
        did = (d.metadata or {}).get("doc_id")
        if did and did not in seen:
            seen.add(did)
            out.append((did, s))
    return out

# ---------------------- 지표 ----------------------
def _recall_at_k(ranks: List[int], k: int) -> float:
    return float(np.mean([1.0 if (r is not None and r < k) else 0.0 for r in ranks]))

def _mrr_at_k(ranks: List[int], k: int) -> float:
    vals = []
    for r in ranks:
        if r is None or r >= k:
            vals.append(0.0)
        else:
            vals.append(1.0 / (r + 1))
    return float(np.mean(vals))

def _ndcg_at_k(ranks: List[int], k: int) -> float:
    vals = []
    for r in ranks:
        if r is None or r >= k:
            vals.append(0.0)
        else:
            vals.append(1.0 / math.log2(r + 2))
    return float(np.mean(vals))

def _hit_at_k(ranks: List[int], k: int) -> float:
    return _recall_at_k(ranks, k)

@dataclass
class RunStats:
    model: str
    topk: int
    n_query: int
    recall: float
    ndcg: float
    mrr: float
    hit: float
    embed_secs: float
    embed_speed_doc_per_s: float
    search_secs: float
    p50_ms: float
    p95_ms: float
    index_bytes: int
    dim: int

# ---------------------- 실행 파이프라인 ----------------------
def run_one(
    model_name: str,
    chunked_docs: List[Document],
    queries: List[Tuple[str, str]],
    topk: int = 10,
) -> RunStats:
    """
    1) 문서 임베딩 → FAISS 인덱스(코사인) 구축
    2) 쿼리 검색/지표
    3) 속도/지연/인덱스 메모리 추정
    """
    emb = make_embedder(model_name)

    t0 = time.perf_counter()
    vs = FAISS.from_documents(
        chunked_docs,
        embedding=emb,
        distance_strategy=DistanceStrategy.COSINE
    )
    embed_secs = time.perf_counter() - t0
    embed_speed = len(chunked_docs) / (embed_secs + 1e-9)

    # dim 추정용 한 번 호출
    sample_vec = emb.embed_query("healthcare ai news")
    dim = len(sample_vec)
    index_bytes = len(chunked_docs) * dim * 4  # float32 기준

    # 검색
    ranks, lat_ms = [], []
    for qtext, gt_doc_id in tqdm(queries, desc=f"Search@{model_name}", unit="q"):
        t1 = time.perf_counter()
        res = vs.similarity_search_with_score(qtext, k=topk)
        lat_ms.append((time.perf_counter() - t1) * 1000.0)
        ranked = _dedup_rank_by_doc(res)

        found = None
        for rank, (doc_id, _) in enumerate(ranked):
            if doc_id == gt_doc_id:
                found = rank
                break
        ranks.append(found)

    recall = _recall_at_k(ranks, topk)
    ndcg   = _ndcg_at_k(ranks, topk)
    mrr    = _mrr_at_k(ranks, topk)
    hit    = _hit_at_k(ranks, topk)

    p50 = float(np.percentile(lat_ms, 50))
    p95 = float(np.percentile(lat_ms, 95))
    search_secs = float(np.sum(lat_ms) / 1000.0)

    return RunStats(
        model=model_name,
        topk=topk,
        n_query=len(queries),
        recall=recall,
        ndcg=ndcg,
        mrr=mrr,
        hit=hit,
        embed_secs=embed_secs,
        embed_speed_doc_per_s=embed_speed,
        search_secs=search_secs,
        p50_ms=p50,
        p95_ms=p95,
        index_bytes=index_bytes,
        dim=dim,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_q", type=int, default=300, help="평가에 사용할 최대 질의 수(무작위 샘플)")
    ap.add_argument(
        "--models", nargs="+",
        default=[
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "text-embedding-3-small"  # OpenAI
            "voyage-3.5-lite",      # 필요 시 추가
        ]
    )
    ap.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", "500")))
    ap.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "120")))
    ap.add_argument("--news_url", type=str, default=os.getenv("NEWS_API_URL", ""))
    ap.add_argument("--out", type=str, default="emb_results.csv")
    args = ap.parse_args()

    if not args.news_url:
        raise SystemExit("NEWS_API_URL가 .env 또는 --news_url 로 설정되어야 합니다.")

    # 0) 뉴스 로드 → Document
    news_docs = load_news_json(args.news_url)
    if not news_docs:
        raise SystemExit("뉴스 문서가 0건입니다. NEWS_API_URL 또는 API 응답을 확인하세요.")

    # 1) 문서 ID 부여 + 청킹(동일 규칙)
    news_docs = _assign_doc_ids(news_docs)
    chunked_docs = split_documents(news_docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # 2) 질의 세트(Q=제목, GT=문서 doc_id)
    all_q = _build_queries_from_news(news_docs)
    if len(all_q) > args.max_q:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_q), size=args.max_q, replace=False)
        eval_q = [all_q[i] for i in idx]
    else:
        eval_q = all_q

    # 3) 실행
    rows = []
    for m in args.models:
        stats = run_one(m, chunked_docs, eval_q, topk=args.topk)
        rows.append(stats.__dict__)
        print(f"[{m}] n={stats.n_query}  R@{args.topk}={stats.recall:.3f}  nDCG@{args.topk}={stats.ndcg:.3f} "
              f"MRR@{args.topk}={stats.mrr:.3f}  p50={stats.p50_ms:.1f}ms  p95={stats.p95_ms:.1f}ms "
              f"embed={stats.embed_secs:.1f}s({stats.embed_speed_doc_per_s:.1f} doc/s) "
              f"dim={stats.dim} index≈{stats.index_bytes/1e6:.1f}MB")

    df = pd.DataFrame(rows)
    df["index_size_MB_est"] = df["index_bytes"] / (1024*1024)
    df = df.drop(columns=["index_bytes"])
    df.to_csv(args.out, index=False)
    print("\n=== 결과 요약 ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
