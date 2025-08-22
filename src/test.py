#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

NEWS_API_URL = os.getenv("NEWS_API_URL", "http://127.0.0.1:8000/news")

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from nureongi.indexing import build_index
from nureongi.chain import build_rag_chain

def make_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name=os.getenv("EMB_MODEL","BAAI/bge-base-en-v1.5"),
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="Represent this sentence for searching relevant passages:",
        embed_instruction="Represent this document for retrieval:",
    )

def main():
    emb = make_embeddings()
    vs, info = build_index(
        NEWS_API_URL, emb,
        chunk_size=int(os.getenv("CHUNK_SIZE","500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP","120")),
        index_mode="summary_only",
        use_raptor=True,    # 의존성 없으면 False
        distance="cosine",
    )
    print("[INDEX]", info)

    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k":8, "fetch_k":60, "lambda_mult":0.25})
    chain = build_rag_chain(retriever, persona="ai_industry_professional")  # 또는 다른 별칭/슬러그

    q = "Give me a 5-bullet executive summary of today's top AI news."
    print("[Q]", q)
    print("[A]", chain.invoke(q)[:1500])

if __name__ == "__main__":
    main()
