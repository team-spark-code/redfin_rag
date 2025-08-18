
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import os
import socket
import json
from pathlib import Path

# LangChain vectorstores / embeddings
from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.schema import Document

# Clients
from qdrant_client import QdrantClient

# ----- helpers -----
def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def _to_texts_and_metas(
    docs: Sequence[Document] | None,
    texts: Optional[Sequence[str]] = None,
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if docs is not None:
        t = [d.page_content for d in docs]
        m = [dict(d.metadata or {}) for d in docs]
        return t, m
    if texts is None:
        raise ValueError("Either docs or texts must be provided")
    m = [dict(x) for x in (metadatas or [{}] * len(texts))]
    if len(m) != len(texts):
        raise ValueError("len(metadatas) must equal len(texts)")
    return list(texts), m

# ----- Qdrant builder -----
def _build_qdrant_from_texts(
    embedding: Embeddings,
    *, 
    collection_name: str,
    docs: Sequence[Document] | None = None,
    texts: Optional[Sequence[str]] = None,
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    texts_, metas_ = _to_texts_and_metas(docs, texts, metadatas)

    # 1) env 우선
    url = url or os.getenv("QDRANT_URL")
    api_key = api_key or os.getenv("QDRANT_API_KEY")

    if url:
        client = QdrantClient(url=url, api_key=api_key)
        client.get_collections()  # connectivity check
        return LCQdrant.from_texts(
            texts=texts_,
            embedding=embedding,
            metadatas=metas_,
            client=client,
            collection_name=collection_name,
        ), {"backend": "qdrant", "url": url, "auth": bool(api_key), "source": "env"}

    # 2) localhost 스캔
    if _port_open("127.0.0.1", 6333):
        local = "http://localhost:6333"
        client = QdrantClient(url=local)
        client.get_collections()
        return LCQdrant.from_texts(
            texts=texts_,
            embedding=embedding,
            metadatas=metas_,
            client=client,
            collection_name=collection_name,
        ), {"backend": "qdrant", "url": local, "auth": False, "source": "localhost"}

    raise ConnectionError("Qdrant not reachable (env URL missing and localhost:6333 closed)")

# ----- FAISS builder -----
def _build_faiss_from_texts(
    embedding: Embeddings,
    *, 
    docs: Sequence[Document] | None = None,
    texts: Optional[Sequence[str]] = None,
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    vectors: Optional[Sequence[Sequence[float]]] = None,
    persist_dir: Optional[Path] = None,
):
    texts_, metas_ = _to_texts_and_metas(docs, texts, metadatas)

    if vectors is not None:
        vs = FAISS.from_embeddings(
            text_embeddings=zip(texts_, vectors),
            embedding=embedding,
            metadatas=metas_,
        )
    else:
        # Let FAISS embed internally via embedding passed
        vs = FAISS.from_texts(texts_, embedding=embedding, metadatas=metas_)

    info = {"backend": "faiss", "source": "in-memory"}
    if persist_dir:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(persist_dir))
        info.update({"persist_path": str(persist_dir), "persisted": True})
    return vs, info

# ----- Public: auto Qdrant -> FAISS fallback -----
@dataclass
class VSReturn:
    backend: str
    vectorstore: Any
    details: Dict[str, Any]

def auto_qdrant_faiss(
    embedding: Embeddings,
    *, 
    collection_name: str,
    docs: Sequence[Document] | None = None,
    texts: Optional[Sequence[str]] = None,
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    # If you already computed embeddings, pass here to avoid re-embedding for FAISS path.
    vectors: Optional[Sequence[Sequence[float]]] = None,
    faiss_dir: Optional[Union[str, Path]] = None,
    # Override Qdrant connection if desired
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    quiet: bool = True,
) -> VSReturn:
    """
    Try Qdrant first. If it fails, fall back to FAISS.
    - Supports either docs or (texts, metadatas).
    - If FAISS path taken and `faiss_dir` provided, it persists index to disk.
    """
    last_err = None
    try:
        vs, info = _build_qdrant_from_texts(
            embedding,
            collection_name=collection_name,
            docs=docs, texts=texts, metadatas=metadatas,
            url=qdrant_url, api_key=qdrant_api_key,
        )
        if not quiet:
            print(f"[auto_vstore] Qdrant connected: {json.dumps(info)}")
        return VSReturn("qdrant", vs, info)
    except Exception as e:
        last_err = e
        if not quiet:
            print(f"[auto_vstore] Qdrant unavailable -> falling back to FAISS ({e})")

    # Fallback to FAISS
    vs, info = _build_faiss_from_texts(
        embedding,
        docs=docs, texts=texts, metadatas=metadatas,
        vectors=vectors, persist_dir=Path(faiss_dir) if faiss_dir else None,
    )
    if not quiet:
        print(f"[auto_vstore] Using FAISS: {json.dumps(info)}")
    # Attach root-cause for observability
    info["qdrant_error"] = str(last_err)
    return VSReturn("faiss", vs, info)

# ----- Retriever helper (MMR default to match common config) -----
def as_retriever(vs: Any, *, k: int = 8, fetch_k: int = 60, lambda_mult: float = 0.25, search_type: str = "mmr"):
    return vs.as_retriever(search_type=search_type, search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})
