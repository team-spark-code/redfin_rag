from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
from pathlib import Path

from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

_DEF_EMB = os.environ.get("EMB_MODEL", "BAAI/bge-base-en-v1.5")
_DEFAULT_CHROMA_DIR = os.environ.get("CHROMA_DIR", "./.chroma")  # 영속 디렉터리

# ---------------- Embeddings ----------------
def build_embedder(model_name: str | None = None):
    model = model_name or _DEF_EMB
    return HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"normalize_embeddings": True},  # cosine/inner-product 품질 ↑
        multi_process=False,
    )

# ---------------- Helpers ----------------
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

def _map_distance(distance: str) -> str:
    d = (distance or "cosine").lower()
    if d in ("cos", "cosine"):
        return "cosine"
    if d in ("l2", "euclid", "euclidean"):
        return "l2"
    if d in ("dot", "ip", "inner", "max_inner_product"):
        return "ip"
    raise ValueError(f"Unsupported distance: {distance}")

# ---------------- Metadata filter helper ----------------
def _sanitize_metadatas(metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in metas:
        mm: Dict[str, Any] = {}
        for k, v in (m or {}).items():
            k = str(k)  # 키를 문자열화
            if v is None:
                continue  # 또는 mm[k] = "" 로 대체해도 무방
            if isinstance(v, (str, int, float, bool)):
                mm[k] = v if not isinstance(v, str) else v[:2000]  # 과도한 길이 컷
            else:
                # 리스트/딕트/기타 타입 → 문자열(필요 시 json.dumps로 바꿔도 됨)
                s = str(v)
                mm[k] = s[:2000]
        out.append(mm)
    return out

# ---------------- Chroma builder ----------------
def _build_chroma_from_texts(
    embedding: Embeddings,
    *,
    collection_name: str,
    docs: Sequence[Document] | None = None,
    texts: Optional[Sequence[str]] = None,
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    persist_dir: Optional[Union[str, Path]] = None,
    distance: str = "cosine",
):
    texts_, metas_ = _to_texts_and_metas(docs, texts, metadatas)
    metas_ = _sanitize_metadatas(metas_)

    persist_dir = str(persist_dir or _DEFAULT_CHROMA_DIR)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    space = _map_distance(distance)  # "cosine" | "l2" | "ip"

    vs = Chroma.from_texts(
        texts=texts_,
        embedding=embedding,
        metadatas=metas_,
        collection_name=collection_name,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": space},
        client_settings=Settings(anonymized_telemetry=False),
    )
    # ⛔️ Chroma 0.4+는 자동 영속. persist() 호출 제거.
    info = {"backend": "chroma", "persist_directory": persist_dir, "distance": space}
    return vs, info

# ---------------- Public API (Chroma only) ----------------
@dataclass
class VSReturn:
    backend: str
    vectorstore: Any
    details: Dict[str, Any]

def create_chroma_store(
    embedding: Embeddings,
    *,
    collection_name: str,
    docs: Sequence[Document] | None = None,
    texts: Optional[Sequence[str]] = None,
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    distance: str = "cosine",
    persist_dir: Optional[Union[str, Path]] = None,
    quiet: bool = True,
) -> VSReturn:
    vs, info = _build_chroma_from_texts(
        embedding,
        collection_name=collection_name,
        docs=docs,
        texts=texts,
        metadatas=metadatas,
        persist_dir=persist_dir or _DEFAULT_CHROMA_DIR,
        distance=distance,
    )
    if not quiet:
        print(f"[vectorstore] Using Chroma: {info}")
    return VSReturn("chroma", vs, info)

# ---- Deprecated wrapper for backward compatibility ----
def auto_qdrant_faiss(  # noqa: D401
    embedding: Embeddings,
    *,
    collection_name: str,
    docs: Sequence[Document] | None = None,
    texts: Optional[Sequence[str]] = None,
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    vectors: Optional[Sequence[Sequence[float]]] = None,  # unused
    faiss_dir: Optional[Union[str, Path]] = None,         # unused
    qdrant_url: Optional[str] = None,                     # unused
    qdrant_api_key: Optional[str] = None,                 # unused
    distance: str = "cosine",
    content_payload_key: str = "page_content",            # unused
    prefer_grpc: Optional[bool] = None,                   # unused
    quiet: bool = True,
) -> VSReturn:
    import warnings
    warnings.warn(
        "auto_qdrant_faiss() is deprecated. Use create_chroma_store() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_chroma_store(
        embedding=embedding,
        collection_name=collection_name,
        docs=docs,
        texts=texts,
        metadatas=metadatas,
        distance=distance,
        persist_dir=_DEFAULT_CHROMA_DIR,
        quiet=quiet,
    )

# ---------------- Retriever helper ----------------
def as_retriever(
    vs: Any,
    *,
    search_type: str = "mmr",
    search_kwargs: Optional[Dict[str, Any]] = None,
    k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    lambda_mult: Optional[float] = None,
):
    """
    LangChain 스타일과 호환: search_kwargs 또는 개별 인자를 모두 지원
    """
    params: Dict[str, Any] = dict(search_kwargs or {})
    if k is not None:           params["k"] = k
    if fetch_k is not None:     params["fetch_k"] = fetch_k
    if lambda_mult is not None: params["lambda_mult"] = lambda_mult
    return vs.as_retriever(search_type=search_type, search_kwargs=params)