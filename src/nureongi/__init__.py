# nureongi/__init__.py

"""
nureongi: Redfin RAG 패키지의 Public API 집합
- RAPTOR, VectorStore, Chain을 공식 공개 API로 승격
- 기존 import 경로 호환 유지
"""

__version__ = "0.2.0"  # 버전 관리 권장

# --- Core loaders / indexing ---
from .loaders import NewsLoader
from .indexing import TextChunker, build_index

# --- Personas / formatting ---
from .persona import PersonaSlug, get_persona_by_alias  # 기존 호환 유지
from .format import format_ctx

# --- Vector store (Qdrant/FAISS) ---
from .vectorstore import (
    auto_qdrant_faiss,
    as_retriever,
    build_embedder,
)

# --- RAPTOR (기본 사용) ---
from .raptor import raptor_build_and_compress, RaptorParams

# --- RAG chain (LLM 호출) ---
from .chain import build_rag_chain

# --- Optional cache utils ---
from .cache import load_cache, save_cache


__all__ = [
    # Core
    "NewsLoader",
    "TextChunker",
    "build_index",

    # Personas / format
    "PersonaSlug",
    "get_persona_by_alias",
    "format_ctx",

    # Vector store
    "auto_qdrant_faiss",
    "as_retriever",
    "build_embedder",

    # RAPTOR
    "raptor_build_and_compress",
    "RaptorParams",

    # Chain
    "build_rag_chain",

    # Cache
    "load_cache",
    "save_cache",

    # Meta
    "__version__",
]
