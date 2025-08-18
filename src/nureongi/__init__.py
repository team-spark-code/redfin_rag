"""
Nureongi RAG 패키지

정부 문서 기반 RAG 시스템을 위한 모듈들
"""

from .utils import find_pdf_files, serialize_document, save_jsonl
from .loaders import load_single_pdf
from .caches import load_cache, save_cache
from .chain import build_routed_rag_chain
from .router import choose_prompt, route_persona
from .format import format_ctx, format_docs, build_cite
from .persona import PersonaSlug, PROMPT_REGISTRY, ALIAS_TO_SLUG
from .vectorstore import auto_qdrant_faiss, as_retriever

__version__ = "0.1.0"
__all__ = [
    "find_pdf_files",
    "serialize_document", 
    "save_jsonl",
    "load_single_pdf",
    "load_cache",
    "save_cache",
    "build_routed_rag_chain",
    "choose_prompt",
    "route_persona",
    "format_ctx",
    "format_docs",
    "build_cite",
    "PersonaSlug",
    "PROMPT_REGISTRY",
    "ALIAS_TO_SLUG",
    "auto_qdrant_faiss",
    "as_retriever"
]
