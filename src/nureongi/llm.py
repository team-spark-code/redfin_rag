# [신규 파일] nureongi/llm.py
from __future__ import annotations
import os
from typing import Any

def build_default_llm() -> Any:
    """
    [설명] 뉴스 체인에서도 rag_service에 의존하지 않고 직접 LLM 생성.
    우선순위:
      - GOOGLE_API_KEY → Gemini (기본모델: gemini-2.0-flash)
      - OPENAI_API_KEY → OpenAI (기본모델: gpt-4.1-mini)
    """
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    raise RuntimeError("No LLM provider key found (GOOGLE_API_KEY or OPENAI_API_KEY).")
