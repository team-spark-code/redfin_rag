# nureongi/chain.py
from __future__ import annotations
import os
from datetime import datetime
from typing import Any
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .format import format_ctx
from .persona import get_persona_by_alias, render_prompt_args

def _default_llm():
    # 기본: Gemini, 없으면 OpenAI, 둘 다 없으면 에러
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL","gemini-2.0-flash"), temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("MODEL_LLM","gpt-4o-mini"), temperature=0)
    raise RuntimeError("No LLM key found. Set GOOGLE_API_KEY or OPENAI_API_KEY.")

def build_rag_chain(
    retriever: Any,
    *,
    persona: str = "ai_industry_professional",
    llm=None,
):
    llm = llm or _default_llm()

    # 페르소나 템플릿 선택
    spec = get_persona_by_alias(persona)
    if spec is None:
        # 안전한 기본 프롬프트
        tmpl = PromptTemplate.from_template(
            "Use the CONTEXT to answer the QUESTION concisely. "
            "If not found, say you don't know.\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}\n"
        )
        def to_prompt(inp):
            return tmpl.format(question=inp["question"], context=inp["context"])
    else:
        def to_prompt(inp):
            args = render_prompt_args(
                question=inp["question"],
                context=inp["context"],
                now=datetime.now().strftime("%Y-%m-%d %H:%M"),
                role=spec.label,
            )
            return spec.template.format(**args)

    def with_context(x: dict):
        ctx = format_ctx(retriever.get_relevant_documents(x["question"]))
        return {"question": x["question"], "context": ctx}

    chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(with_context)
        | RunnableLambda(to_prompt)
        | llm
        | StrOutputParser()
    )
    return chain
