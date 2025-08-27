# # nureongi/chain.py
# from __future__ import annotations
# import os
# from datetime import datetime
# from typing import Any
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.documents import Document
# from typing import List

# from .raptor import raptor_build_and_compress, RaptorParams
# from .format import format_ctx
# from .persona import get_persona_by_alias, render_prompt_args

# def _default_llm():
#     # 기본: Gemini, 없으면 OpenAI, 둘 다 없으면 에러
#     if os.getenv("GOOGLE_API_KEY"):
#         from langchain_google_genai import ChatGoogleGenerativeAI
#         return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL","gemini-2.0-flash"), temperature=0)
#     if os.getenv("OPENAI_API_KEY"):
#         from langchain_openai import ChatOpenAI
#         return ChatOpenAI(model=os.getenv("MODEL_LLM","gpt-4o-mini"), temperature=0)
#     raise RuntimeError("No LLM key found. Set GOOGLE_API_KEY or OPENAI_API_KEY.")

# RAPTOR_REQUIRED = os.getenv("RAPTOR_REQUIRED", "true").lower() != "false"

# def build_rag_chain(
#     retriever: Any,
#     *,
#     persona: str = "ai_industry_professional",
#     strategy: str = "stuff",
#     llm=None,
# ):
#     llm = llm or _default_llm()

#     # 페르소나 템플릿 선택
#     spec = get_persona_by_alias(persona)
#     if spec is None:
#         # 안전한 기본 프롬프트
#         tmpl = PromptTemplate.from_template(
#             "Use the CONTEXT to answer the QUESTION concisely. "
#             "If not found, say you don't know.\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}\n"
#         )
#         def to_prompt(inp):
#             return tmpl.format(question=inp["question"], context=inp["context"])
#     else:
#         def to_prompt(inp):
#             args = render_prompt_args(
#                 question=inp["question"],
#                 context=inp["context"],
#                 now=datetime.now().strftime("%Y-%m-%d %H:%M"),
#                 role=spec.label,
#             )
#             return spec.template.format(**args)

#     def with_context(x: dict):
#         # 1) 검색
#         leaf_docs: List[Document] = retriever.get_relevant_documents(x["question"])

#         # 2) RAPTOR(필수) — 실패 시에만 leaf 그대로 사용
#         ctx_docs: List[Document]
#         if RAPTOR_REQUIRED:
#             try:
#                 ctx_docs = raptor_build_and_compress(leaf_docs, llm=llm, params=RaptorParams())
#             except Exception:
#                 ctx_docs = leaf_docs
#         else:
#             ctx_docs = leaf_docs

#         # 3) 컨텍스트 문자열 생성(프로젝트의 format_ctx 재사용)
#         ctx_text = format_ctx(ctx_docs)
#         return {"question": x["question"], "context": ctx_text}

#     if strategy == "stuff":
#         chain = (
#             {"question": RunnablePassthrough()}
#             | RunnableLambda(with_context)   # ctx=RAPTOR+format_ctx
#             | RunnableLambda(to_prompt)      # 페르소나 템플릿
#             | llm
#             | StrOutputParser()
#         )
#     elif strategy == "map_refine":
#         # map 단계: 문서별 부분답 생성 → refine 단계: 통합
#         # (여기선 개념만; 실제 구현은 현재 구조를 크게 흔들지 않는 범위에서)
#         chain = build_map_refine_chain(retriever, to_prompt, llm)
#     elif strategy == "cod":
#         # 비교/설명형에 맞춘 프롬프트 힌트(최종답만)로 구성
#         chain = build_cod_chain(retriever, to_prompt, llm)
#     else:
#         chain = (
#             {"question": RunnablePassthrough()}
#             | RunnableLambda(with_context)
#             | RunnableLambda(to_prompt)
#             | llm
#             | StrOutputParser()
#         )
#         return chain
# nureongi/chain.py
from __future__ import annotations
import os
from datetime import datetime
from typing import Any, List, Dict
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from .format import format_ctx
from .persona import get_persona_by_alias, render_prompt_args
from .raptor import raptor_build_and_compress, RaptorParams


def _default_llm():
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL","gemini-2.0-flash"), temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("MODEL_LLM","gpt-4o-mini"), temperature=0)
    raise RuntimeError("No LLM key found. Set GOOGLE_API_KEY or OPENAI_API_KEY.")


# ---- 필수 RAPTOR (기본값 true) ----
RAPTOR_REQUIRED = os.getenv("RAPTOR_REQUIRED", "true").lower() != "false"

# 리트리버 호출 전에도 문자열 보정 + deprecation 해결
def _as_query_text(q) -> str:
    # dict로 들어오면 'question' 키 우선, 아니면 문자열화
    if isinstance(q, dict):
        return q.get("question", str(q))
    return q if isinstance(q, str) else str(q)

def _retrieve_with_raptor(retriever: Any, question: str, llm) -> List[Document]:
    """Retriever 결과를 가져오고, 필수 RAPTOR 트리 요약으로 압축."""
    q = _as_query_text(question)
    
    # ✅ deprecation 제거: get_relevant_documents → invoke
    leaf_docs = retriever.invoke(q) 
    if not RAPTOR_REQUIRED:
        return leaf_docs
    try:
        return raptor_build_and_compress(leaf_docs, llm=llm, params=RaptorParams())
    except Exception:
        # 요약 실패 시 leaf 그대로 사용
        return leaf_docs


def _make_to_prompt(persona: str):
    """기존 페르소나 로직을 그대로 감싼 prompt builder(문자열 반환)."""
    spec = get_persona_by_alias(persona)
    if spec is None:
        tmpl = PromptTemplate.from_template(
            "Use the CONTEXT to answer the QUESTION concisely. "
            "If not found, say you don't know.\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}\n"
        )
        def _to_prompt(inp: Dict[str, str]) -> str:
            return tmpl.format(question=inp["question"], context=inp["context"])
        return _to_prompt

    def _to_prompt(inp: Dict[str, str]) -> str:
        args = render_prompt_args(
            question=inp["question"],
            context=inp["context"],
            now=datetime.now().strftime("%Y-%m-%d %H:%M"),
            role=spec.label,
        )
        return spec.template.format(**args)

    return _to_prompt


# ---------------------------
# 전략 1) stuff (기존과 동일)
# ---------------------------
def _build_stuff_chain(retriever: Any, persona: str, llm):
    to_prompt = _make_to_prompt(persona)

    def with_context(x: dict):
        docs = _retrieve_with_raptor(retriever, x["question"], llm)
        ctx_text = format_ctx(docs)
        return {"question": x["question"], "context": ctx_text}

    chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(with_context)
        | RunnableLambda(to_prompt)
        | llm
        | StrOutputParser()
    )
    return chain


# ------------------------------------------
# 전략 2) map_refine (문서별 map → 최종 refine)
# ------------------------------------------
def _build_map_refine_chain(retriever: Any, persona: str, llm):
    to_prompt = _make_to_prompt(persona)

    # map 단계: 문서별 부분 답변 생성
    def _map_step(x: dict):
        q = x["question"]
        docs = _retrieve_with_raptor(retriever, q, llm)
        partials: List[str] = []
        for d in docs:
            # 문서 하나만 컨텍스트로 구성
            ctx_one = format_ctx([d])
            p = to_prompt({"question": q, "context": ctx_one})
            ans = llm.invoke(p)
            partials.append(str(ans))
        return {"question": q, "partials": partials}

    # refine 단계: 부분 답변들을 통합
    REFINE_TMPL = PromptTemplate.from_template(
        """아래 부분 응답들을 사실관계가 맞도록 중복 없이 통합하라.
- 허위/추측 배제, 수치·날짜·고유명사 유지
- 핵심만 간결하게 한국어로 작성

[질문]
{question}

[부분 응답들]
{partials}

[출력]
"""
    )

    def _refine_step(x: dict):
        q = x["question"]
        parts = x["partials"]
        joined = "\n\n---\n\n".join(parts) if parts else ""
        prompt = REFINE_TMPL.format(question=q, partials=joined)
        return prompt

    chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(_map_step)          # -> {"question", "partials"}
        | RunnableLambda(_refine_step)       # -> prompt(str)
        | llm
        | StrOutputParser()
    )
    return chain


# ----------------------------------------------------------
# 전략 3) cod (비교/설명형: 내부적 추론 유도, 최종 답만 출력)
# ----------------------------------------------------------
def _build_cod_chain(retriever: Any, persona: str, llm):
    to_prompt = _make_to_prompt(persona)

    COD_WRAPPER_TMPL = PromptTemplate.from_template(
        """아래는 질문과 컨텍스트에 기반한 답변 요청이다.
내부적으로는 단계별로 사고하되, 최종 결론만 한국어로 간결하게 제시하라. 사고 과정은 출력하지 마라.

[질문]
{question}

[컨텍스트]
{context}

[출력 지침]
- 비교/설명 포맷을 사용하되 증거 기반으로 기술
- 근거가 부족하면 '추가 근거 필요'를 명시
"""
    )

    def with_context(x: dict):
        q = x["question"]
        docs = _retrieve_with_raptor(retriever, q, llm)
        # COD는 비교/설명을 강조하므로 다문서 컨텍스트를 사용
        ctx = format_ctx(docs)
        # 페르소나 템플릿으로 기본 뼈대를 만든 뒤, COD 지침을 감싼다
        base_prompt = to_prompt({"question": q, "context": ctx})
        cod_prompt = COD_WRAPPER_TMPL.format(question=q, context=ctx)
        # base_prompt를 참고 텍스트로 삼도록 합성
        full_prompt = f"{cod_prompt}\n\n[참고 프롬프트]\n{base_prompt}\n"
        return {"prompt": full_prompt}

    chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(with_context)     # -> {"prompt": str}
        | RunnableLambda(lambda x: x["prompt"])
        | llm
        | StrOutputParser()
    )
    return chain


# ---------------------------
# 공개: 전략 선택 빌더
# ---------------------------
def build_rag_chain(
    retriever: Any,
    *,
    persona: str = "ai_industry_professional",
    strategy: str = "stuff",
    llm=None,
):
    llm = llm or _default_llm()
    strategy = (strategy or "stuff").lower()

    if strategy == "map_refine":
        return _build_map_refine_chain(retriever, persona, llm)
    elif strategy == "cod":
        return _build_cod_chain(retriever, persona, llm)
    else:
        # 기본값: stuff
        return _build_stuff_chain(retriever, persona, llm)
