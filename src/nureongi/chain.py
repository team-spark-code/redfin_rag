# 수정: 프롬프트 조립을 persona.build_persona_prompt_text로 위임 + 밸리데이터 후단 연결
from __future__ import annotations
import os
from datetime import datetime
from typing import Any, List, Dict
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from contextvars import ContextVar

from .format import format_ctx
from .persona import build_persona_prompt_text
from .raptor import raptor_build_and_compress, RaptorParams
from .validators import validate_and_fix

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

# NEW: 요청 컨텍스트별 RAPTOR 적용 여부 플래그 (기본 False)
RAPTOR_APPLIED: ContextVar[bool] = ContextVar("RAPTOR_APPLIED", default=False)

def _as_query_text(q) -> str:
    if isinstance(q, dict):
        return q.get("question", str(q))
    return q if isinstance(q, str) else str(q)

def _retrieve_with_raptor(retriever: Any, question: str, llm) -> List[Document]:
    q = _as_query_text(question)
    leaf_docs = retriever.invoke(q)  # LangChain 0.2: invoke 사용
    
    # 기본값 리셋
    RAPTOR_APPLIED.set(False)

    if not RAPTOR_REQUIRED:
        return leaf_docs

    try:
        docs = raptor_build_and_compress(leaf_docs, llm=llm, params=RaptorParams())
        RAPTOR_APPLIED.set(True)   # NEW: 성공적으로 RAPTOR 압축 사용
        return docs
    except Exception:
        # 요약 실패 시 leaf 그대로 사용
        RAPTOR_APPLIED.set(False)  # NEW: 실패/미사용
        return leaf_docs

def _make_to_prompt(persona: str):
    """수정: 최종 프롬프트 문자열을 persona.build_persona_prompt_text로 생성"""
    def _to_prompt(inp: Dict[str, str]) -> str:
        return build_persona_prompt_text(
            persona=persona,
            question=inp["question"],
            docs=[],                          # context_text를 직접 전달하므로 docs는 비움
            context_text=inp["context"],
            now=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
    return _to_prompt

# ---------------------------
# 전략 1) stuff
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
        | RunnableLambda(lambda s: validate_and_fix(s, max_issues=7))   # ✅ 포맷 보증
    )
    return chain

# ------------------------------------------
# 전략 2) map_refine
# ------------------------------------------
def _build_map_refine_chain(retriever: Any, persona: str, llm):
    """
    map_refine 체인
    - Map 단계: 문서별 프롬프트를 구성해 llm.batch(...)로 병렬 호출하여 부분 응답 생성
    - Refine 단계: 부분 응답들을 통합 템플릿으로 정리하여 최종 응답 생성
    - 출력: validate_and_fix()로 포맷 보정
    """
    import time

    to_prompt = _make_to_prompt(persona)

    # ---- 운영 튜닝 파라미터(환경변수) ----
    MAP_REFINE_MAX_DOCS = int(os.getenv("MAP_REFINE_MAX_DOCS", "4"))     # Map 단계에서 처리할 문서 상한
    MAP_MAX_CONCURRENCY = int(os.getenv("MAP_MAX_CONCURRENCY", "4"))     # llm.batch 동시성
    REFINE_MAX_PARTIALS = int(os.getenv("REFINE_MAX_PARTIALS", "6"))     # Refine에 투입할 부분 응답 상한
    RETRY_MAX = int(os.getenv("MAP_BATCH_RETRY_MAX", "2"))               # 429 등 일시 오류 재시도 횟수
    RETRY_BACKOFF = float(os.getenv("MAP_BATCH_RETRY_BACKOFF", "1.5"))   # 지수 백오프 계수

    def _map_step(x: dict):
        q = x["question"]

        # 1) 리트리브(+RAPTOR 압축; 실패 시 leaf 폴백)
        docs = _retrieve_with_raptor(retriever, q, llm)

        # 2) 상위 N개만 Map 처리(지연·비용 제어)
        docs = docs[:MAP_REFINE_MAX_DOCS]

        # 3) 문서별 프롬프트 구성
        prompts: List[str] = []
        for d in docs:
            ctx_one = format_ctx([d])
            prompts.append(to_prompt({"question": q, "context": ctx_one}))

        # 4) 병렬 호출(batch) + 간단 재시도(429 대비)
        attempt = 0
        results = None
        last_exc: Exception | None = None

        while attempt <= RETRY_MAX:
            try:
                results = llm.batch(
                    prompts,
                    config={"max_concurrency": MAP_MAX_CONCURRENCY},
                )
                break
            except Exception as e:
                last_exc = e
                # 지수 백오프
                sleep_s = (RETRY_BACKOFF ** attempt)
                time.sleep(sleep_s)
                attempt += 1

        # 5) 재시도 모두 실패하면 최소 직렬 폴백
        if results is None:
            partials_fallback: List[str] = []
            for p in prompts:
                try:
                    ans = llm.invoke(p)
                    content = getattr(ans, "content", str(ans))
                    partials_fallback.append(content)
                except Exception:
                    continue
            return {"question": q, "partials": partials_fallback}

        # 6) 정상 결과 수집(AIMessage.content 우선)
        partials: List[str] = []
        for r in results:
            content = getattr(r, "content", str(r))
            partials.append(content)

        return {"question": q, "partials": partials}

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
        # Refine에 투입할 부분 응답도 상한 적용(입력 토큰 폭증 방지)
        parts = (x.get("partials") or [])[:REFINE_MAX_PARTIALS]
        joined = "\n\n---\n\n".join(parts) if parts else ""
        prompt = REFINE_TMPL.format(question=q, partials=joined)
        return prompt

    chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(_map_step)
        | RunnableLambda(_refine_step)
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda s: validate_and_fix(s, max_issues=7))   # 최종 포맷/품질 보정
    )
    return chain

# ----------------------------------------------------------
# 전략 3) cod
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
        ctx = format_ctx(docs)
        base_prompt = to_prompt({"question": q, "context": ctx})
        cod_prompt = COD_WRAPPER_TMPL.format(question=q, context=ctx)
        full_prompt = f"{cod_prompt}\n\n[참고 프롬프트]\n{base_prompt}\n"
        return {"prompt": full_prompt}

    chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(with_context)
        | RunnableLambda(lambda x: x["prompt"])
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda s: validate_and_fix(s, max_issues=7))   # ✅ 포맷 보증
    )
    return chain

# ---------------------------
# 공개: 전략 선택
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
        return _build_stuff_chain(retriever, persona, llm)
