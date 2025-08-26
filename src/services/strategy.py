from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict

Strategy = Literal["stuff", "map_refine", "cod"]
IndexMode = Literal["summary_only", "hybrid"]

@dataclass
class StrategyPlan:
    strategy: Strategy
    reason: str
    # Retriever
    k: int = 8
    fetch_k: int = 60
    lambda_mult: float = 0.25
    # Index / RAPTOR
    use_raptor: bool = True
    index_mode: IndexMode = "summary_only"   # summary_only: 비용↓, hybrid: 정밀도↑
    raptor_max_depth: int = 2                # 2~3
    raptor_branching: int = 6                # 4~8
    # Generation
    allow_cot: bool = False                  # 체인 오브 생각(CoD)
    max_context_tokens: int = 1800           # 컨텍스트 상한
    # 기타 힌트
    needs_citation: bool = False
    needs_numeric_exactness: bool = False

def estimate_tokens(text: str) -> int:
    # 대략적 근사(영/한 혼용 고려, 단어수*1.3 정도)
    return int(len(text.split()) * 1.3)

def classify_intent(q: str) -> Dict[str, bool]:
    ql = q.lower()
    return {
        "is_compare": any(x in q for x in ["비교", "장단점"]) or "compare" in ql or "vs" in ql,
        "is_explain": any(x in q for x in ["왜", "어떻게"]) or "explain" in ql or "how " in ql,
        "is_brief": any(x in q for x in ["요약", "tl;dr", "한줄", "bullet"]) or "summary" in ql,
        "needs_citation": any(x in ql for x in ["출처", "cite", "citation", "근거"]),
        "needs_numbers": any(x in q for x in ["수치", "정량", "숫자", "수준", "정확"]) or "number" in ql,
        "is_time_sensitive": any(x in q for x in ["최신", "오늘", "방금"]) or "today" in ql or "latest" in ql,
        "is_scope_wide": any(x in q for x in ["전반", "전체", "개요"]) or "overview" in ql,
    }

def choose_strategy_advanced(
    question: str,
    est_context_tokens: Optional[int] = None,
    k_hint: Optional[int] = None,
    doc_count_hint: Optional[int] = None,
    token_budget: int = 3000,     # 프롬프트+컨텍스트 총 예산(모델 정책에 맞게 조정)
) -> StrategyPlan:

    intents = classify_intent(question)
    qtoks = estimate_tokens(question)
    # 컨텍스트 추정치가 없으면 질문 길이·k 추정으로 대략 잡는다
    if est_context_tokens is None:
        base = 150 if intents["is_brief"] else 350  # 요약성 질문은 컨텍스트 적게
        k_guess = k_hint or 8
        est_context_tokens = base * k_guess

    plan = StrategyPlan(
        strategy="map_refine",
        reason="기본값",
    )

    # 1) 정확도·인용이 필요한 경우 → HYBRID 인덱스 + k↑, fetch_k↑
    if intents["needs_citation"] or intents["needs_numbers"]:
        plan.index_mode = "hybrid"
        plan.k = max(k_hint or 8, 10)
        plan.fetch_k = max(80, plan.k * 8)
        plan.lambda_mult = 0.2
        plan.needs_citation = intents["needs_citation"]
        plan.needs_numeric_exactness = intents["needs_numbers"]

    # 2) 비교/설명형 → CoD 허용 + Map-Refine
    if intents["is_compare"] or intents["is_explain"]:
        plan.strategy = "cod"
        plan.allow_cot = True
        plan.k = max(k_hint or plan.k, 8)
        plan.fetch_k = max(plan.fetch_k, 80)
        plan.lambda_mult = 0.25

    # 3) 매우 짧은 답변/브리핑 → Stuff + RAPTOR 상위 요약 위주
    if intents["is_brief"] and est_context_tokens <= 1200:
        plan.strategy = "stuff"
        plan.index_mode = "summary_only"
        plan.k = min(k_hint or 6, 8)
        plan.fetch_k = 40
        plan.lambda_mult = 0.3
        plan.raptor_max_depth = 2
        plan.max_context_tokens = 1200
        plan.reason = "짧은 브리핑/요약 질의"

    # 4) 범위가 넓거나 문서가 많음 → Map-Refine + RAPTOR 깊이 3
    if intents["is_scope_wide"] or (doc_count_hint and doc_count_hint > 200):
        plan.strategy = "map_refine"
        plan.raptor_max_depth = 3
        plan.k = max(k_hint or plan.k, 10)
        plan.fetch_k = max(plan.fetch_k, 90)
        plan.lambda_mult = 0.25
        plan.reason = "광범위/대량 컨텍스트"

    # 5) 토큰 예산 고려한 조정
    # 컨텍스트 상한치 = 토큰예산 * 0.6 정도로 설정(나머지는 시스템/질문/출력)
    plan.max_context_tokens = min(plan.max_context_tokens, int(token_budget * 0.6))
    if est_context_tokens > plan.max_context_tokens:
        # 과하면 k와 fetch_k를 줄이고, summary_only를 우선
        shrink_ratio = plan.max_context_tokens / max(1, est_context_tokens)
        if shrink_ratio < 0.75:
            plan.k = max(4, int(plan.k * shrink_ratio))
            plan.fetch_k = max(plan.k * 6, int(plan.fetch_k * shrink_ratio))
            plan.index_mode = "summary_only"
            plan.raptor_max_depth = min(plan.raptor_max_depth, 2)
            plan.reason = (plan.reason + " | 토큰예산에 맞춰 축소").strip(" |")

    # 6) 최신성 강조 시 검색 다양성 ↑ (MMR 후보 확장)
    if intents["is_time_sensitive"]:
        plan.fetch_k = max(plan.fetch_k, 100)
        plan.lambda_mult = 0.3
        plan.reason = (plan.reason + " | 최신성 고려").strip(" |")

    # 7) 최종 기본값/사유 정리
    if plan.reason == "기본값":
        plan.reason = "질문 특성에 따른 기본 휴리스틱 적용"

    return plan

