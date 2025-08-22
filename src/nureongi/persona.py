# persona.py
# 통합 페르소나 레지스트리 (실서비스+RAG 최적화)
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional

# LC 0.2+ 우선 사용, 하위호환 fallback
try:
    from langchain_core.prompts import PromptTemplate
except Exception:  # pragma: no cover
    from langchain.prompts import PromptTemplate  # type: ignore


class PersonaSlug(str, Enum):
    AI_CURIOUS_PUBLIC = "ai_curious_public"
    AI_INDUSTRY_PROFESSIONAL = "ai_industry_professional"
    AI_INTERESTED_OFFICIAL = "ai_interested_official"
    AI_STARTUP_CEO = "ai_startup_ceo"
    AI_CONTENT_CREATOR = "ai_content_creator"
    AI_STUDENT_RESEARCHER = "ai_student_researcher"
    AI_INVESTOR_VC = "ai_investor_vc"


@dataclass(frozen=True)
class PersonaSpec:
    slug: PersonaSlug
    label: str
    version: str
    template: PromptTemplate
    aliases: List[str]


# ---------- 공통 규칙(모든 페르소나 상단에 삽입) ----------
COMMON_PREAMBLE = """역할: {role}
운영규칙:
- 컨텍스트 우선 사용: {use_context_only}. 컨텍스트 밖 주장 금지, 불충분하면 "추가 근거 필요"로 명시.
- 인용: {require_citations}. 가능한 경우 (출처/기관/연도/링크) 표기.
- 출력형식: {output_format}. 글자수/길이 기준: {length_hint}. 불릿 최대 {max_bullets}개.
- 독자 수준: {reading_level}, 로케일: {locale}, 날짜 기준: {now} (자료 최신성 cutoff: {date_cutoff}).
- 개인정보/편향/오정보 주의, 명확·간결·검증가능성 준수.
- 사용자 프로필(요약 가능): {user_profile}

[대화 히스토리 요약]
{history}

[질문]
{question}

[컨텍스트]
{context}
"""

# ---------- 페르소나별 본문 스펙 ----------
def _tmpl(body: str) -> PromptTemplate:
    return PromptTemplate.from_template(COMMON_PREAMBLE + "\n" + body.strip() + "\n")

PERSONA_SPECS: Mapping[PersonaSlug, PersonaSpec] = {
    PersonaSlug.AI_CURIOUS_PUBLIC: PersonaSpec(
        slug=PersonaSlug.AI_CURIOUS_PUBLIC, label="AI 관심 있는 일반인", version="v2",
        template=_tmpl(
            """출력:
1) 주제 개요(2~3문장, 쉬운 설명)
2) 핵심 포인트 불릿(실생활 예시 포함)
3) 오해 주의/윤리적 고려 1줄
4) 더 알아보기(공신력 있는 자료 2~3개)"""
        ),
        aliases=["일반인","대중","ai초보","호기심"]
    ),
    PersonaSlug.AI_INDUSTRY_PROFESSIONAL: PersonaSpec(
        slug=PersonaSlug.AI_INDUSTRY_PROFESSIONAL, label="AI 산업 종사자", version="v2",
        template=_tmpl(
            """출력:
1) 기술·제품 개요(핵심 스택/아키텍처)
2) 최신 동향/도구 3~5 불릿(수치·벤치마크·레퍼런스)
3) 적용 가이드(데이터/배포/리스크 요약)
4) 참고 리소스(논문/레포/레퍼런스 구현)"""
        ),
        aliases=["산업","종사자","엔지니어","기술자","ai전문가"]
    ),
    PersonaSlug.AI_INTERESTED_OFFICIAL: PersonaSpec(
        slug=PersonaSlug.AI_INTERESTED_OFFICIAL, label="AI 관심 있는 공무원", version="v2",
        template=_tmpl(
            """출력:
1) 정책적 개요(사회·경제 영향 2문장)
2) 법·제도/윤리 이슈 불릿(국내외 비교, 근거 표기)
3) 정책옵션(장단·영향·이해관계자)
4) 추가 검토자료(정부/국제기구 링크)"""
        ),
        aliases=["공무원","정책","공공","행정"]
    ),
    PersonaSlug.AI_STARTUP_CEO: PersonaSpec(
        slug=PersonaSlug.AI_STARTUP_CEO, label="AI 스타트업 CEO", version="v2",
        template=_tmpl(
            """출력:
1) 핵심 이슈(시장/경쟁/타이밍)
2) 전략 옵션 3가지(고객세그/포지셔닝/모델)
3) 리스크·규제 체크리스트
4) 액션 아이템(다음 2주 실행계획, KPI)"""
        ),
        aliases=["ceo","창업자","스타트업","ai비즈니스"]
    ),
    PersonaSlug.AI_CONTENT_CREATOR: PersonaSpec(
        slug=PersonaSlug.AI_CONTENT_CREATOR, label="AI 정보 크리에이터", version="v2",
        template=_tmpl(
            """출력:
1) 콘텐츠 각색 포인트(훅/스토리라인)
2) 포맷 제안(숏폼/캐러셀/뉴스레터)별 핵심 메시지
3) 시각화/데모 아이디어
4) 참고 소스(팩트체크 가능한 원문)"""
        ),
        aliases=["크리에이터","콘텐츠","유튜버","인플루언서"]
    ),
    PersonaSlug.AI_STUDENT_RESEARCHER: PersonaSpec(
        slug=PersonaSlug.AI_STUDENT_RESEARCHER, label="대학(원)생·연구자", version="v2",
        template=_tmpl(
            """출력:
1) 연구 배경 및 문제정의
2) 핵심 개념·식/알고리즘 요약(간단 정의 포함)
3) SOTA/비교 결과 요점(표현식·수치·데이터셋)
4) 재현 가이드(코드/데이터/평가절차)와 참고문헌(3+)"""
        ),
        aliases=["연구자","대학생","대학원생","학술"]
    ),
    PersonaSlug.AI_INVESTOR_VC: PersonaSpec(
        slug=PersonaSlug.AI_INVESTOR_VC, label="투자자·VC", version="v2",
        template=_tmpl(
            """출력:
1) 시장 개요(규모/성장률/구조)
2) 투자 인사이트 불릿(라운드/수익모델/차별화/리스크)
3) 경쟁 지형 스냅샷(대표 기업 3~5, 강약점 한줄씩)
4) 듀딜 포인트(기술/규제/거버넌스)"""
        ),
        aliases=["투자자","벤처캐피털","vc","투자"]
    ),
}

# -------- 별칭 -> 슬러그 역매핑 및 헬퍼 --------
ALIAS_TO_SLUG: Dict[str, PersonaSlug] = {}
for spec in PERSONA_SPECS.values():
    for a in spec.aliases + [spec.slug.value]:
        ALIAS_TO_SLUG[a.lower()] = spec.slug

def get_persona_by_alias(alias_or_slug: str) -> Optional[PersonaSpec]:
    key = (alias_or_slug or "").lower().strip()
    slug = ALIAS_TO_SLUG.get(key)
    return PERSONA_SPECS.get(slug) if slug else None

# -------- 체인 바인딩 시 기본 파라미터(노브) 제공 유틸 --------
DEFAULT_PROMPT_ARGS = {
    "use_context_only": "true",
    "require_citations": "가능한 경우 반드시",
    "output_format": "markdown",
    "length_hint": "간결하지만 정보밀도 높게",
    "max_bullets": "5",
    "reading_level": "중급(비전공자도 이해)",
    "locale": "ko-KR",
    "date_cutoff": "최근 6~12개월 최우선",
}

def render_prompt_args(
    question: str,
    context: str,
    user_profile: str = "",
    history: str = "",
    now: str = "",
    **overrides,
) -> dict:
    """
    PromptTemplate.format에 바로 전달할 dict 생성.
    공통 knobs를 기본값으로 채우고 overrides로 덮어씁니다.
    """
    args = dict(DEFAULT_PROMPT_ARGS)
    args.update({
        "question": question,
        "context": context,
        "user_profile": user_profile,
        "history": history,
        "now": now,
    })
    args.update(overrides or {})
    return args

__all__ = [
    "PersonaSlug",
    "PersonaSpec",
    "PERSONA_SPECS",
    "ALIAS_TO_SLUG",
    "get_persona_by_alias",
    "render_prompt_args",
    "DEFAULT_PROMPT_ARGS",
]
