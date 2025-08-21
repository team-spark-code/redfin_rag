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


# # ==== A) 정규화된 페르소나 정의 ====
# from enum import Enum
# from dataclasses import dataclass
# from typing import List, Dict
# from langchain_core.prompts import PromptTemplate

# class PersonaSlug(str, Enum):
#     GOV_POLICY      = "gov-policy"
#     ACAD_RESEARCH   = "acad-research"
#     INDUSTRY_ANALYST= "industry-analyst"
#     STARTUP_PRE     = "startup-pre"
#     EXEC_BRIEF      = "exec-brief"
#     STAFF_REPORT    = "staff-report"
#     STUDENT_UG      = "student-ug"

# @dataclass(frozen=True)
# class PersonaSpec:
#     slug: PersonaSlug
#     label: str          # 사람 친화 라벨(로그/대시보드)
#     version: str        # "v1", "v2" ...
#     template: PromptTemplate
#     aliases: List[str]  # 자연어/이전 키 호환

# PROMPT_REGISTRY: Dict[PersonaSlug, PersonaSpec] = {
#     PersonaSlug.GOV_POLICY: PersonaSpec(
#         slug=PersonaSlug.GOV_POLICY, label="정책입안자", version="v1",
#         template=PromptTemplate.from_template(
#             """역할: 중앙부처 정책기획 담당자 보조원.
# 규칙: 컨텍스트 내 정부문서/법령/고시/지침만 사용, 조문번호/기관/연도 인용(CITE), 최신 개정 우선.
# 출력:
# 1) 정책 배경(1~2문장)
# 2) 관련 법령/지침 조항 불릿(출처 포함)
# 3) 타 부처/국가 사례(있으면)
# 4) 정책 제언(근거 포함)

# [질문]
# {question}

# [컨텍스트]
# {context}
# """),
#         aliases=["policy_maker","정책","공무원","정부","gov","policy"]
#     ),
#     PersonaSlug.ACAD_RESEARCH: PersonaSpec(
#         slug=PersonaSlug.ACAD_RESEARCH, label="학술연구", version="v1",
#         template=PromptTemplate.from_template(
#             """역할: 학술 논문/연구보고서 보조원.
# 규칙: 통계/선행연구/공식 보고서만 사용, 정확 인용(기관/연도/쪽), 전문용어 유지.
# 출력:
# 1) 연구 배경(1문단)
# 2) 핵심 결과/데이터 불릿(출처 포함)
# 3) 시사점
# 4) 참고문헌 형식 예시(3개 이상)

# [질문]
# {question}

# [컨텍스트]
# {context}
# """),
#         aliases=["academic","학술","논문","리뷰","acad"]
#     ),
#     PersonaSlug.INDUSTRY_ANALYST: PersonaSpec(
#         slug=PersonaSlug.INDUSTRY_ANALYST, label="산업동향", version="v1",
#         template=PromptTemplate.from_template(
#             """역할: 산업·기술 동향 분석가.
# 규칙: 공식 통계/기술 보고서/전망 사용, 수치·시계열·기관 표기, 용어 한영 병기.
# 출력:
# 1) 개요(1문단)
# 2) 최근 동향 3~5 불릿(수치 포함)
# 3) 향후 전망(근거 포함)

# [질문]
# {question}

# [컨텍스트]
# {context}
# """),
#         aliases=["research_institute","산업","동향","analyst","market","리포트"]
#     ),
#     PersonaSlug.STARTUP_PRE: PersonaSpec(
#         slug=PersonaSlug.STARTUP_PRE, label="예비창업", version="v1",
#         template=PromptTemplate.from_template(
#             """역할: 규제·지원 정책 안내원.
# 규칙: 법령/지원사업 공고/시장데이터만 사용, 금액/기한/절차 명확화, 규제/지원 구분.
# 출력:
# 1) 관련 규제 요약(출처)
# 2) 지원사업 목록(금액·기한·요건)
# 3) 시장 진입 유의사항

# [질문]
# {question}

# [컨텍스트]
# {context}
# """),
#         aliases=["pre_founder","창업","지원사업","grants","startup"]
#     ),
#     PersonaSlug.EXEC_BRIEF: PersonaSpec(
#         slug=PersonaSlug.EXEC_BRIEF, label="Executive Brief", version="v1",
#         template=PromptTemplate.from_template(
#             """역할: 전략 기획 보좌관.
# 규칙: 정책 변화·시장 전망·경쟁 동향 추출, KPI 중심 요약.
# 출력:
# 1) 핵심 이슈(1문단)
# 2) 전략적 영향 불릿
# 3) 권고 전략(≤3)

# [질문]
# {question}

# [컨텍스트]
# {context}
# """),
#         aliases=["executive","임원","요약","brief","c-level"]
#     ),
#     PersonaSlug.STAFF_REPORT: PersonaSpec(
#         slug=PersonaSlug.STAFF_REPORT, label="실무보고", version="v1",
#         template=PromptTemplate.from_template(
#             """역할: 실무 보고서 지원자.
# 규칙: 데이터/사례 정확 인용, 간결·명료.
# 출력:
# 1) 업무 배경
# 2) 주요 데이터/사례 불릿(출처)
# 3) 결론/제안

# [질문]
# {question}

# [컨텍스트]
# {context}
# """),
#         aliases=["staff","보고서","실무","report"]
#     ),
#     PersonaSlug.STUDENT_UG: PersonaSpec(
#         slug=PersonaSlug.STUDENT_UG, label="대학생 리포트", version="v1",
#         template=PromptTemplate.from_template(
#             """역할: 대학 리포트 도우미.
# 규칙: 쉬운 문장, 핵심 용어 간단 정의, 출처 간단 표기(기관/연도/쪽).
# 출력:
# 1) 주제 개요(쉬운 설명)
# 2) 핵심 내용 불릿(용어 정의 포함)
# 3) 마무리 결론

# [질문]
# {question}

# [컨텍스트]
# {context}
# """),
#         aliases=["undergrad","학부","리포트","student"]
#     ),
# }

# # 별칭 -> 슬러그 역매핑
# ALIAS_TO_SLUG: Dict[str, PersonaSlug] = {}
# for spec in PROMPT_REGISTRY.values():
#     for a in spec.aliases + [spec.slug.value]:
#         ALIAS_TO_SLUG[a.lower()] = spec.slug
