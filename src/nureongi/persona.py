# ==== A) 정규화된 페르소나 정의 ====
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
from langchain_core.prompts import PromptTemplate

class PersonaSlug(str, Enum):
    GOV_POLICY      = "gov-policy"
    ACAD_RESEARCH   = "acad-research"
    INDUSTRY_ANALYST= "industry-analyst"
    STARTUP_PRE     = "startup-pre"
    EXEC_BRIEF      = "exec-brief"
    STAFF_REPORT    = "staff-report"
    STUDENT_UG      = "student-ug"

@dataclass(frozen=True)
class PersonaSpec:
    slug: PersonaSlug
    label: str          # 사람 친화 라벨(로그/대시보드)
    version: str        # "v1", "v2" ...
    template: PromptTemplate
    aliases: List[str]  # 자연어/이전 키 호환

PROMPT_REGISTRY: Dict[PersonaSlug, PersonaSpec] = {
    PersonaSlug.GOV_POLICY: PersonaSpec(
        slug=PersonaSlug.GOV_POLICY, label="정책입안자", version="v1",
        template=PromptTemplate.from_template(
            """역할: 중앙부처 정책기획 담당자 보조원.
규칙: 컨텍스트 내 정부문서/법령/고시/지침만 사용, 조문번호/기관/연도 인용(CITE), 최신 개정 우선.
출력:
1) 정책 배경(1~2문장)
2) 관련 법령/지침 조항 불릿(출처 포함)
3) 타 부처/국가 사례(있으면)
4) 정책 제언(근거 포함)

[질문]
{question}

[컨텍스트]
{context}
"""),
        aliases=["policy_maker","정책","공무원","정부","gov","policy"]
    ),
    PersonaSlug.ACAD_RESEARCH: PersonaSpec(
        slug=PersonaSlug.ACAD_RESEARCH, label="학술연구", version="v1",
        template=PromptTemplate.from_template(
            """역할: 학술 논문/연구보고서 보조원.
규칙: 통계/선행연구/공식 보고서만 사용, 정확 인용(기관/연도/쪽), 전문용어 유지.
출력:
1) 연구 배경(1문단)
2) 핵심 결과/데이터 불릿(출처 포함)
3) 시사점
4) 참고문헌 형식 예시(3개 이상)

[질문]
{question}

[컨텍스트]
{context}
"""),
        aliases=["academic","학술","논문","리뷰","acad"]
    ),
    PersonaSlug.INDUSTRY_ANALYST: PersonaSpec(
        slug=PersonaSlug.INDUSTRY_ANALYST, label="산업동향", version="v1",
        template=PromptTemplate.from_template(
            """역할: 산업·기술 동향 분석가.
규칙: 공식 통계/기술 보고서/전망 사용, 수치·시계열·기관 표기, 용어 한영 병기.
출력:
1) 개요(1문단)
2) 최근 동향 3~5 불릿(수치 포함)
3) 향후 전망(근거 포함)

[질문]
{question}

[컨텍스트]
{context}
"""),
        aliases=["research_institute","산업","동향","analyst","market","리포트"]
    ),
    PersonaSlug.STARTUP_PRE: PersonaSpec(
        slug=PersonaSlug.STARTUP_PRE, label="예비창업", version="v1",
        template=PromptTemplate.from_template(
            """역할: 규제·지원 정책 안내원.
규칙: 법령/지원사업 공고/시장데이터만 사용, 금액/기한/절차 명확화, 규제/지원 구분.
출력:
1) 관련 규제 요약(출처)
2) 지원사업 목록(금액·기한·요건)
3) 시장 진입 유의사항

[질문]
{question}

[컨텍스트]
{context}
"""),
        aliases=["pre_founder","창업","지원사업","grants","startup"]
    ),
    PersonaSlug.EXEC_BRIEF: PersonaSpec(
        slug=PersonaSlug.EXEC_BRIEF, label="Executive Brief", version="v1",
        template=PromptTemplate.from_template(
            """역할: 전략 기획 보좌관.
규칙: 정책 변화·시장 전망·경쟁 동향 추출, KPI 중심 요약.
출력:
1) 핵심 이슈(1문단)
2) 전략적 영향 불릿
3) 권고 전략(≤3)

[질문]
{question}

[컨텍스트]
{context}
"""),
        aliases=["executive","임원","요약","brief","c-level"]
    ),
    PersonaSlug.STAFF_REPORT: PersonaSpec(
        slug=PersonaSlug.STAFF_REPORT, label="실무보고", version="v1",
        template=PromptTemplate.from_template(
            """역할: 실무 보고서 지원자.
규칙: 데이터/사례 정확 인용, 간결·명료.
출력:
1) 업무 배경
2) 주요 데이터/사례 불릿(출처)
3) 결론/제안

[질문]
{question}

[컨텍스트]
{context}
"""),
        aliases=["staff","보고서","실무","report"]
    ),
    PersonaSlug.STUDENT_UG: PersonaSpec(
        slug=PersonaSlug.STUDENT_UG, label="대학생 리포트", version="v1",
        template=PromptTemplate.from_template(
            """역할: 대학 리포트 도우미.
규칙: 쉬운 문장, 핵심 용어 간단 정의, 출처 간단 표기(기관/연도/쪽).
출력:
1) 주제 개요(쉬운 설명)
2) 핵심 내용 불릿(용어 정의 포함)
3) 마무리 결론

[질문]
{question}

[컨텍스트]
{context}
"""),
        aliases=["undergrad","학부","리포트","student"]
    ),
}

# 별칭 -> 슬러그 역매핑
ALIAS_TO_SLUG: Dict[str, PersonaSlug] = {}
for spec in PROMPT_REGISTRY.values():
    for a in spec.aliases + [spec.slug.value]:
        ALIAS_TO_SLUG[a.lower()] = spec.slug
