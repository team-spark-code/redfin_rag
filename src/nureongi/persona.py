# 수정: 템플릿 하드코딩 제거 → .md 파일 로딩/조립 방식으로 변경
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional

# langchain 호환
try:
    from langchain_core.documents import Document
except Exception:
    class Document:  # type: ignore
        page_content: str
        metadata: dict

# ===== 경로/캐시 =====
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_INSIGHT_PATH = PROMPTS_DIR / "system_insight.md"

_FILE_CACHE: Dict[Path, str] = {}
# (추가할 코드) ← 파일만 읽고, None/미존재/디렉터리는 빈 문자열
def _read(path: Optional[Path]) -> str:
    if not path:
        return ""
    if not path.exists() or path.is_dir():
        return ""                      # 델타가 없거나 잘못 주입되면 조용히 빈 문자열
    if path in _FILE_CACHE:
        return _FILE_CACHE[path]
    txt = path.read_text(encoding="utf-8")
    _FILE_CACHE[path] = txt
    return txt

# ===== 페르소나 정의 =====
class PersonaSlug(str, Enum):
    AI_CURIOUS_PUBLIC = "ai_curious_public"
    AI_INDUSTRY_PROFESSIONAL = "ai_industry_professional"
    AI_INTERESTED_OFFICIAL = "ai_interested_official"
    AI_STARTUP_CEO = "ai_startup_ceo"
    AI_CONTENT_CREATOR = "ai_content_creator"
    AI_STUDENT_RESEARCHER = "ai_student_researcher"
    AI_INVESTOR_VC = "ai_investor_vc"
    NEWS_INSIGHT_BASE = "news_insight_base"  # 빈값 기본

@dataclass(frozen=True)
class PersonaSpec:
    slug: PersonaSlug
    label: str
    version: str
    delta_path: Optional[Path]  # 델타 프롬프트 파일 경로 (없을 수도 있음)
    aliases: List[str]

PERSONA_SPECS: Mapping[PersonaSlug, PersonaSpec] = {
    PersonaSlug.AI_CURIOUS_PUBLIC: PersonaSpec(
        slug=PersonaSlug.AI_CURIOUS_PUBLIC,
        label="AI 관심 있는 일반인",
        version="v1",
        delta_path=PROMPTS_DIR / "personas" / "ai_curious_public.md",
        aliases=["일반인","대중","ai초보","호기심"],
    ),
    PersonaSlug.AI_INDUSTRY_PROFESSIONAL: PersonaSpec(
        slug=PersonaSlug.AI_INDUSTRY_PROFESSIONAL,
        label="AI 산업 종사자",
        version="v1",
        delta_path=PROMPTS_DIR / "personas" / "ai_industry_professional.md",
        aliases=["산업","종사자","엔지니어","ai전문가","기술자"],
    ),
    PersonaSlug.AI_INTERESTED_OFFICIAL: PersonaSpec(
        slug=PersonaSlug.AI_INTERESTED_OFFICIAL,
        label="AI 관심 있는 공무원",
        version="v1",
        delta_path=PROMPTS_DIR / "personas" / "ai_interested_official.md",
        aliases=["공무원","정책","공공","행정"],
    ),
    PersonaSlug.AI_STARTUP_CEO: PersonaSpec(
        slug=PersonaSlug.AI_STARTUP_CEO,
        label="AI 스타트업 CEO",
        version="v1",
        delta_path=PROMPTS_DIR / "personas" / "ai_startup_ceo.md",
        aliases=["ceo","창업자","스타트업","ai비즈니스"],
    ),
    PersonaSlug.AI_CONTENT_CREATOR: PersonaSpec(
        slug=PersonaSlug.AI_CONTENT_CREATOR,
        label="AI 정보 크리에이터",
        version="v1",
        delta_path=PROMPTS_DIR / "personas" / "ai_content_creator.md",
        aliases=["크리에이터","콘텐츠","유튜버","인플루언서"],
    ),
    PersonaSlug.AI_STUDENT_RESEARCHER: PersonaSpec(
        slug=PersonaSlug.AI_STUDENT_RESEARCHER,
        label="대학(원)생·연구자",
        version="v1",
        delta_path=PROMPTS_DIR / "personas" / "ai_student_researcher.md",
        aliases=["연구자","대학생","대학원생","학술"],
    ),
    PersonaSlug.AI_INVESTOR_VC: PersonaSpec(
        slug=PersonaSlug.AI_INVESTOR_VC,
        label="투자자·VC",
        version="v1",
        delta_path=PROMPTS_DIR / "personas" / "ai_investor_vc.md",
        aliases=["투자자","벤처캐피털","vc","투자"],
    ),
    PersonaSlug.NEWS_INSIGHT_BASE: PersonaSpec(
        slug=PersonaSlug.NEWS_INSIGHT_BASE,
        label="기본 인사이트(사건·이슈·트렌드)",
        version="v1",
        delta_path=None,  # 델타 없음
        aliases=["auto","default","기본","news_base","insight_base",""],
    ),
}

ALIAS_TO_SLUG: Dict[str, PersonaSlug] = {}
for spec in PERSONA_SPECS.values():
    for a in spec.aliases + [spec.slug.value]:
        ALIAS_TO_SLUG[a.lower()] = spec.slug
# 빈 문자열 대응
ALIAS_TO_SLUG[""] = PersonaSlug.NEWS_INSIGHT_BASE

def get_persona_by_alias(alias_or_slug: str) -> Optional[PersonaSpec]:
    key = (alias_or_slug or "").lower().strip()
    slug = ALIAS_TO_SLUG.get(key)
    return PERSONA_SPECS.get(slug) if slug else None

# ===== 공통 프리엠블(메타) =====
COMMON_PREAMBLE = """역할: {role}
운영규칙:
- 컨텍스트 우선: {use_context_only}. 컨텍스트 밖 주장은 금지, 불충분하면 "추가 근거 필요".
- 인용: {require_citations}. (출처/기관/연도/링크) 권장.
- 출력형식: {output_format}. 길이: {length_hint}. 불릿 최대 {max_bullets}개.
- 독자 수준: {reading_level}, 로케일: {locale}, 기준시각: {now} (최신성 기준: {date_cutoff})
- 개인정보/편향/오정보 주의, 명확·간결·검증가능성 준수.

[질문]
{question}

[컨텍스트]
{context}
"""

DEFAULT_PROMPT_ARGS = {
    "use_context_only": "true",
    "require_citations": "가능한 경우 반드시",
    "output_format": "markdown",
    "length_hint": "간결하지만 정보밀도 높게",
    "max_bullets": "7",
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

def build_persona_prompt_text(
    persona: str,
    question: str,
    docs: List[Document],
    now: str = "",
    user_profile: str = "",
    history: str = "",
    context_text: str | None = None,
    **overrides,
) -> str:
    """
    수정: system_insight.md(공통 형식) + persona 델타(.md) + 공통 프리엠블 + 질문/컨텍스트를 조립해 최종 프롬프트 문자열 반환
    """
    spec = get_persona_by_alias(persona) or PERSONA_SPECS[PersonaSlug.NEWS_INSIGHT_BASE]

    # 컨텍스트 백업 구성(너무 길지 않게)
    if context_text is None:
        parts, length, limit = [], 0, 6000
        for d in docs or []:
            md = getattr(d, "metadata", {}) or {}
            title = md.get("title") or md.get("headline") or ""
            src = md.get("source") or md.get("site") or md.get("publisher") or "source"
            piece = f"- [{src}] {title}\n{(d.page_content or '').strip()}\n\n"
            if length + len(piece) > limit:
                break
            parts.append(piece)
            length += len(piece)
        context_text = "".join(parts)

    system_prompt = _read(SYSTEM_INSIGHT_PATH).strip()
    delta_prompt = _read(spec.delta_path).strip() if spec.delta_path else ""

    args = render_prompt_args(
        question=question,
        context=context_text or "",
        user_profile=user_profile,
        history=history,
        now=now,
        role=spec.label,
        **overrides,
    )

    # 조립 순서: [공통 프리엠블] + [공통 시스템 프롬프트] + [페르소나 델타]
    full = (
        COMMON_PREAMBLE.format(**args)
        + "\n\n"
        + system_prompt
        + ("\n\n" + delta_prompt if delta_prompt else "")
    )
    return full

__all__ = [
    "PersonaSlug",
    "PersonaSpec",
    "PERSONA_SPECS",
    "ALIAS_TO_SLUG",
    "get_persona_by_alias",
    "render_prompt_args",
    "DEFAULT_PROMPT_ARGS",
    "build_persona_prompt_text",
]
