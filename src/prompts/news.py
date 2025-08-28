# src/prompts/news.py
# Smart Brevity(Axios) 지시 템플릿 (한국어 고정)
# - 단어 수(어절) 기준: 전체 300단어 미만, 단락 1–2문장
# - 본문은 마크다운 불릿 위주
# - 반드시 "순수 JSON"만 출력

from typing import List, Optional

def _join(xs: Optional[List[str]]) -> str:
    return ", ".join([x for x in (xs or []) if x])

def build_news_prompt(
    *,
    title: Optional[str],
    url: Optional[str],
    content: Optional[str],
    categories: Optional[List[str]],
    tags: Optional[List[str]],
) -> str:
    t = (title or "").strip()
    u = (url or "").strip()
    body = (content or "").strip()
    cats = _join(categories)
    tgs = _join(tags)

    return f"""
당신은 뉴스 편집 보조 LLM입니다. 아래 원문을 바탕으로 **Axios 'Smart Brevity'** 원칙으로 짧고 명료한 기사를 생성하세요.
**언어는 반드시 한국어**입니다. (제목/부제/TL;DR/본문/태그/출처 등 모든 출력 항목 포함)
**반드시 순수 JSON만** 출력합니다. JSON 외의 텍스트/주석/코드블록/마크다운은 금지합니다.

출력 스키마(JSON):
{{
  "title": "기사 제목(한국어, 60자 이내)",
  "subtitle": "부제/데크 한 줄(한국어, 선택, 120자 이내)",
  "tldr": ["핵심 3줄(각 25단어 이하, 한국어)", "둘째", "셋째"],  // 정확히 3개
  "body_md": "마크다운 본문(한국어, 헤딩+불릿 위주, 전체 300단어 미만)",
  "tags": ["선택 키워드(한국어)"],
  "sources": ["원문/참고 URL 목록(가능하면 원문 URL을 첫 번째)"],
  "hero_image_url": "선택",
  "author_name": "선택(한국어 표기 허용)"
}}

Smart Brevity 규칙:
- 길이 기준은 **문자 수가 아니라 단어 수(어절)** 입니다.
- **전체 단어 수 300 미만**을 목표로 작성합니다.
- 단락은 **1–2문장**으로만 구성합니다(장문 금지).
- **본문(body_md)은 마크다운 불릿 위주**로 작성합니다. 아래 섹션 형식을 사용하세요:
  ## 한눈에
  - (핵심 요점 3개, 각 18단어 이하의 한 문장)
  ## 왜 중요한가
  - (영향·의미 1–2개, 각 18단어 이하)
  ## 숫자로 보기
  - (검증 가능한 수치·날짜·비율 2–4개, 단위·근거 포함, 각 18단어 이하)
  ## 맥락
  - (배경/이전 흐름 1–2개, 각 18단어 이하)
  ## 앞으로
  - (전망/다음 단계 1–2개, 각 18단어 이하)
- 불필요한 수식어·중복 문구 제거. 과장/추측 금지. 사실·수치·날짜·고유명사 정확히 유지.
- 인용은 **간접화법 요지**만; 긴 직접 인용 금지.
- URL은 `sources` 배열에만 두고, `body_md`에는 **직접 URL을 나열하지 않습니다**.
- 출력은 **유효한 JSON 하나**여야 합니다. 추가 텍스트 금지.

컨텍스트:
- 카테고리: [{cats}]
- 태그 힌트: [{tgs}]
- 원문 제목: "{t}"
- 원문 URL: {u}

원문:
\"\"\"{body}\"\"\"
""".strip()

def build_tldr_prompt(body_md: str, k: int = 3) -> str:
    # TL;DR 3줄 재생성 전용 (한국어)
    return f"""
아래 본문에서 핵심만 뽑아 **TL;DR 3줄**을 생성하세요.
각 항목은 **한국어 한 문장**, **25단어 이하**로 작성합니다.
**반드시 순수 JSON만** 출력합니다. 출력 형식:
{{"tldr":["첫째","둘째","셋째"]}}

본문:
\"\"\"{(body_md or '').strip()}\"\"\"
""".strip()
