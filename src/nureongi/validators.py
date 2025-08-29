# 강화 밸리데이터: 헤더 동의어 정규화 + 문단→불릿 변환 + 상한 적용 + 제목 자동 생성/중복 제거
from __future__ import annotations
import re
from typing import List, Tuple

# ===== 공용 패턴 =====
H1_RE = re.compile(r"^#\s+.+", re.M)
ISSUE_CANON_RE = re.compile(r"^##\s+Issue List\s*$", re.M)
TREND_CANON_RE = re.compile(r"^##\s+Trend\s*$", re.M)

# 동의어(행 전체 매칭)
ISSUE_SYNS = [
    r"Issue\s*List", r"이슈\s*리스트", r"주요\s*이슈", r"핵심\s*이슈", r"핵심\s*포인트", r"핵심\s*요점",
]
TREND_SYNS = [
    r"Trend", r"종합\s*동향\s*및\s*전망", r"종합\s*동향", r"전망", r"향후\s*과제(\s*및\s*트렌드)?", r"동향", r"전망\s*및\s*체크포인트",
]

ISSUE_SYNS_RE = re.compile(r"^\s*(?:##\s+)?(?:" + "|".join(ISSUE_SYNS) + r")\s*:?$", re.M | re.I)
TREND_SYNS_RE = re.compile(r"^\s*(?:##\s+)?(?:" + "|".join(TREND_SYNS) + r")\s*:?$", re.M | re.I)

# 이슈 불릿/하위항목
ISSUE_BULLET_RE = re.compile(r"^\s*-\s+\*\*(.+?)\*\*:", re.M)     # - **이슈명**:
SUB_WHY_RE = re.compile(r"^\s{2,}-\s*왜 중요한가\s*:", re.M)
SUB_EVID_RE = re.compile(r"^\s{2,}-\s*근거\s*:", re.M)

# 앵커(수치/날짜/단위) 휴리스틱
ANCHOR_PAT = re.compile(
    r"(\d{4}[-./]?\d{1,2}[-./]?\d{1,2}|\d{4}\s*년|\d{1,2}\s*월|\d+%|\d+(?:\.\d+)?\s*(ms|배|건|회|개|억|만|달러|원))"
)

# ===== 전처리 =====
def _dedupe_consecutive_lines(text: str) -> str:
    lines = text.splitlines()
    out, prev = [], None
    for ln in lines:
        if ln.strip() and prev is not None and ln.strip() == prev.strip():
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out)

def _normalize_headers(text: str) -> str:
    t = ISSUE_SYNS_RE.sub("## Issue List", text)
    t = TREND_SYNS_RE.sub("## Trend", t)
    return t

# ===== 제목 생성 =====
def _synthesize_title(text: str) -> str:
    m = ISSUE_BULLET_RE.search(text)
    if m:
        base = m.group(1).strip()
        return f"# {base} — 이슈 브리핑"
    m2 = re.search(r"^최근[^\n]{3,60}", text, re.M)
    if m2:
        return "# " + m2.group(0).strip().rstrip(" .")
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return "# " + (first_line[:60] + ("…" if len(first_line) > 60 else "")) if first_line else "# 이슈 브리핑"

def _ensure_title(text: str) -> str:
    if H1_RE.search(text):
        return text
    return _synthesize_title(text) + "\n\n" + text.lstrip()

# ===== 섹션 분리 =====
def _split_to_sections(text: str) -> Tuple[str, str, str]:
    """제목 이후 본문을 Issue/Trend로 분리. 없으면 빈 문자열."""
    m_issue = ISSUE_CANON_RE.search(text)
    m_trend = TREND_CANON_RE.search(text)

    if not m_issue and not m_trend:
        return text, "", ""  # 둘 다 없으면 본문 전체가 head로 남음

    head_end = min(m.start() for m in [m for m in [m_issue, m_trend] if m])
    head = text[: head_end]

    issue_block = ""
    trend_block = ""
    if m_issue and m_trend:
        issue_block = text[m_issue.end(): m_trend.start()].strip()
        trend_block = text[m_trend.end(): ].strip()
    elif m_issue:
        issue_block = text[m_issue.end(): ].strip()
    else:
        trend_block = text[m_trend.end(): ].strip()

    return head.rstrip(), issue_block, trend_block

# ===== 문단 → 이슈 불릿 변환 =====
def _paragraphs(block: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", block) if p.strip()]

def _to_issue_bullets_from_paragraphs(paras: List[str]) -> List[str]:
    bullets = []
    for para in paras:
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        if not lines:
            continue
        title = lines[0]  # 첫 줄을 이슈명으로 가정
        rest = lines[1:]
        # why/evid 추출
        why = ""
        evid = ""
        for ln in rest:
            if not why:
                why = ln
            if ln.startswith(("관련 근거", "근거:")):
                evid = re.sub(r"^\s*(관련\s*)?근거\s*:\s*", "", ln).strip()
        if not evid:
            for ln in rest:
                if ANCHOR_PAT.search(ln):
                    evid = ln
                    break
        if not why:  why = "추가 근거 필요"
        if not evid: evid = "추가 근거 필요"

        bullet = [
            f"- **{title}**: {why if len(why) <= 140 else why[:140] + '…'}",
            f"  - 왜 중요한가: {why}",
            f"  - 근거: {evid}",
        ]
        bullets.append("\n".join(bullet))
    return bullets

def _normalize_issue_block(issue_block: str) -> str:
    if not issue_block.strip():
        return ""
    # 이미 형식이 맞으면 그대로
    if ISSUE_BULLET_RE.search(issue_block) and SUB_WHY_RE.search(issue_block) and SUB_EVID_RE.search(issue_block):
        return issue_block.strip()
    # 문단을 불릿으로 변환
    paras = _paragraphs(issue_block)
    bullets = _to_issue_bullets_from_paragraphs(paras)
    return "\n\n".join(bullets).strip() if bullets else ""

def _cap_issues(issue_block: str, max_issues: int) -> str:
    idxs = [m.start() for m in ISSUE_BULLET_RE.finditer(issue_block)]
    if len(idxs) <= max_issues:
        return issue_block
    parts, count = [], 0
    for line in issue_block.splitlines():
        if ISSUE_BULLET_RE.match(line):
            count += 1
        if count <= max_issues:
            parts.append(line)
    return "\n".join(parts)

# ===== Trend 정규화 =====
def _normalize_trend(trend_block: str) -> str:
    if not trend_block.strip():
        return "- 추가 근거 필요"
    sents = re.split(r"(?<=[.!?。])\s+", trend_block.strip())
    sents = [s.strip("•- ").strip() for s in sents if s.strip()]
    if len(sents) > 4:
        sents = sents[:4]
    return "\n".join(sents)

# ===== 메인 엔트리 =====
def validate_and_fix(text: str, *, max_issues: int = 7) -> str:
    if not text:
        return "(출력 없음)"

    # 0) 중복 라인 제거 + 헤더 동의어를 캐논으로
    out = _dedupe_consecutive_lines(text)
    out = _normalize_headers(out)

    # 1) 제목 보장
    out = _ensure_title(out)

    # 2) 섹션 분리
    head, issue_block, trend_block = _split_to_sections(out)

    # 2-1) 섹션이 없으면: head 본문에서 이슈/트렌드 추출 시도
    if not issue_block and not trend_block:
        # '종합 동향'/'전망' 단어가 있는 문단 이후를 트렌드로, 그 앞을 이슈로 가정
        paras = _paragraphs(head)
        trend_idx = next((i for i, p in enumerate(paras) if re.search(r"(종합\s*동향|전망|향후\s*과제)", p)), None)
        if trend_idx is not None:
            issue_block = "\n\n".join(paras[:trend_idx]).strip()
            trend_block = "\n\n".join(paras[trend_idx:]).strip()
            head = ""  # 본문은 섹션으로 흡수
        else:
            # 전부 이슈로 가정
            issue_block = head.strip()
            head = ""

    # 3) Issue 정규화 + 상한
    issue_block = _normalize_issue_block(issue_block) or "- **(이슈 없음)**: 추가 근거 필요\n  - 왜 중요한가: 추가 근거 필요\n  - 근거: 추가 근거 필요"
    issue_block = _cap_issues(issue_block, max_issues)
    issue_render = "## Issue List\n" + issue_block

    # 4) Trend 정규화
    trend_render = "## Trend\n" + _normalize_trend(trend_block)

    # 5) 최종 조립 (+ 구분선)
    parts = []
    head = head.strip()
    if head:
        parts.append(head)
    parts.append(issue_render)
    parts.append("\n---\n")
    parts.append(trend_render)

    return "\n\n".join(parts).strip()
