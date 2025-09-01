from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Set
import re

# {{var_name}} 패턴만 치환합니다. 중괄호 2개 고정.
_VAR_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

def load_md_template(path: str | Path, *, encoding: str = "utf-8") -> str:
    """
    .md 템플릿 파일을 읽어 문자열로 반환.
    - 경로가 틀리면 FileNotFoundError 발생
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Template not found: {p}")
    return p.read_text(encoding=encoding)

def find_placeholders(md: str) -> Set[str]:
    """
    템플릿 안에 등장하는 {{placeholder}} 이름들을 집합으로 반환.
    - 디버깅/검증용
    """
    return set(m.group(1) for m in _VAR_PATTERN.finditer(md))

def render_template(md: str, /, **vars: object) -> str:
    """
    매우 단순한 치환기: {{name}} → vars["name"]
    - 누락된 키는 빈 문자열("")로 치환합니다.
    - 값은 str()로 강제 변환하여 넣습니다.
    """
    def _repl(m: re.Match) -> str:
        key = m.group(1)
        val = vars.get(key, "")
        return "" if val is None else str(val)
    return _VAR_PATTERN.sub(_repl, md)

def render_template_strict(md: str, /, **vars: object) -> str:
    """
    strict 모드: 템플릿에 등장한 모든 키가 반드시 제공되어야 합니다.
    - 누락 시 KeyError 발생
    """
    needed = find_placeholders(md)
    missing = [k for k in needed if k not in vars]
    if missing:
        raise KeyError(f"Missing template variables: {missing}")
    return render_template(md, **vars)
