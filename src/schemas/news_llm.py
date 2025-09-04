# -*- coding: utf-8 -*-
# =============================================================================
# 변경 요약 (2025-09-04)
# - 다양한 LLM/체인 응답 형식(dict / 문자열(JSON 포함) / AIMessage)을
#   스키마 레벨에서 "관대하게" 수용하도록 보강.
# - root_validator(pre=True)로 입력을 dict로 정규화한 뒤
#   body_md, tldr, tags, sources 등을 일관되게 파싱.
# - 문자열 리스트화 보강(콤마/줄바꿈/세미콜론 분해).
# =============================================================================

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, validator, root_validator
import json
import re

# [CHG] 문자열 내 JSON 블록만 잘라 파싱
def _json_block(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    return json.loads(s)

# [CHG] 문자열 → 리스트화 (콤마/줄바꿈/세미콜론 분할)
def _to_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None and str(x).strip() != ""]
    s = str(v).strip()
    parts = re.split(r"[,\n;]+", s)
    return [p.strip() for p in parts if p.strip()]

class NewsLLMOut(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    tldr: List[str] = Field(default_factory=list)
    body_md: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    hero_image_url: Optional[str] = None
    author_name: Optional[str] = None

    # [CHG] 어떤 입력 형식이 와도 dict로 정규화
    @root_validator(pre=True)
    def _coerce_any_to_dict(cls, v):
        """
        허용:
        - dict (그대로)
        - 문자열(JSON 전체/앞뒤 잡음 포함)
        - AIMessage/Message 계열 (content 보유)
        - {"data": {...}}, {"result": {...}}, {"output": {...}}, {"response": {...}} 래핑
        """
        if v is None:
            return {}

        # AIMessage 등: content 추출
        if hasattr(v, "content"):
            v = getattr(v, "content")

        # 문자열 → JSON 추출 시도, 실패 시 그 문자열을 body_md로 간주
        if isinstance(v, str):
            try:
                v = _json_block(v)
            except Exception:
                return {"body_md": v}

        # 한 겹 래핑 해제
        if isinstance(v, dict):
            for k in ("data", "result", "output", "response"):
                if isinstance(v.get(k), dict):
                    v = v[k]
                    break

        # content에 또 JSON/본문이 들어있는 경우 body_md 보정
        if isinstance(v, dict) and isinstance(v.get("content"), (str, dict)):
            try:
                inner = v["content"]
                if isinstance(inner, dict):
                    if "body_md" in inner and not v.get("body_md"):
                        v["body_md"] = inner["body_md"]
                else:
                    data = _json_block(inner)
                    if isinstance(data, dict) and "body_md" in data and not v.get("body_md"):
                        v["body_md"] = data["body_md"]
            except Exception:
                pass
        return v

    @validator("tldr", "tags", "sources", pre=True)
    def _listify(cls, v):
        return _to_list(v)

    @validator("title", "subtitle", "body_md", "hero_image_url", "author_name", pre=True)
    def _to_str(cls, v):
        return None if v is None else str(v)


# from typing import List, Optional
# from pydantic import BaseModel, Field, validator

# class NewsLLMOut(BaseModel):
#     title: Optional[str] = None
#     subtitle: Optional[str] = None
#     tldr: List[str] = Field(default_factory=list)
#     body_md: Optional[str] = None
#     tags: List[str] = Field(default_factory=list)
#     sources: List[str] = Field(default_factory=list)
#     hero_image_url: Optional[str] = None
#     author_name: Optional[str] = None

#     @validator("tldr", "tags", "sources", pre=True)
#     def _listify(cls, v):
#         if v is None:
#             return []
#         if isinstance(v, list):
#             return [str(x) for x in v if x is not None]
#         return [str(v)]

#     @validator("title", "subtitle", "body_md", "hero_image_url", "author_name", pre=True)
#     def _to_str(cls, v):
#         return None if v is None else str(v)
