# src/observability/langsmith.py
# 목적: 요청마다 LangSmith 프로젝트를 "안전하게" 분리 기록하기 위해
#       전역 환경변수 스왑 대신 LangChainTracer(project_name=...)을 생성해
#       invoke(..., config={"callbacks":[tracer]})에 주입합니다.
#       -> 동시 요청에서도 프로젝트 섞임 방지.

import os
import re
from typing import Any, Dict, List, Optional

# LangChain v0.2 콜백 트레이서
from langchain.callbacks.tracers import LangChainTracer

# 서비스명 → 프로젝트명 환경키 매핑 (원하는대로 커스터마이즈)
SERVICE_TO_ENV = {
    "redfin_target-insight": "LANGCHAIN_PROJECT_REDFIN_TARGET",
    "redfin_news": "LANGCHAIN_PROJECT_REDFIN_NEWS",
}

def get_project_for(service_name: str) -> str:
    """
    서비스명을 받아 LangSmith 프로젝트명을 결정.
    우선순위: SERVICE_TO_ENV에 매핑된 환경키 → 없으면 서비스명 그대로.
    """
    env_key = SERVICE_TO_ENV.get(service_name)
    if env_key and os.getenv(env_key):
        return os.getenv(env_key)  # 예: "api_rag", "redfin_news-publish"
    return service_name

def make_tracer_for(service_name: str) -> LangChainTracer:
    """
    서비스명 규칙에 따라 해당 서비스 전용 프로젝트 트레이서를 생성.
    - 동시 요청에서도 안전함(전역 env 변경 없음).
    """
    project = get_project_for(service_name)
    return LangChainTracer(project_name=project)

def make_tracer_explicit(project_name: Optional[str]) -> Optional[LangChainTracer]:
    """
    프로젝트명을 직접 지정하고 싶을 때 사용(뉴스 라우터 등에서 명시적으로).
    None/빈값이면 콜백 주입 없이 진행.
    """
    if not project_name:
        return None
    return LangChainTracer(project_name=project_name)

# --- 간단 PII 마스킹 (필요 시 고도화) ---
EMAIL = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
PHONE = re.compile(r'\b(?:\+?\d{1,3}[ -]?)?(?:\d{2,4}[ -]?){2,4}\d{2,4}\b')
IDNUM = re.compile(r'\b\d{6,}\b')

def redact(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    text = IDNUM.sub("[NUMBER]", text)
    return text

def maybe_redact(text: str) -> str:
    return redact(text) if os.getenv("ENABLE_PII_REDACTION", "false").lower() in ("1","true","yes") else text

def build_trace_config(
    service_name: str,
    user_id: str,
    plan: Dict[str, Any] | None = None,
    extra_tags: List[str] | None = None,
) -> Dict[str, Any]:
    """
    LangChain Runnable.invoke(..., config=여기 반환값) 에 전달할 설정.
    - run_name/tags/metadata만 구성. callbacks는 라우터에서 주입.
    """
    tags = ["service:"+service_name, "persona", "rag", "mmr"]
    if extra_tags:
        tags.extend(extra_tags)
    metadata: Dict[str, Any] = {"service": service_name, "user_id": user_id}
    if plan:
        metadata["plan"] = plan
    return {
        "run_name": f"{service_name}-invoke",
        "tags": tags,
        "metadata": metadata,
        # 주의: callback는 라우터에서 주입
    }
