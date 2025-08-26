# src/observability/langsmith.py
import os
import re
import contextlib
from typing import Any, Dict, List

SERVICE_TO_ENV = {
    "redfin_target-insight": "LANGCHAIN_PROJECT_REDFIN_TARGET",
    "redfin_news": "LANGCHAIN_PROJECT_REDFIN_NEWS",
}

def get_project_for(service_name: str) -> str:
    env_key = SERVICE_TO_ENV.get(service_name)
    if env_key and os.getenv(env_key):
        return os.getenv(env_key)
    # fallback: 서비스명 그대로
    return service_name

@contextlib.contextmanager
def set_ls_project(service_name: str):
    """요청 단위로 LangSmith 프로젝트를 스왑"""
    prev = os.getenv("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_PROJECT"] = get_project_for(service_name)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("LANGCHAIN_PROJECT", None)
        else:
            os.environ["LANGCHAIN_PROJECT"] = prev

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
    # LangChain 런 컨피그: run_name/tags/metadata
    tags = ["service:"+service_name, "persona", "rag", "mmr"]
    if extra_tags:
        tags.extend(extra_tags)

    metadata = {"service": service_name, "user_id": user_id}
    if plan:
        metadata["plan"] = plan

    # LangChain v0.2: Runnable.invoke(..., config={...})
    return {
        "run_name": f"{service_name}-invoke",
        "tags": tags,
        "metadata": metadata,
    }
