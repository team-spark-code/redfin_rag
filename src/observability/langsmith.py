# src/observability/langsmith.py
# Helpers for routing LangSmith traces per service while scrubbing sensitive text.

import os
import re
from typing import Any, Dict, List, Optional

from langchain.callbacks.tracers import LangChainTracer

SERVICE_TO_ENV = {
    "redfin_target-insight": "LANGCHAIN_PROJECT_REDFIN_TARGET",
}


def get_project_for(service_name: str) -> str:
    """Resolve the LangSmith project name for a given service."""
    env_key = SERVICE_TO_ENV.get(service_name)
    if env_key and os.getenv(env_key):
        return os.getenv(env_key)
    return service_name


def make_tracer_for(service_name: str) -> LangChainTracer:
    """Create a tracer bound to the resolved project."""
    project = get_project_for(service_name)
    return LangChainTracer(project_name=project)


def make_tracer_explicit(project_name: Optional[str]) -> Optional[LangChainTracer]:
    if not project_name:
        return None
    return LangChainTracer(project_name=project_name)

# Simple PII scrubbing helpers
EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE = re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\d{2,4}[ -]?){2,4}\d{2,4}\b")
IDNUM = re.compile(r"\b\d{6,}\b")


def redact(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    text = IDNUM.sub("[NUMBER]", text)
    return text


def maybe_redact(text: str) -> str:
    enabled = os.getenv("ENABLE_PII_REDACTION", "false").lower() in {"1", "true", "yes"}
    return redact(text) if enabled else text


def build_trace_config(
    service_name: str,
    user_id: str,
    plan: Dict[str, Any] | None = None,
    extra_tags: List[str] | None = None,
) -> Dict[str, Any]:
    """Return the config payload passed to LangChain runnables."""
    tags = [f"service:{service_name}", "persona", "rag", "mmr"]
    if extra_tags:
        tags.extend(extra_tags)

    metadata: Dict[str, Any] = {"service": service_name, "user_id": user_id}
    if plan:
        metadata["plan"] = plan

    return {
        "run_name": f"{service_name}-invoke",
        "tags": tags,
        "metadata": metadata,
    }
