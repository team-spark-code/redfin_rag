# src/routers/redfin.py
from time import perf_counter
from uuid import uuid4
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Any, Dict
from fastapi import APIRouter, Request, HTTPException

from core import settings
from schemas.query import QueryRequest
from schemas.response import QueryResponseV1
from services import rag_service
from services.strategy import choose_strategy_advanced
from nureongi import get_persona_by_alias
from nureongi.chain import RAPTOR_APPLIED, RAPTOR_REQUIRED
from observability.mongo_logger import log_api_event

# >>> 추가: LangSmith 트레이서/컨피그 유틸 가져오기
from observability.langsmith import make_tracer_for, build_trace_config

router = APIRouter(tags=["redfin_target-insight"])

def _resolve_persona(user_persona: str | None) -> str:
    if not user_persona:
        return "ai_industry_professional"
    spec = get_persona_by_alias(user_persona)
    return spec.slug.value if spec else user_persona

@router.post(
    "/redfin_target-insight",
    response_model=QueryResponseV1,
    summary="RAG 답변 생성 (redfin_target-insight)",
    description="사용자 입력 프롬프트를 받아 RAG로 인사이트를 생성합니다."
)
def redfin_target_insight(req: QueryRequest, request: Request):
    request_received_at = datetime.now(timezone.utc)
    start = perf_counter()

    plan = choose_strategy_advanced(
        question=req.question,
        est_context_tokens=None,
        k_hint=req.top_k,
        doc_count_hint=None,
        token_budget=settings.app.token_budget,
    )
    rag_service.set_search_kwargs(k=plan.k, fetch_k=plan.fetch_k, lambda_mult=plan.lambda_mult)

    persona_slug = _resolve_persona(req.persona)
    strategy = plan.strategy if req.strategy == "auto" else req.strategy

    req_meta: Dict[str, Any] = {
        "service": settings.app.service_name,
        "question": req.question,
        "top_k": plan.k,
        "fetch_k": plan.fetch_k,
        "lambda_mult": plan.lambda_mult,
        "plan": asdict(plan),
    }

    # >>> 추가: 이 요청만 LangSmith 프로젝트(api_rag 등)로 기록되도록 트레이서/컨피그 생성
    tracer = make_tracer_for("redfin_target-insight")
    run_cfg = build_trace_config(
        service_name="redfin_target-insight",
        user_id=(req.user_id or "notuser"),
        plan=asdict(plan),
    )
    run_cfg["callbacks"] = [tracer]  # 중요: 요청 단위로 콜백 주입

    try:
        resp = rag_service.run_query(
            question=req.question,
            persona=persona_slug,
            strategy=strategy,
            user_id=(req.user_id or "notuser"),
            req_meta=req_meta,
            service_name=settings.app.service_name,
            # >>> 추가: 요청 단위 트레이스 설정 전달
            run_config=run_cfg,
        )

        elapsed_ms = int((perf_counter() - start) * 1000)
        response_generated_at = datetime.now(timezone.utc)

        try:
            envelope = resp.dict() if hasattr(resp, "dict") else resp
            answer_text = ""
            try:
                if isinstance(envelope, dict):
                    data = envelope.get("data") or {}
                    ans = data.get("answer") or {}
                    answer_text = ans.get("text") or ""
            except Exception:
                pass

            log_api_event(
                envelope=envelope,
                status=200,
                endpoint=str(request.url.path),
                error=None,
                extra={
                    "request_received_at": request_received_at.isoformat(),
                    "response_generated_at": response_generated_at.isoformat(),
                    "duration_ms": elapsed_ms,
                    "client": {
                        "ip": getattr(request.client, "host", None),
                        "user_agent": request.headers.get("user-agent"),
                        "origin": request.headers.get("origin"),
                    },
                    "request_meta": {
                        "persona": persona_slug,
                        "strategy": strategy,
                        "top_k": plan.k,
                        "fetch_k": plan.fetch_k,
                        "lambda_mult": plan.lambda_mult,
                        "user_id": req.user_id or "notuser",
                    },
                    "retrieval": {
                        "raptor_required": bool(RAPTOR_REQUIRED),
                        "raptor_applied": bool(RAPTOR_APPLIED.get()),
                        "top_k": plan.k,
                        "fetch_k": plan.fetch_k,
                        "lambda_mult": plan.lambda_mult,
                    },
                    "answer_meta": {
                        "chars": len(answer_text),
                        "tokens": None,
                    },
                    "service": settings.app.service_name,
                },
            )
        except Exception as log_err:
            print(f"[warn] Mongo log failed: {log_err}")

        return resp

    except Exception as e:
        try:
            log_api_event(
                envelope={
                    "version": "v1",
                    "data": {
                        "answer": {"text": "", "bullets": None, "format": "markdown"},
                        "persona": None,
                        "strategy": None,
                        "sources": [],
                    },
                    "meta": {
                        "user": {"user_id": req.user_id or "notuser", "session_id": None},
                        "request": {
                            "service": settings.app.service_name,
                            "question": req.question,
                            "top_k": plan.k,
                            "fetch_k": plan.fetch_k,
                            "lambda_mult": plan.lambda_mult,
                            "plan": asdict(plan),
                        },
                        "pipeline": None,
                    },
                },
                status=500,
                endpoint=str(request.url.path),
                error=str(e),
                extra={
                    "request_received_at": request_received_at.isoformat(),
                    "response_generated_at": datetime.now(timezone.utc).isoformat(),
                    "service": settings.app.service_name,
                    "retrieval": {
                        "raptor_required": bool(RAPTOR_REQUIRED),
                        "raptor_applied": bool(RAPTOR_APPLIED.get()),
                        "top_k": plan.k,
                        "fetch_k": plan.fetch_k,
                        "lambda_mult": plan.lambda_mult,
                    },
                },
            )
        except Exception as log_err:
            print(f"[warn] Mongo log failed on error: {log_err}")

        raise HTTPException(status_code=500, detail="internal error")
