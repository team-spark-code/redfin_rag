# src/core/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core import settings
from observability.mongo_logger import init_mongo, ensure_collections, get_news_collection

# 두 파이프라인 로드
from services import rag_service
from services import news_service
from services.news_service import publish_from_env  # 동기 함수

# LangSmith: 요청 단위 프로젝트 분리를 위해 tracer/run_config 생성
from observability.langsmith import make_tracer_explicit, build_trace_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === startup ===
    init_mongo()
    ensure_collections()

    # 진단 출력
    print("[diag] settings.news.seed_on_startup =", settings.news.seed_on_startup)
    print("[diag] settings.news.api_url         =", settings.news.api_url)
    print("[diag] settings.news.collection      =", settings.news.collection)
    print("[diag] settings.news.prompt_path     =", settings.news.prompt_path)

    # (1) 일반 RAG 인덱스 초기화: /redfin_target-insight 에서 사용
    try:
        if hasattr(rag_service, "init_index"):
            if not settings.news.api_url:
                print("[rag init_index] skipped: settings.news.api_url is empty")
            else:
                res_rag = rag_service.init_index(
                    news_url=settings.news.api_url,          # 피드 URL
                    emb_model=settings.rag.emb_model,        # 예: "BAAI/bge-base-en-v1.5"
                    chunk_size=1200,                          # 고정 청킹 크기
                    chunk_overlap=120,                        # 10% 오버랩
                    use_raptor=False,                         # 출간/질의 속도 우선이면 False 권장
                    distance="cosine",                        # 벡터 거리 메트릭
                )
                print("[rag init_index] ok:", res_rag if res_rag is not None else "initialized")
        else:
            print("[rag init_index] skipped: function not found")
    except Exception as e:
        print("[warn] rag init_index failed:", e)

    # (2) 서버 시작 시 뉴스 자동 출간 (시드) —— 반드시 '뉴스' 프로젝트로 기록되게 run_config 주입
    created = 0
    if bool(getattr(settings.news, "seed_on_startup", False)) and settings.news.api_url:
        try:
            print("[news] seeding from settings.news.api_url on startup …")

            # ★ 뉴스 전용 LangSmith run_config (프로젝트 분리 핵심)
            tracer = make_tracer_explicit(
                getattr(settings.news, "langsmith_project", None) or "redfin_news-publish"
            )
            run_cfg = build_trace_config(
                service_name="redfin_news",  # 서비스 메타는 'redfin_news'로 고정
                user_id="system",
                plan=None,
            )
            run_cfg["callbacks"] = [tracer] if tracer else []

            # 부팅 시드에도 run_config를 전달해야 뉴스 프로젝트로 기록됩니다
            result = publish_from_env(run_config=run_cfg)
            created = len(result.get("created", []))
            skipped = len(result.get("skipped", []))
            errors = len(result.get("errors", []))
            print(f"[news] seed result: created={created} skipped={skipped} errors={errors}")
        except Exception as e:
            print(f"[news] seed failed: {e}")
    else:
        print("[news] seeding skipped: seed_on_startup=False or api_url missing")

    # (3) 뉴스 인덱스 초기화 —— 반드시 '시드 후'에 실행 (문서 없으면 스킵)
    try:
        n_docs = 0
        col = get_news_collection()
        if col is not None:
            try:
                n_docs = col.count_documents({})
            except Exception:
                n_docs = 0

        if hasattr(news_service, "init_news_index_fixed"):
            if n_docs > 0 or created > 0:
                res_fixed = news_service.init_news_index_fixed(
                    chunk_size=1200,   # 1,200~1,400 권장
                    chunk_overlap=120  # 10%
                )
                print("[news fixed index]", res_fixed)
            else:
                print("[news fixed index] skipped: no docs in collection")
        else:
            print("[news fixed index] skipped: function not found")
    except Exception as e:
        print("[warn] news fixed index init failed:", e)

    # === app running ===
    yield
    # === shutdown ===
