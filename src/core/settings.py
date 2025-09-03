from __future__ import annotations

# .env를 현재 작업폴더 기준으로 상위 경로까지 탐색해서 명시적으로 로드
from dotenv import load_dotenv, find_dotenv
load_dotenv(dotenv_path=find_dotenv(usecwd=True), override=True)

from typing import List, Literal
from pydantic import BaseModel, Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


# ───────── Sub-settings ─────────

class AppSettings(BaseModel):
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("HOST"))
    port: int = Field(default=8030,       validation_alias=AliasChoices("PORT"))
    token_budget: int = Field(default=3500, validation_alias=AliasChoices("TOKEN_BUDGET"))

    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5500",
            "http://localhost:8000",
            "http://localhost:8001",
            "http://localhost:8030",
            "http://127.0.0.1:5500",
            "http://192.168.0.23:5500",
            "http://192.168.0.23:8030",
            "http://192.168.0.123:5500",
            "http://192.168.0.123:8030",
        ],
        validation_alias=AliasChoices("ALLOWED_ORIGINS"),
    )

    cors_origin_regex: str = Field(
        default=r"^https?://192\.168\.0\.\d{1,3}(:\d+)?$",
        validation_alias=AliasChoices("CORS_ORIGIN_REGEX"),
    )

    service_name: str = Field(
        default="redfin_target-insight",
        validation_alias=AliasChoices("SERVICE_NAME"),
    )


class RagSettings(BaseModel):
    emb_model: str = Field(default="BAAI/bge-base-en-v1.5",
                           validation_alias=AliasChoices("EMB_MODEL"))
    # 필요 시: collection, persist_dir 등 추가


class NewsSettings(BaseModel):
    """
    NEWS는 가급적 NEWS__* 중첩 키만 사용.
    과거 호환: api_url(NEWS_API_URL) / seed_on_startup만 유지.
    """
    prompt_path: str = "src/prompts/templates/news_publish_v1.md"

    # 과거 호환 유지
    api_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NEWS_API_URL", "NEWS__API_URL")
    )

    # 뉴스 전용 벡터 컬렉션/저장경로
    collection: str = "news_logs"
    persist_dir: str = "./.chroma"

    use_llm: bool = True
    top_k: int = 6
    recency_days: int | None = None
    default_publish: bool = True

    # 수집 소스: NEWS__INGEST_SOURCE=mongo|http
    ingest_source: Literal["http", "mongo"] = "http"

    # Mongo 원본 컬렉션: NEWS__SOURCE_COLLECTION=extract
    source_collection: str = "extract"

    # 관측/네이밍
    langsmith_project: str = "redfin_news-publish"
    service_name: str = "redfin_news"

    # (선택) 보강 파라미터
    enable_enrichment: bool = False
    enrich_k: int = 3
    enrich_fetch_k: int = 8
    enrich_lambda: float = 0.2

    emb_model: str = "BAAI/bge-base-en-v1.5"

    # 출간본 벡터/자동 인덱싱
    vector_collection_posts: str = "news_posts_v1"
    index_on_publish: bool = True

    # 서버 기동 시 뉴스 시드 여부 (과거 호환 포함)
    seed_on_startup: bool = Field(
        default=True,
        validation_alias=AliasChoices("NEWS__SEED_ON_STARTUP", "NEWS_SEED_ON_STARTUP"),
    )


class MongoSettings(BaseModel):
    """
    권장: MONGO__* 중첩키 사용.
    과거 호환: MONGODB_URI, MONGO_URI, NEWS__COL 등도 함께 허용.
    """
    uri: str = Field(
        default="mongodb://admin:Redfin7620%21@192.168.0.123:27017/redfin?authSource=admin",
        validation_alias=AliasChoices(
            "MONGO__URI",          # 권장 nested
            "MONGODB__URI",        # 사용 중이면 유지
            "MONGODB_URI",         # 과거 단일 키
            "MONGO_URI",           # 과거 단일 키
        ),
    )
    db: str = Field(
        default="redfin",
        validation_alias=AliasChoices("MONGO__DB", "MONGO_DB"),
    )
    logs_collection: str = Field(
        default="rag_logs",
        validation_alias=AliasChoices("MONGO__LOGS_COLLECTION", "MONGO_COL"),
    )
    news_collection: str = Field(
        default="news_logs",
        validation_alias=AliasChoices(
            "MONGO__NEWS_COLLECTION",  # 권장 nested
            "NEWS__COL",               # 과거 다른 섹션 키
            "NEWS_COL",                # 과거 다른 섹션 키
        ),
    )
    timeout_ms: int = Field(
        default=3000,
        validation_alias=AliasChoices("MONGO__TIMEOUT_MS", "MONGO_TIMEOUT_MS"),
    )


# ───────── Root settings ─────────

class Settings(BaseSettings):
    """
    env_nested_delimiter='__' 로 NEWS__INGEST_SOURCE 등 nested 키 자동 매핑.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppSettings = AppSettings()
    rag: RagSettings = RagSettings()
    news: NewsSettings = NewsSettings()
    mongo: MongoSettings = MongoSettings()


# 전역 인스턴스
settings = Settings()
__all__ = ["settings", "Settings", "AppSettings", "RagSettings", "NewsSettings", "MongoSettings"]
