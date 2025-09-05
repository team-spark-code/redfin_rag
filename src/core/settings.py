from __future__ import annotations
from dotenv import load_dotenv ; load_dotenv()
from typing import List
from pydantic import BaseModel, Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

# ───────── Sub-settings (도메인별) ─────────

class AppSettings(BaseModel):
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("HOST"))
    port: int = Field(default=8030,       validation_alias=AliasChoices("PORT"))
    token_budget: int = Field(default=3500, validation_alias=AliasChoices("TOKEN_BUDGET"))

    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8001",
        ],
        validation_alias=AliasChoices("CORS_ORIGINS", "ALLOWED_ORIGINS"),
    )

    # 192.168.0.0/24 전체 허용 (http/https, 임의 포트)
    cors_origin_regex: str = Field(
        default=r"^https?://192\.168\.0\.\d{1,3}(:\d+)?$",
        validation_alias=AliasChoices("CORS_ORIGIN_REGEX"),
    )

    service_name: str = Field(default="redfin_target-insight", validation_alias=AliasChoices("SERVICE_NAME"))


class RagSettings(BaseModel):
    emb_model: str = Field(default="BAAI/bge-base-en-v1.5", validation_alias=AliasChoices("EMB_MODEL"))
    
    langsmith_project: str = Field(
        default="redfin_target-insight",
        validation_alias=AliasChoices("LANGCHAIN_PROJECT_REDFIN_TARGET", "RAG__LANGSMITH_PROJECT"),
    )


# [추가] NewsSettings에 인덱스/Enrichment 옵션을 명시적으로 둡니다.
class NewsSettings(BaseModel):
    prompt_path: str = "src/prompts/templates/news_publish_v1.md"

    # 기존: API URL (HTTP 사용 시에만 의미)
    api_url: str | None = Field(default=None, validation_alias=AliasChoices("NEWS__API_URL", "API_URL"))

    # [추가] 입력 소스 스위치 ("http" | "mongo")
    ingest_source: str = Field(
        default="mongo",
        validation_alias=AliasChoices("NEWS__INGEST_SOURCE")
    )

    # [추가] Mongo 입력 컬렉션명 (기본: extract)
    source_collection: str = Field(
        default="extract",
        validation_alias=AliasChoices("NEWS__SOURCE_COLLECTION")
    )
    
    collection: str = "news_logs"   # [추가] 뉴스 전용 Chroma 컬렉션명
    persist_dir: str = "./.chroma"         # [추가] Chroma 저장 경로(공용 가능)
    use_llm: bool = True
    recency_days: int = 14
    top_k: int = 6
    default_publish: bool = True

    # 관측/네이밍
    langsmith_project: str = Field(
        default="redfin_news-publish",
        validation_alias=AliasChoices("LANGCHAIN_PROJECT_REDFIN_NEWS", "NEWS__LANGSMITH_PROJECT"),
    )
    service_name: str = "redfin_news"

    # [추가] 컨텍스트 보강(선택) — 기본은 False로 꺼둠(출간은 편집/요약만)
    enable_enrichment: bool = False
    enrich_k: int = 3
    enrich_fetch_k: int = 8
    enrich_lambda: float = 0.2
    emb_model: str = "BAAI/bge-base-en-v1.5"   # [추가] 뉴스 인덱스/리트리버 임베딩
    
    # [추가] 우리 기사(출간본) 전용 벡터 컬렉션
    vector_collection_posts: str = "news_posts_v1"   # [추가]
    # [추가] 출간 직후 자동 인덱싱 여부
    index_on_publish: bool = True                    # [추가]
    
    # [신규] 서버 기동 시 뉴스 시드 여부(환경변수와 연결)
    seed_on_startup: bool = Field(
        default=True,
        validation_alias=AliasChoices("NEWS__SEED_ON_STARTUP", "NEWS_SEED_ON_STARTUP"),
    )


class MongoSettings(BaseModel):
    """
    기존 .env 키와 호환:
      - uri: MONGODB_URI 또는 MONGO_URI 지원
      - db: MONGO_DB
      - logs_collection: MONGO_COL
      - news_collection: NEWS_COL
      - timeout_ms: MONGO_TIMEOUT_MS
    """
    uri: str = Field(
        default="mongodb://admin:Redfin7620%21@192.168.0.123:27017/redfin?authSource=admin",
        validation_alias=AliasChoices("MONGODB_URI", "MONGO_URI")
    )
    db: str = Field(default="redfin", validation_alias=AliasChoices("MONGO_DB"))
    logs_collection: str = Field(default="rag_logs", validation_alias=AliasChoices("MONGO_COL"))
    news_collection: str = Field(default="news_logs", validation_alias=AliasChoices("NEWS_COL"))
    timeout_ms: int = Field(default=3000, validation_alias=AliasChoices("MONGO_TIMEOUT_MS"))


# ───────── Root settings ─────────

class Settings(BaseSettings):
    """
    최상위 Settings: .env 로드 + 서브 설정을 한 번에 구성
    - env_nested_delimiter="__" 덕분에 MONGO__URI 같은 nested 키도 사용 가능
      (기존 키도 AliasChoices로 호환하므로, 지금은 그대로 두셔도 됩니다)
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

# 전역 인스턴스 (권장 임포트: from core import settings)
settings = Settings()
__all__ = ["settings", "Settings", "AppSettings", "RagSettings", "NewsSettings", "MongoSettings"]

# src/core/settings.py (파일 끝 또는 앱 시작 지점에서 한 번만)
if __name__ == "__main__" or True:
    import os
    print("[boot] LANGCHAIN_PROJECT_REDFIN_NEWS =", os.getenv("LANGCHAIN_PROJECT_REDFIN_NEWS"))
    print("[boot] settings.news.langsmith_project =", Settings().news.langsmith_project)
