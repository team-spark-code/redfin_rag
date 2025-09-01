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
            "http://localhost:8000",
            "http://localhost:8001",
        ],
        validation_alias=AliasChoices("ALLOWED_ORIGINS")
    )

    service_name: str = Field(default="redfin_target-insight", validation_alias=AliasChoices("SERVICE_NAME"))


class RagSettings(BaseModel):
    emb_model: str = Field(default="BAAI/bge-base-en-v1.5", validation_alias=AliasChoices("EMB_MODEL"))


class NewsSettings(BaseModel):
    api_url: str = Field(default="http://192.168.0.123:8000/news/extract", validation_alias=AliasChoices("NEWS_API_URL"))

    # 청킹/프롬프트/컬렉션
    chunk_strategy: str = Field(default="default", validation_alias=AliasChoices("NEWS_CHUNK_STRATEGY"))
    prompt_path: str = Field(default="src/prompts/templates/news_publish_v1.md", validation_alias=AliasChoices("NEWS_PROMPT_PATH"))

    # 벡터 DB 상의 컬렉션명 (뉴스용)
    collection: str = Field(default="news_default_v1", validation_alias=AliasChoices("NEWS_COLLECTION"))
    
    # 추가 코드: 자동 출간 여부를 제어하는 환경변수
    seed_on_startup: bool = Field(
        default=True,
        validation_alias=AliasChoices("NEWS_SEED_ON_STARTUP", "news_seed_on_startup"),
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
    news_collection: str = Field(default="news_semantic_v1", validation_alias=AliasChoices("NEWS_COL"))
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
