from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("HOST"))
    port: int = Field(default=8030, validation_alias=AliasChoices("PORT"))
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

    cors_origin_regex: str = Field(
        default=r"^https?://192\.168\.0\.\d{1,3}(:\d+)?$",
        validation_alias=AliasChoices("CORS_ORIGIN_REGEX"),
    )

    service_name: str = Field(
        default="redfin_target-insight",
        validation_alias=AliasChoices("SERVICE_NAME"),
    )


class RagSettings(BaseModel):
    emb_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        validation_alias=AliasChoices("EMB_MODEL"),
    )
    langsmith_project: str = Field(
        default="redfin_target-insight",
        validation_alias=AliasChoices(
            "RAG_LANGSMITH_PROJECT",
            "LANGCHAIN_PROJECT_REDFIN_TARGET",
            "RAG__LANGSMITH_PROJECT",
        ),
    )


class NewsSettings(BaseModel):
    api_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NEWS_API_URL", "API_URL", "NEWS__API_URL"),
    )
    ingest_source: str = Field(
        default="mongo",
        validation_alias=AliasChoices("NEWS_INGEST_SOURCE", "NEWS__INGEST_SOURCE"),
    )
    source_collection: str = Field(
        default="extract",
        validation_alias=AliasChoices("NEWS_SOURCE_COLLECTION", "NEWS__SOURCE_COLLECTION"),
    )


class MongoSettings(BaseModel):
    uri: str = Field(
        default=None,
        validation_alias=AliasChoices("MONGODB_URI", "MONGO_URI"),
    )
    db: str = Field(
        default="redfin",
        validation_alias=AliasChoices("MONGO_DB"),
    )
    logs_collection: str = Field(
        default="rag_logs",
        validation_alias=AliasChoices("MONGO_LOGS_COLLECTION", "MONGO_COL"),
    )
    timeout_ms: int = Field(
        default=3000,
        validation_alias=AliasChoices("MONGO_TIMEOUT_MS"),
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = AppSettings()
    rag: RagSettings = RagSettings()
    news: NewsSettings = NewsSettings()
    mongo: MongoSettings = MongoSettings()


settings = Settings()
__all__ = [
    "settings",
    "Settings",
    "AppSettings",
    "RagSettings",
    "NewsSettings",
    "MongoSettings",
]
