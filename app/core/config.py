"""
Application configuration loaded from environment variables.

Uses Pydantic BaseSettings for validation and .env file loading.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root for .env resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AppSettings(BaseSettings):
    """Application metadata and runtime flags."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    name: str = Field(default="Enterprise RAG Platform", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")


class GroqSettings(BaseSettings):
    """Groq LLM API configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GROQ_",
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(default="", description="Groq API key")
    model: str = Field(default="llama-3.3-70b-versatile", description="Default model name")


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = Field(default="http://localhost:6333", description="Qdrant instance URL")
    api_key: Optional[str] = Field(default=None, description="API key (required for Qdrant Cloud)")
    collection_name: str = Field(default="documents", description="Default collection name")


class PostgresSettings(BaseSettings):
    """Neon PostgreSQL database configuration."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="",
        validation_alias="DATABASE_URL",
        description="PostgreSQL connection string",
    )


class RedisSettings(BaseSettings):
    """Upstash Redis configuration (REST API)."""

    model_config = SettingsConfigDict(
        env_prefix="UPSTASH_REDIS_",
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rest_url: str = Field(default="", description="Upstash Redis REST API URL")
    rest_token: str = Field(default="", description="Upstash Redis REST API token")


class LangfuseSettings(BaseSettings):
    """Langfuse observability configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    public_key: str = Field(default="", description="Langfuse public key (safe for frontend)")
    secret_key: str = Field(default="", description="Langfuse secret key (server-side only)")
    host: str = Field(default="https://cloud.langfuse.com", description="Langfuse host URL")


class HuggingFaceSettings(BaseSettings):
    """Hugging Face models configuration."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    token: str = Field(default="", validation_alias="HUGGINGFACE_TOKEN")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias="EMBEDDING_MODEL",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="RERANKER_MODEL",
    )


class Settings(BaseSettings):
    """Root settings aggregating all configuration groups."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings. Loads from .env on first call."""
    return Settings()
