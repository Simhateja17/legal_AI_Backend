from pathlib import Path
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str

    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_chat_deployment: str = "gpt-4o-mini"

    # Google Gemini (optional)
    google_api_key: str = ""

    # Application
    llm_provider: Literal["azure", "gemini"] = "gemini"
    log_level: str = "INFO"
    environment: str = "development"

    # RAG defaults
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.3
    max_context_tokens: int = 4000
    max_conversation_turns: int = 10
    enable_guard: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
