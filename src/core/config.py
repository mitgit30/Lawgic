from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Legal Guidance Assistant", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    vector_db_path: Path = Field(default=Path("vector_db"), alias="VECTOR_DB_PATH")
    chroma_collection: str = Field(default="legal_chunks", alias="CHROMA_COLLECTION")

    embedding_model_name: str = Field(default="BAAI/bge-small-en", alias="EMBEDDING_MODEL_NAME")
    embedding_model_local_path: str | None = Field(default=None, alias="EMBEDDING_MODEL_LOCAL_PATH")
    legal_model_name: str = Field(default="law-ai/CustomInLawBERT", alias="LEGAL_MODEL_NAME")
    legal_model_local_path: str | None = Field(default=None, alias="LEGAL_MODEL_LOCAL_PATH")

    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_base_url: str = Field(default="https://ollama.com", alias="OLLAMA_BASE_URL")
    ollama_api_key: str | None = Field(default=None, alias="OLLAMA_API_KEY")
    ollama_model: str = Field(default="gpt-oss:120b", alias="OLLAMA_MODEL")
    ollama_temperature: float = Field(default=0.2, alias="OLLAMA_TEMPERATURE")
    ollama_timeout_seconds: int = Field(default=120, alias="OLLAMA_TIMEOUT_SECONDS")

    google_client_id: str | None = Field(default=None, alias="GOOGLE_CLIENT_ID")
    google_client_secret: str | None = Field(default=None, alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str | None = Field(default="http://localhost:8000/auth/google/callback", alias="GOOGLE_REDIRECT_URI")
    frontend_base_url: str = Field(default="http://localhost:8501", alias="FRONTEND_BASE_URL")
    backend_public_url: str = Field(default="http://localhost:8000", alias="BACKEND_PUBLIC_URL")
    auth_secret_key: str = Field(default="change-me-please", alias="AUTH_SECRET_KEY")
    auth_token_ttl_seconds: int = Field(default=86400, alias="AUTH_TOKEN_TTL_SECONDS")

    top_k_retrieval: int = Field(default=10, alias="TOP_K_RETRIEVAL")
    max_clause_tokens: int = Field(default=512, alias="MAX_CLAUSE_TOKENS")
    penalty_threshold_inr: float = Field(default=100000.0, alias="PENALTY_THRESHOLD_INR")
    tesseract_cmd: str | None = Field(default=None, alias="TESSERACT_CMD")
    ocr_lang: str = Field(default="eng", alias="OCR_LANG")
    ocr_dpi: int = Field(default=300, alias="OCR_DPI")

    backend_base_url: str = Field(default="http://localhost:8000", alias="BACKEND_BASE_URL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
