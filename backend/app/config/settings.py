import json
from functools import lru_cache
from typing import List, Literal, Optional, Union
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application settings loaded from environment variables and .env file.
    Validates configuration and provides defaults for local and production setups.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --------------------------------------------------------------------------
    # Application & Environment
    # --------------------------------------------------------------------------
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = True
    PROJECT_NAME: str = "Enterprise AI Knowledge & Research Copilot"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "change-this-in-production-to-a-secure-random-secret-key"

    # --------------------------------------------------------------------------
    # CORS
    # --------------------------------------------------------------------------
    BACKEND_CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, str) and v.startswith("["):
            try:
                return json.loads(v)
            except Exception:
                return [v]
        return v

    # --------------------------------------------------------------------------
    # PostgreSQL Database
    # --------------------------------------------------------------------------
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "enterprise_copilot"

    DATABASE_URL: Optional[str] = None
    DATABASE_URL_SYNC: Optional[str] = None
    DATABASE_SQLITE_FALLBACK: bool = True
    DATABASE_SQLITE_PATH: str = "./data/copilot_dev.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: float = 30.0
    DATABASE_POOL_RECYCLE: int = 1800

    @property
    def async_database_url(self) -> str:
        """Returns the async database connection string."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def sync_database_url(self) -> str:
        """Returns the synchronous database connection string for migrations."""
        if self.DATABASE_URL_SYNC:
            return self.DATABASE_URL_SYNC
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # --------------------------------------------------------------------------
    # Qdrant Vector Database
    # --------------------------------------------------------------------------
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "enterprise_knowledge"
    QDRANT_LOCAL_FALLBACK: bool = True
    QDRANT_TIMEOUT_SECONDS: float = 3.0

    @property
    def resolved_qdrant_url(self) -> str:
        """Returns the base Qdrant HTTP URL."""
        if self.QDRANT_URL:
            return self.QDRANT_URL
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    # --------------------------------------------------------------------------
    # LLM & Generation Configuration (Milestone 5)
    # --------------------------------------------------------------------------
    LLM_PROVIDER: str = "mock"  # mock, openai, anthropic, local
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_BASE: Optional[str] = "https://api.openai.com/v1"
    LLM_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_API_BASE: Optional[str] = "https://api.anthropic.com"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: float = 30.0
    CITATION_MIN_SUPPORT_THRESHOLD: float = 0.40
    CITATION_PARTIAL_SUPPORT_THRESHOLD: float = 0.20

    # --------------------------------------------------------------------------
    # Embeddings & Lexical Retrieval (Milestone 3)
    # --------------------------------------------------------------------------
    EMBEDDING_PROVIDER: str = "fastembed"  # fastembed, openai, mock
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 64
    BM25_INDEX_PATH: str = "./data/indexes/bm25_index.pkl"
    RETRIEVAL_DEFAULT_TOP_K: int = 10
    RETRIEVAL_RRF_K: int = 60

    # --------------------------------------------------------------------------
    # Cross-Encoder Reranking & Context Assembly (Milestone 4)
    # --------------------------------------------------------------------------
    RERANKER_PROVIDER: str = "bge"  # bge, mock
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_BATCH_SIZE: int = 32
    RERANKER_ENABLED: bool = True
    RERANKER_CANDIDATE_TOP_K: int = 20
    RETRIEVAL_FINAL_TOP_K: int = 5
    CONTEXT_MAX_TOKENS: int = 2000
    CONTEXT_MAX_CHARACTERS: int = 8000
    MIN_RELEVANCE_THRESHOLD: Optional[float] = None

    # --------------------------------------------------------------------------
    # Storage & Upload Limits
    # --------------------------------------------------------------------------
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "txt", "md", "html"]

    # --------------------------------------------------------------------------
    # Document Chunking Configuration
    # --------------------------------------------------------------------------
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MIN_CHUNK_SIZE: int = 50

    # --------------------------------------------------------------------------
    # Observability
    # --------------------------------------------------------------------------
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: Optional[str] = "https://cloud.langfuse.com"

    # --------------------------------------------------------------------------
    # Caching Configuration (Milestone 9)
    # --------------------------------------------------------------------------
    CACHE_ENABLED: bool = True
    CACHE_EMBEDDING_MAX_SIZE: int = 20000
    CACHE_EMBEDDING_TTL_SECONDS: int = 86400
    CACHE_RERANKER_MAX_SIZE: int = 20000
    CACHE_RERANKER_TTL_SECONDS: int = 86400

    # --------------------------------------------------------------------------
    # API Limits & Timeouts
    # --------------------------------------------------------------------------
    MAX_SUBQUERIES_LIMIT: int = 5
    MAX_RETRIEVAL_TOP_K: int = 50
    API_REQUEST_TIMEOUT_SECONDS: float = 60.0

    # --------------------------------------------------------------------------
    # Production Validation
    # --------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if self.ENVIRONMENT == "production":
            default_secret = "change-this-in-production-to-a-secure-random-secret-key"
            if self.SECRET_KEY == default_secret or len(self.SECRET_KEY) < 32:
                raise ValueError(
                    "In production environment, SECRET_KEY must be a secure random string of at least 32 characters."
                )
            if self.DEBUG:
                raise ValueError("In production environment, DEBUG must be set to False.")
            if self.DATABASE_SQLITE_FALLBACK:
                raise ValueError("In production environment, DATABASE_SQLITE_FALLBACK must be set to False.")
            if self.QDRANT_LOCAL_FALLBACK:
                raise ValueError("In production environment, QDRANT_LOCAL_FALLBACK must be set to False.")
            if self.LLM_PROVIDER.lower() == "openai" and not self.LLM_API_KEY:
                raise ValueError("In production environment with LLM_PROVIDER='openai', LLM_API_KEY must be configured.")
            if self.LLM_PROVIDER.lower() == "anthropic" and not self.ANTHROPIC_API_KEY:
                raise ValueError("In production environment with LLM_PROVIDER='anthropic', ANTHROPIC_API_KEY must be configured.")
        return self



@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
