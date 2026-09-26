import pytest
from backend.app.config.settings import Settings


def test_default_settings_instantiation():
    """Verify settings initialize with expected defaults."""
    settings = Settings()
    assert settings.PROJECT_NAME == "Enterprise AI Knowledge & Research Copilot"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.QDRANT_COLLECTION_NAME == "enterprise_knowledge"
    assert settings.EMBEDDING_DIMENSION == 384
    assert settings.DEBUG is not None


def test_cors_origins_parsing():
    """Verify CORS origins correctly parse comma-separated or JSON list formats."""
    # Test JSON string parsing
    s1 = Settings(BACKEND_CORS_ORIGINS='["http://localhost:3000", "http://example.com"]')
    assert "http://localhost:3000" in s1.BACKEND_CORS_ORIGINS
    assert "http://example.com" in s1.BACKEND_CORS_ORIGINS

    # Test comma-separated string parsing
    s2 = Settings(BACKEND_CORS_ORIGINS="http://localhost:3000, http://test.org")
    assert "http://localhost:3000" in s2.BACKEND_CORS_ORIGINS
    assert "http://test.org" in s2.BACKEND_CORS_ORIGINS


def test_database_url_assembly():
    """Verify computed database URLs correctly format connection strings."""
    settings = Settings(
        _env_file=None,
        DATABASE_URL=None,
        DATABASE_URL_SYNC=None,
        POSTGRES_SERVER="db.internal",
        POSTGRES_PORT=5432,
        POSTGRES_USER="copilot_user",
        POSTGRES_PASSWORD="securepassword",
        POSTGRES_DB="copilot_prod",
    )
    assert settings.async_database_url == (
        "postgresql+asyncpg://copilot_user:securepassword@db.internal:5432/copilot_prod"
    )
    assert settings.sync_database_url == (
        "postgresql+psycopg2://copilot_user:securepassword@db.internal:5432/copilot_prod"
    )


def test_qdrant_url_resolution():
    """Verify resolved Qdrant URL generation."""
    s1 = Settings(_env_file=None, QDRANT_URL=None, QDRANT_HOST="vector-host", QDRANT_PORT=6333)
    assert s1.resolved_qdrant_url == "http://vector-host:6333"

    s2 = Settings(_env_file=None, QDRANT_URL="https://remote-qdrant.cloud:443")
    assert s2.resolved_qdrant_url == "https://remote-qdrant.cloud:443"
