"""Unit tests for Milestone 10 production configuration safeguards and validation."""

import pytest
from pydantic import ValidationError
from backend.app.config.settings import Settings


def test_development_configuration_defaults():
    """Verify that development mode allows safe local defaults."""
    s = Settings(ENVIRONMENT="development", DEBUG=True)
    assert s.ENVIRONMENT == "development"
    assert s.DEBUG is True
    assert s.DATABASE_SQLITE_FALLBACK is True
    assert s.QDRANT_LOCAL_FALLBACK is True
    assert s.DATABASE_POOL_SIZE == 10
    assert s.CACHE_ENABLED is True
    assert s.CACHE_EMBEDDING_MAX_SIZE == 20000


def test_production_rejects_default_secret_key():
    """Verify production mode rejects the default placeholder SECRET_KEY."""
    with pytest.raises(ValidationError, match="SECRET_KEY must be a secure random string"):
        Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            SECRET_KEY="change-this-in-production-to-a-secure-random-secret-key",
            DATABASE_SQLITE_FALLBACK=False,
            QDRANT_LOCAL_FALLBACK=False,
        )


def test_production_rejects_debug_true():
    """Verify production mode rejects DEBUG=True."""
    with pytest.raises(ValidationError, match="DEBUG must be set to False"):
        Settings(
            ENVIRONMENT="production",
            DEBUG=True,
            SECRET_KEY="a-very-secure-random-secret-key-for-prod-32-chars",
            DATABASE_SQLITE_FALLBACK=False,
            QDRANT_LOCAL_FALLBACK=False,
        )


def test_production_rejects_sqlite_fallback():
    """Verify production mode rejects SQLite fallback."""
    with pytest.raises(ValidationError, match="DATABASE_SQLITE_FALLBACK must be set to False"):
        Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            SECRET_KEY="a-very-secure-random-secret-key-for-prod-32-chars",
            DATABASE_SQLITE_FALLBACK=True,
            QDRANT_LOCAL_FALLBACK=False,
        )


def test_production_rejects_qdrant_local_fallback():
    """Verify production mode rejects Qdrant in-memory fallback."""
    with pytest.raises(ValidationError, match="QDRANT_LOCAL_FALLBACK must be set to False"):
        Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            SECRET_KEY="a-very-secure-random-secret-key-for-prod-32-chars",
            DATABASE_SQLITE_FALLBACK=False,
            QDRANT_LOCAL_FALLBACK=True,
        )


def test_production_rejects_missing_openai_key():
    """Verify production mode rejects unconfigured OpenAI API key when provider is openai."""
    with pytest.raises(ValidationError, match="LLM_API_KEY must be configured"):
        Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            SECRET_KEY="a-very-secure-random-secret-key-for-prod-32-chars",
            DATABASE_SQLITE_FALLBACK=False,
            QDRANT_LOCAL_FALLBACK=False,
            LLM_PROVIDER="openai",
            LLM_API_KEY=None,
        )


def test_valid_production_configuration_succeeds():
    """Verify that a compliant production configuration validates successfully."""
    s = Settings(
        ENVIRONMENT="production",
        DEBUG=False,
        SECRET_KEY="a-very-secure-random-secret-key-for-prod-32-chars",
        DATABASE_SQLITE_FALLBACK=False,
        QDRANT_LOCAL_FALLBACK=False,
        LLM_PROVIDER="mock",
    )
    assert s.ENVIRONMENT == "production"
    assert s.DEBUG is False
    assert s.DATABASE_SQLITE_FALLBACK is False
    assert s.QDRANT_LOCAL_FALLBACK is False
