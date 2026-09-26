import pytest
from backend.app.services.vector_service import VectorService


def test_vector_service_initialization(in_memory_vector_service: VectorService):
    """Verify vector service initializes and reports connection status."""
    assert in_memory_vector_service.is_connected() is True


def test_vector_service_ensure_collection(in_memory_vector_service: VectorService):
    """Verify collection creation with custom dimension."""
    test_collection = "test_eval_collection"
    success = in_memory_vector_service.ensure_collection(
        collection_name=test_collection, vector_size=384
    )
    assert success is True

    collections = in_memory_vector_service.get_collections()
    assert test_collection in collections


def test_vector_service_health_check(in_memory_vector_service: VectorService):
    """Verify health check diagnostics format and no sensitive data leakage."""
    health = in_memory_vector_service.check_health()
    assert health["status"] == "connected"
    assert "latency_ms" in health
    assert "collections" in health
    assert "api_key" not in health
    assert "password" not in health


def test_vector_service_client_unavailable(monkeypatch):
    """Verify clean handling when remote Qdrant is unreachable and fallback disabled."""
    from backend.app.config.settings import Settings

    strict_settings = Settings(
        QDRANT_URL="http://192.0.2.1:6333",  # Non-routable test IP
        QDRANT_LOCAL_FALLBACK=False,
        QDRANT_TIMEOUT_SECONDS=0.1,
    )
    strict_service = VectorService(settings=strict_settings)
    assert strict_service.is_connected() is False

    health = strict_service.check_health()
    assert health["status"] == "unreachable"
    assert "latency_ms" in health
