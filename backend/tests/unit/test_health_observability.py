"""Unit tests for system health and readiness verification across all 6 dependencies."""

import pytest
from backend.app.services.health_service import get_system_health, get_system_readiness


@pytest.mark.asyncio
async def test_get_system_health():
    health = await get_system_health()
    assert health["status"] == "healthy"
    assert "app" in health
    assert "version" in health
    assert "environment" in health
    assert "timestamp" in health


@pytest.mark.asyncio
async def test_get_system_readiness_structure():
    readiness = await get_system_readiness()

    assert readiness["status"] in ("ready", "degraded", "unhealthy")
    assert "services" in readiness
    assert "models" in readiness

    services = readiness["services"]
    required_components = [
        "database",
        "vector_store",
        "embedding_provider",
        "reranker",
        "llm_provider",
        "bm25_index",
    ]
    for comp in required_components:
        assert comp in services, f"Component '{comp}' missing from readiness services"
        comp_status = services[comp].get("status")
        assert comp_status in (
            "healthy",
            "degraded",
            "unavailable",
            "not_configured",
        ), f"Invalid status '{comp_status}' for component '{comp}'"


@pytest.mark.asyncio
async def test_readiness_non_fatal_resilience(monkeypatch):
    from backend.app.config.settings import get_settings

    settings = get_settings()
    # Temporarily set OpenAI with no API key to simulate unconfigured external LLM
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_API_KEY", "")

    readiness = await get_system_readiness()
    llm_info = readiness["services"]["llm_provider"]
    assert llm_info["status"] == "not_configured"
    # Overall status should degrade gracefully, NOT crash
    assert readiness["status"] in ("ready", "degraded")
