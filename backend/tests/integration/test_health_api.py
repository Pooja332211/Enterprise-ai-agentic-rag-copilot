import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_liveness_endpoint(async_client: AsyncClient):
    """Verify GET /health returns HTTP 200 and basic liveness payload."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
    assert "X-Process-Time" in response.headers


@pytest.mark.asyncio
async def test_api_v1_liveness_endpoint(async_client: AsyncClient):
    """Verify GET /api/v1/health returns HTTP 200."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_readiness_endpoint(async_client: AsyncClient):
    """Verify GET /health/ready returns service-level diagnostics."""
    response = await async_client.get("/health/ready")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "database" in data["services"]
    assert "vector_store" in data["services"]
    assert "models" in data

    # Verify no credentials leaked
    raw_text = response.text.lower()
    assert "password" not in raw_text
    assert "secret" not in raw_text
    assert "api_key" not in raw_text
    assert "postgres:" not in raw_text


@pytest.mark.asyncio
async def test_openapi_docs_endpoint(async_client: AsyncClient):
    """Verify OpenAPI documentation is reachable."""
    response = await async_client.get("/docs")
    assert response.status_code == 200
