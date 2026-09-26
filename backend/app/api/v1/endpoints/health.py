from typing import Any, Dict
from fastapi import APIRouter, Response, status
from backend.app.services.health_service import get_system_health, get_system_readiness

router = APIRouter()


@router.get(
    "/health",
    summary="Liveness Check",
    description="Returns HTTP 200 if the FastAPI application process is up and running.",
    response_model=Dict[str, Any],
)
async def health_check() -> Dict[str, Any]:
    """Basic liveness probe."""
    return await get_system_health()


@router.get(
    "/health/ready",
    summary="Readiness Check",
    description="Evaluates connectivity to PostgreSQL and Qdrant. Returns status and diagnostic latencies.",
    response_model=Dict[str, Any],
)
async def readiness_check(response: Response) -> Dict[str, Any]:
    """Deep readiness probe checking database and vector store connectivity."""
    readiness = await get_system_readiness()
    if readiness["status"] == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return readiness
