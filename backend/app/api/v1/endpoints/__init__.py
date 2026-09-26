from backend.app.api.v1.endpoints.documents import router as documents_router
from backend.app.api.v1.endpoints.health import router as health_router
from backend.app.api.v1.endpoints.ingestion import router as ingestion_router

__all__ = ["health_router", "documents_router", "ingestion_router"]
