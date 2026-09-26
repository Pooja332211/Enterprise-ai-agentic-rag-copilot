from fastapi import APIRouter
from backend.app.api.v1.endpoints.documents import router as documents_router
from backend.app.api.v1.endpoints.health import router as health_router
from backend.app.api.v1.endpoints.index import router as index_router
from backend.app.api.v1.endpoints.ingestion import router as ingestion_router
from backend.app.api.v1.endpoints.research import router as research_router
from backend.app.api.v1.endpoints.search import router as search_router

api_v1_router = APIRouter()

# Register endpoint routers
api_v1_router.include_router(health_router, tags=["Health & Diagnostics"])
api_v1_router.include_router(documents_router, prefix="/documents", tags=["Documents"])
api_v1_router.include_router(ingestion_router, prefix="/ingestion", tags=["Ingestion"])
api_v1_router.include_router(search_router, prefix="/search", tags=["Search & Retrieval"])
api_v1_router.include_router(research_router, prefix="/research", tags=["Grounded Research & Synthesis"])
api_v1_router.include_router(index_router, prefix="/index", tags=["Index Management"])
