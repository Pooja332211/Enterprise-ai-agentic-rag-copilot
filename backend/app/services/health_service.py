from datetime import datetime, timezone
from typing import Any, Dict
from backend.app.config.settings import get_settings
from backend.app.database.session import check_database_health
from backend.app.services.bm25_service import get_bm25_service
from backend.app.services.embedding_service import get_embedding_service
from backend.app.services.reranker_service import get_reranker_service
from backend.app.services.vector_service import get_vector_service
from backend.app.utils.logger import logger


async def get_system_health() -> Dict[str, Any]:
    """
    Returns basic application liveness health information.
    Fast and light check for container orchestrator probes.
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def get_system_readiness() -> Dict[str, Any]:
    """
    Evaluates end-to-end readiness across all 6 application dependencies:
    1. database (PostgreSQL / SQLite)
    2. vector_store (Qdrant)
    3. embedding_provider (FastEmbed / OpenAI / Mock)
    4. reranker (BGE ONNX / Mock)
    5. llm_provider (Gemini / OpenAI / Mock)
    6. bm25_index (BM25Plus lexical engine)

    Assigns status enum: 'healthy' | 'degraded' | 'unavailable' | 'not_configured'.
    External dependencies failing or unconfigured do NOT make startup fatal.
    """
    settings = get_settings()

    # 1. Database
    try:
        db_health = await check_database_health()
        db_status = "healthy" if db_health.get("status") == "connected" else "unavailable"
    except Exception as exc:
        db_health = {"status": "error", "error": str(exc)}
        db_status = "unavailable"

    # 2. Vector Store
    try:
        vector_service = get_vector_service()
        qdrant_health = vector_service.check_health()
        vs_status = "healthy" if qdrant_health.get("status") == "connected" else "unavailable"
    except Exception as exc:
        qdrant_health = {"status": "error", "error": str(exc)}
        vs_status = "unavailable"

    # 3. Embedding Provider
    try:
        embedding_service = get_embedding_service()
        emb_status = "healthy"
        emb_details = {
            "provider": settings.EMBEDDING_PROVIDER,
            "model": settings.EMBEDDING_MODEL,
            "dimension": settings.EMBEDDING_DIMENSION,
            "cached": True,
        }
    except Exception as exc:
        emb_status = "degraded"
        emb_details = {"provider": settings.EMBEDDING_PROVIDER, "error": str(exc)}

    # 4. Reranker
    try:
        if not settings.RERANKER_ENABLED:
            reranker_status = "not_configured"
            reranker_details = {"enabled": False}
        else:
            reranker_service = get_reranker_service()
            reranker_status = "healthy"
            reranker_details = {
                "enabled": True,
                "provider": settings.RERANKER_PROVIDER,
                "model": reranker_service.model_name,
            }
    except Exception as exc:
        reranker_status = "degraded"
        reranker_details = {"enabled": settings.RERANKER_ENABLED, "error": str(exc)}

    # 5. LLM Provider
    provider = settings.LLM_PROVIDER.lower()
    has_key = bool(
        getattr(settings, "LLM_API_KEY", None)
        or getattr(settings, "OPENAI_API_KEY", None)
        or getattr(settings, "GEMINI_API_KEY", None)
    )
    if provider == "mock":
        llm_status = "healthy"
        llm_details = {"provider": "mock", "model": settings.LLM_MODEL}
    elif provider in ("gemini", "openai", "anthropic"):
        if has_key:
            llm_status = "healthy"
            llm_details = {"provider": provider, "model": settings.LLM_MODEL, "configured": True}
        else:
            llm_status = "not_configured"
            llm_details = {"provider": provider, "model": settings.LLM_MODEL, "warning": "API key not set"}
    else:
        llm_status = "degraded"
        llm_details = {"provider": provider, "warning": f"Unrecognized provider '{provider}'"}

    # 6. BM25 Index
    try:
        bm25_service = get_bm25_service()
        bm25_status = "healthy"
        bm25_details = {
            "indexed_chunks": len(bm25_service._chunks),
            "is_indexed": bm25_service._bm25 is not None,
            "index_path": str(bm25_service.index_path),
        }
    except Exception as exc:
        bm25_status = "unavailable"
        bm25_details = {"error": str(exc)}

    # Evaluate overall status
    component_statuses = [db_status, vs_status, emb_status, reranker_status, llm_status, bm25_status]

    # Primary infrastructure dependencies (Database & Vector Store) are critical for serving
    if db_status == "unavailable" or vs_status == "unavailable":
        overall_status = "unhealthy"
    elif any(s in ("unavailable", "degraded") for s in component_statuses):
        overall_status = "degraded"
    else:
        overall_status = "ready"

    return {
        "status": overall_status,
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "database": {**db_health, "status": db_status, "connection": db_health.get("status")},
            "vector_store": {**qdrant_health, "status": vs_status, "connection": qdrant_health.get("status")},
            "embedding_provider": {"status": emb_status, **emb_details},
            "reranker": {"status": reranker_status, **reranker_details},
            "llm_provider": {"status": llm_status, **llm_details},
            "bm25_index": {"status": bm25_status, **bm25_details},
        },
        "models": {
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "reranker_model": settings.RERANKER_MODEL,
            "reranker_enabled": settings.RERANKER_ENABLED,
        },
    }
