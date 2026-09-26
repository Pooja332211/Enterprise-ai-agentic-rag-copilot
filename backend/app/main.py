import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.app.api.router import api_router
from backend.app.config.settings import get_settings
from backend.app.database.session import close_db, init_db
from backend.app.observability.context import get_request_id, set_request_id
from backend.app.observability.metrics import metrics_registry
from backend.app.services.embedding_service import get_embedding_service
from backend.app.services.health_service import get_system_health, get_system_readiness
from backend.app.services.reranker_service import get_reranker_service
from backend.app.services.vector_service import get_vector_service
from backend.app.utils.logger import logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager managing startup and shutdown tasks.
    Initializes database schema, verifies vector collection, and warms up neural models.
    """
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")

    # 1. Initialize database schema
    try:
        await init_db()
        logger.info("Database schema verification completed.")
    except Exception as exc:
        logger.error(f"Database initialization warning: {exc}")

    # 2. Ensure Qdrant collection is ready
    try:
        vector_service = get_vector_service()
        if vector_service.is_connected():
            vector_service.ensure_collection()
            logger.info(f"Vector collection '{settings.QDRANT_COLLECTION_NAME}' verified.")

        # Ensure indexes are synchronized with relational DB chunks on startup
        try:
            from sqlalchemy import select, func
            from backend.app.database.session import get_session_factory
            from backend.app.database.models import DocumentChunk
            from backend.app.services.bm25_service import get_bm25_service
            from backend.app.services.indexing_service import get_indexing_service

            bm25_svc = get_bm25_service()
            session_factory = get_session_factory()
            async with session_factory() as session:
                chunk_count_stmt = select(func.count()).select_from(DocumentChunk)
                chunk_res = await session.execute(chunk_count_stmt)
                db_chunk_count = chunk_res.scalar() or 0

                if db_chunk_count > 0:
                    vec_count = 0
                    try:
                        stats = vector_service.get_collection_stats()
                        vec_count = stats.get("points_count", 0)
                    except Exception:
                        vec_count = 0
                    bm25_count = len(bm25_svc._chunks)
                    if vec_count == 0 or bm25_count == 0:
                        indexing_svc = get_indexing_service()
                        logger.info(
                            f"Synchronizing indexes on startup (DB chunks: {db_chunk_count}, "
                            f"BM25: {bm25_count}, Vector: {vec_count})..."
                        )
                        await indexing_svc.rebuild_all_indexes(session)
                        logger.info("Startup index synchronization completed successfully.")
        except Exception as sync_exc:
            logger.warning(f"Startup index synchronization check warning: {sync_exc}")
    except Exception as exc:
        logger.warning(f"Vector service initialization warning: {exc}")

    # 3. Model warmup (non-fatal)
    try:
        emb_service = get_embedding_service()
        emb_service.embed_query("startup health check")
        logger.info("Embedding model warmup completed.")
    except Exception as exc:
        logger.warning(f"Embedding warmup warning (non-fatal): {exc}")

    try:
        reranker_service = get_reranker_service()
        reranker_service.warmup()
    except Exception as exc:
        logger.warning(f"Reranker warmup warning (non-fatal): {exc}")

    yield

    # Clean shutdown
    logger.info("Initiating graceful shutdown...")
    try:
        await close_db()
    except Exception as exc:
        logger.error(f"Error closing database pool: {exc}")

    try:
        vector_service = get_vector_service()
        vector_service.close()
    except Exception as exc:
        logger.error(f"Error closing vector client: {exc}")

    logger.info("Shutdown sequence complete.")


def create_application() -> FastAPI:
    """Application factory creating and configuring the FastAPI instance."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Request Correlation ID, Timing, and Metrics
    @app.middleware("http")
    async def request_observability_middleware(request: Request, call_next: Any) -> Response:
        incoming_req_id = request.headers.get("X-Request-ID")
        req_id = set_request_id(incoming_req_id)
        start_time = time.perf_counter()
        status_code = 500
        response = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            metrics_registry.request_errors.inc(
                labels={"endpoint": request.url.path, "error_type": type(exc).__name__}
            )
            raise
        finally:
            process_time = time.perf_counter() - start_time
            metrics_registry.requests_total.inc(
                labels={"endpoint": request.url.path, "method": request.method, "status": str(status_code)}
            )
            metrics_registry.request_duration.observe(
                process_time,
                labels={"endpoint": request.url.path},
            )
            if response is not None:
                response.headers["X-Request-ID"] = req_id
                response.headers["X-Process-Time"] = f"{process_time:.4f}"

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = get_request_id()
        logger.error(
            f"[{req_id}] Unhandled error processing {request.method} {request.url.path}: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please check system diagnostics.",
                "request_id": req_id,
            },
            headers={"X-Request-ID": req_id},
        )

    # Root Level Health Check (Liveness)
    @app.get(
        "/health",
        tags=["Health & Diagnostics"],
        summary="Root Liveness Probe",
        description="Fast HTTP 200 probe for load balancers and orchestrators.",
    )
    async def root_health() -> Dict[str, Any]:
        return await get_system_health()

    # Root Level Readiness Check
    @app.get(
        "/health/ready",
        tags=["Health & Diagnostics"],
        summary="Root Readiness Probe",
        description="Composite health check verifying database, vector store, embedding provider, reranker, LLM, and BM25 index.",
    )
    async def root_readiness(response: Response) -> Dict[str, Any]:
        readiness = await get_system_readiness()
        if readiness["status"] == "unhealthy":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return readiness

    # Prometheus Metrics Exposition Endpoint
    @app.get(
        "/metrics",
        tags=["Health & Diagnostics"],
        summary="Prometheus Metrics Exposition",
        description="Native dependency-free Prometheus text format exposition endpoint.",
    )
    async def get_metrics() -> Response:
        exposition = metrics_registry.generate_exposition()
        return Response(content=exposition, media_type="text/plain; version=0.0.4; charset=utf-8")

    # Mount API routers
    app.include_router(api_router)

    return app


app = create_application()
