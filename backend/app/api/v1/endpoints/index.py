from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.database.session import get_db
from backend.app.models.schemas import (
    ErrorResponse,
    IndexDocumentResponse,
    IndexRebuildResponse,
    IndexStatsResponse,
)
from backend.app.services.indexing_service import get_indexing_service
from backend.app.utils.logger import logger

router = APIRouter()


@router.post(
    "/rebuild",
    summary="Rebuild All Retrieval Indexes",
    description="Drops and reconstructs both Qdrant vector index and BM25 lexical index from relational DocumentChunk records.",
    response_model=IndexRebuildResponse,
    status_code=status.HTTP_200_OK,
    responses={500: {"model": ErrorResponse, "description": "Failed to rebuild indexes"}},
)
async def rebuild_indexes(db: AsyncSession = Depends(get_db)) -> IndexRebuildResponse:
    """Full rebuild of dense and lexical indexes from database."""
    try:
        service = get_indexing_service()
        result = await service.rebuild_all_indexes(db)
        from backend.app.cache.invalidation import invalidate_all_caches
        invalidate_all_caches()
        return IndexRebuildResponse(**result)
    except Exception as exc:
        logger.error(f"Error rebuilding indexes: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rebuild indexes: {str(exc)}",
        )


@router.post(
    "/documents/{document_id}",
    summary="Re-index Document Chunks",
    description="Indexes or re-indexes all chunks for a specific document into Qdrant and BM25.",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Indexing failure"},
    },
)
async def reindex_document_endpoint(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> IndexDocumentResponse:
    """Re-indexes an individual document's chunks."""
    try:
        service = get_indexing_service()
        result = await service.reindex_document(document_id, db)
        from backend.app.cache.invalidation import invalidate_document_cache
        invalidate_document_cache(document_id)
        return IndexDocumentResponse(**result)
    except Exception as exc:
        logger.error(f"Error re-indexing document '{document_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document '{document_id}': {str(exc)}",
        )


@router.delete(
    "/documents/{document_id}",
    summary="Remove Document from Retrieval Indexes",
    description="Removes document chunks from Qdrant vector store and BM25 lexical index without deleting database records.",
    status_code=status.HTTP_200_OK,
    responses={500: {"model": ErrorResponse, "description": "Index purge error"}},
)
async def remove_document_index_endpoint(document_id: str) -> dict:
    """Removes a document's chunks from Qdrant and BM25."""
    try:
        service = get_indexing_service()
        result = service.remove_document_from_index(document_id)
        from backend.app.cache.invalidation import invalidate_document_cache
        invalidate_document_cache(document_id)
        return {
            "message": f"Document '{document_id}' successfully removed from indexes.",
            "details": result,
        }
    except Exception as exc:
        logger.error(f"Error removing document '{document_id}' from index: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove document '{document_id}' from index: {str(exc)}",
        )


@router.get(
    "/stats",
    summary="Get Index Statistics",
    description="Returns metrics and operational status for Qdrant vector database and BM25 lexical index.",
    response_model=IndexStatsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_index_statistics() -> IndexStatsResponse:
    """Retrieves index sizes, points count, and embedding configuration."""
    service = get_indexing_service()
    stats = service.get_index_stats()
    return IndexStatsResponse(**stats)
