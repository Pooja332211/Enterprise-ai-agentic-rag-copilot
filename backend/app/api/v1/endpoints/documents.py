from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.database.models import Document, DocumentChunk
from backend.app.database.session import get_db
from backend.app.models.schemas import (
    DocumentChunkListResponse,
    DocumentChunkResponse,
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
    IngestionResponse,
)
from backend.app.services.ingestion_service import get_ingestion_service
from backend.app.services.storage_service import get_storage_service
from backend.app.utils.logger import logger

router = APIRouter()


@router.post(
    "",
    summary="Upload and Ingest Document",
    description="Uploads an enterprise document (PDF, DOCX, TXT, MD, HTML), validates, parses, chunks, and persists it.",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format or parameters"},
        409: {"model": IngestionResponse, "description": "Duplicate document already exists"},
        413: {"model": ErrorResponse, "description": "File exceeds size limits"},
    },
)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    force: bool = Query(False, description="Force re-ingestion even if identical hash exists"),
    db: AsyncSession = Depends(get_db),
) -> IngestionResponse:
    ingestion_service = get_ingestion_service()

    filename = file.filename or "uploaded_document"
    content = await file.read()
    mime_type = file.content_type

    try:
        doc, job, is_duplicate = await ingestion_service.ingest_file(
            filename=filename,
            content=content,
            mime_type=mime_type,
            db=db,
            force=force,
        )

        if is_duplicate:
            response.status_code = status.HTTP_409_CONFLICT
            return IngestionResponse(
                document=DocumentResponse.model_validate(doc),
                job=job,
                is_duplicate=True,
                message=f"Duplicate document detected (matches document_id='{doc.id}').",
            )

        return IngestionResponse(
            document=DocumentResponse.model_validate(doc),
            job=job,
            is_duplicate=False,
            message="Document successfully ingested and chunked.",
        )

    except ValueError as val_err:
        err_msg = str(val_err)
        if "exceeds maximum limit" in err_msg:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
    except Exception as exc:
        logger.error(f"Error handling upload for '{filename}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the document.",
        )


@router.get(
    "",
    summary="List Documents",
    description="Retrieves a paginated list of ingested documents, optionally filtered by status.",
    response_model=DocumentListResponse,
)
async def list_documents(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (pending, processing, completed, failed)"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    query = select(Document)
    count_query = select(func.count(Document.id))

    if status_filter:
        query = query.where(Document.status == status_filter.lower())
        count_query = count_query.where(Document.status == status_filter.lower())

    # Get total count
    total_res = await db.execute(count_query)
    total = total_res.scalar_one()

    # Get page items ordered by creation date descending
    query = query.order_by(Document.created_at.desc()).offset(offset).limit(limit)
    items_res = await db.execute(query)
    items = items_res.scalars().all()

    return DocumentListResponse(
        items=[DocumentResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    summary="Get Document",
    description="Retrieves metadata for a specific document by its ID.",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    stmt = select(Document).where(Document.id == document_id)
    res = await db.execute(stmt)
    doc = res.scalars().first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    return DocumentResponse.model_validate(doc)


@router.get(
    "/{document_id}/chunks",
    summary="Get Document Chunks",
    description="Retrieves sequential structure-aware chunks belonging to a document.",
    response_model=DocumentChunkListResponse,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def get_document_chunks(
    document_id: str,
    limit: int = Query(50, ge=1, le=200, description="Chunks per page"),
    offset: int = Query(0, ge=0, description="Chunk offset"),
    db: AsyncSession = Depends(get_db),
) -> DocumentChunkListResponse:
    # Verify document exists
    doc_stmt = select(Document.id).where(Document.id == document_id)
    doc_res = await db.execute(doc_stmt)
    if not doc_res.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    # Get total chunks count
    count_stmt = select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id)
    count_res = await db.execute(count_stmt)
    total = count_res.scalar_one()

    # Get sequential chunks
    query = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .offset(offset)
        .limit(limit)
    )
    chunks_res = await db.execute(query)
    chunks = chunks_res.scalars().all()

    return DocumentChunkListResponse(
        document_id=document_id,
        items=[DocumentChunkResponse.model_validate(c) for c in chunks],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete(
    "/{document_id}",
    summary="Delete Document",
    description="Deletes a document, cascades deletion to all chunks and jobs, and cleans up storage files.",
    status_code=status.HTTP_200_OK,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    stmt = select(Document).where(Document.id == document_id)
    res = await db.execute(stmt)
    doc = res.scalars().first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    # Delete physical file from storage
    storage = get_storage_service()
    if doc.file_path:
        storage.delete_file(doc.file_path)

    # Clean up from vector index and BM25 index
    try:
        from backend.app.services.indexing_service import get_indexing_service
        indexing_service = get_indexing_service()
        indexing_service.remove_document_from_index(document_id)
    except Exception as index_exc:
        logger.warning(f"Failed to remove document '{document_id}' from index: {index_exc}")

    # Invalidate embedding and reranking caches
    try:
        from backend.app.cache.invalidation import invalidate_document_cache
        invalidate_document_cache(document_id)
    except Exception as cache_exc:
        logger.warning(f"Cache invalidation warning for '{document_id}': {cache_exc}")

    # Delete record (foreign keys cascade to chunks and jobs)
    await db.delete(doc)
    await db.commit()

    logger.info(f"document_deleted: document_id={document_id}")
    return {"message": f"Document '{document_id}' and all associated chunks were successfully deleted."}
