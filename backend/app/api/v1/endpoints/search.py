from fastapi import APIRouter, HTTPException, status
from backend.app.models.schemas import ErrorResponse, SearchRequest, SearchResponse
from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service
from backend.app.utils.logger import logger

router = APIRouter()


@router.post(
    "",
    summary="Hybrid Retrieval Search",
    description="Performs dense semantic (Qdrant), sparse lexical (BM25), or hybrid retrieval with Reciprocal Rank Fusion (RRF).",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
        500: {"model": ErrorResponse, "description": "Internal retrieval error"},
    },
)
async def search_chunks(request: SearchRequest) -> SearchResponse:
    """Retrieves ranked relevant chunks using dense, lexical, or hybrid fusion."""
    valid_modes = {"hybrid", "dense", "sparse"}
    mode = request.mode.lower()
    if mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{request.mode}'. Supported modes: {list(valid_modes)}",
        )

    valid_fusion = {"rrf", "weighted"}
    fusion_method = request.fusion_method.lower()
    if fusion_method not in valid_fusion:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid fusion_method '{request.fusion_method}'. Supported: {list(valid_fusion)}",
        )

    try:
        retrieval_service = get_hybrid_retrieval_service()
        result = await retrieval_service.search(
            query=request.query,
            mode=mode,
            top_k=request.top_k,
            document_ids=request.document_ids,
            alpha=request.alpha,
            fusion_method=fusion_method,
            candidate_top_k=request.candidate_top_k,
            rerank=request.rerank,
            min_relevance_threshold=request.min_relevance_threshold,
            context_assembly=request.context_assembly,
            context_max_tokens=request.context_max_tokens,
            context_max_characters=request.context_max_characters,
        )
        return SearchResponse(**result)
    except Exception as exc:
        logger.error(f"Search endpoint failure for query '{request.query}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(exc)}",
        )
