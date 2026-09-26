from fastapi import APIRouter, HTTPException, status
from backend.app.models.schemas import (
    AgentResearchRequest,
    AgentResearchResponse,
    ErrorResponse,
    ResearchRequest,
    ResearchResponse,
)
from backend.app.services.research_service import get_research_service
from backend.app.utils.logger import logger

router = APIRouter()


@router.post(
    "",
    summary="Grounded Agentic Research",
    description=(
        "Executes end-to-end grounded research synthesis: hybrid retrieval, cross-encoder reranking, "
        "context budget assembly, structured LLM generation, and deterministic citation verification."
    ),
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
        500: {"model": ErrorResponse, "description": "Internal research synthesis error"},
    },
)
async def research_query(request: ResearchRequest) -> ResearchResponse:
    """Executes grounded research synthesis and returns verified claims with citations."""
    valid_modes = {"hybrid", "dense", "sparse"}
    mode = request.mode.lower()
    if mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{request.mode}'. Supported modes: {list(valid_modes)}",
        )

    try:
        research_svc = get_research_service()
        response = await research_svc.execute_research(request)
        return response
    except Exception as exc:
        logger.error(f"Research endpoint failure for query '{request.query}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research synthesis error: {str(exc)}",
        )


@router.post(
    "/agent",
    summary="LangGraph Agentic Research Workflow",
    description=(
        "Executes agentic multi-sub-query research workflow using LangGraph state machines: "
        "query analysis, decomposition, sub-query hybrid retrieval, cross-encoder reranking, "
        "evidence aggregation with provenance, sufficiency assessment, iterative refinement loops, "
        "grounded generation, and citation verification."
    ),
    response_model=AgentResearchResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
        500: {"model": ErrorResponse, "description": "Internal agent workflow error"},
    },
)
async def research_agent_query(request: AgentResearchRequest) -> AgentResearchResponse:
    """Executes LangGraph agentic research workflow with query decomposition and iterative retrieval."""
    try:
        from backend.app.agents.agent_service import get_agent_research_service
        agent_svc = get_agent_research_service()
        response = await agent_svc.execute_agent_research(request)
        return response
    except Exception as exc:
        logger.error(f"Agent research endpoint failure for query '{request.query}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent research error: {str(exc)}",
        )
