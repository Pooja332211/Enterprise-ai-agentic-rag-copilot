"""Agent service coordinator orchestrating the LangGraph research workflow."""

from typing import Optional
from backend.app.agents.research_graph import ResearchAgentGraph
from backend.app.config.settings import Settings, get_settings
from backend.app.models.schemas import (
    AgentResearchRequest,
    AgentResearchResponse,
    CitationItem,
    ClaimItem,
    VerificationSummary,
)
from backend.app.services.citation_verification_service import CitationVerificationService
from backend.app.services.grounded_generation_service import GroundedGenerationService
from backend.app.services.hybrid_retrieval_service import HybridRetrievalService
from backend.app.services.llm_service import BaseLLMService, get_llm_service
from backend.app.services.reranker_service import BaseReranker
from backend.app.utils.logger import logger


class AgentResearchService:
    """High-level service interface for LangGraph agentic research queries."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retrieval_service: Optional[HybridRetrievalService] = None,
        reranker_service: Optional[BaseReranker] = None,
        generation_service: Optional[GroundedGenerationService] = None,
        verification_service: Optional[CitationVerificationService] = None,
        llm_service: Optional[BaseLLMService] = None,
    ):
        self.settings = settings or get_settings()
        self.llm_service = llm_service or get_llm_service(self.settings)
        self.graph = ResearchAgentGraph(
            settings=self.settings,
            retrieval_service=retrieval_service,
            reranker_service=reranker_service,
            generation_service=generation_service,
            verification_service=verification_service,
            llm_service=self.llm_service,
        )

    async def execute_agent_research(self, request: AgentResearchRequest) -> AgentResearchResponse:
        """Executes the full LangGraph state machine research workflow."""
        logger.info(f"agent_research_service: starting query='{request.query[:50]}'")

        state_input = {
            "query": request.query,
            "top_k": request.top_k,
            "candidate_top_k": request.candidate_top_k,
            "rerank": request.rerank,
            "max_iterations": request.max_iterations,
            "enable_decomposition": request.enable_decomposition,
            "context_max_tokens": request.context_max_tokens,
            "temperature": request.temperature or 0.0,
            "max_tokens": request.max_tokens or 2048,
        }

        final_state = await self.graph.execute(state_input)

        # Parse claims, citations, verification
        raw_claims = final_state.get("claims", [])
        claims = [ClaimItem(**c) for c in raw_claims]

        raw_citations = final_state.get("citations", [])
        citations = [CitationItem(**cit) for cit in raw_citations]

        raw_verification = final_state.get("citation_audit")
        if raw_verification:
            verification = VerificationSummary(**raw_verification)
        else:
            verification = VerificationSummary(
                status="unverified",
                supported_claims=0,
                partially_supported_claims=0,
                unsupported_claims=0,
                total_claims=0,
                details=[],
            )

        gen_res = final_state.get("generation_result") or {}
        is_insufficient = gen_res.get("insufficient_evidence", False) or final_state.get("status") == "abstained"
        is_grounded = verification.status in ("verified", "partially_verified") and not is_insufficient
        if is_insufficient:
            citations = []

        # Normalize sub-queries for frontend compatibility
        normalized_sub_queries: List[Dict[str, Any]] = []
        for idx, sq in enumerate(final_state.get("sub_queries", []), start=1):
            if isinstance(sq, dict):
                sq_id = sq.get("sub_query_id") or sq.get("id") or f"SQ-{idx}"
                sq_text = sq.get("sub_query_text") or sq.get("query") or ""
                sq_purpose = sq.get("purpose") or sq.get("reasoning") or ""
                sq_status = sq.get("status") or "completed"
                normalized_sub_queries.append({
                    **sq,
                    "id": sq_id,
                    "sub_query_id": sq_id,
                    "query": sq_text,
                    "sub_query_text": sq_text,
                    "reasoning": sq_purpose,
                    "purpose": sq_purpose,
                    "status": sq_status,
                })
            else:
                normalized_sub_queries.append(sq)

        # Normalize execution trace spans for frontend compatibility
        normalized_trace: List[Dict[str, Any]] = []
        for t in final_state.get("execution_trace", []):
            if isinstance(t, dict):
                node_name = t.get("node") or t.get("step") or t.get("stage") or "node"
                dur = t.get("duration_ms") if t.get("duration_ms") is not None else t.get("latency_ms", 0.0)
                normalized_trace.append({
                    **t,
                    "node": node_name,
                    "step": node_name,
                    "stage": node_name,
                    "duration_ms": round(float(dur), 2),
                    "latency_ms": round(float(dur), 2),
                })
            else:
                normalized_trace.append(t)

        timings = final_state.get("timings_ms", {})
        total_duration = (
            timings.get("total_pipeline_ms")
            or timings.get("total_latency_ms")
            or sum(t.get("duration_ms", 0.0) for t in normalized_trace)
            or 0.0
        )

        return AgentResearchResponse(
            query=request.query,
            final_answer=final_state.get("final_answer", ""),
            query_type=final_state.get("query_type", "simple_factual"),
            complexity=final_state.get("complexity", "low"),
            sub_queries=normalized_sub_queries,
            evidence=final_state.get("aggregated_evidence", []),
            claims=claims,
            citations=citations,
            verification=verification,
            iterations_used=final_state.get("iteration_count", 1),
            execution_trace=normalized_trace,
            timings_ms=timings,
            total_duration_ms=round(float(total_duration), 2),
            provider=self.llm_service.provider_name,
            model=self.llm_service.model_name,
            grounded=is_grounded,
            insufficient_evidence=is_insufficient,
            status=final_state.get("status", "completed"),
        )


_agent_research_service_instance: Optional[AgentResearchService] = None


def get_agent_research_service(
    settings: Optional[Settings] = None,
    llm_service: Optional[BaseLLMService] = None,
) -> AgentResearchService:
    """Returns singleton AgentResearchService instance."""
    global _agent_research_service_instance
    if _agent_research_service_instance is None:
        _agent_research_service_instance = AgentResearchService(
            settings=settings,
            llm_service=llm_service,
        )
    return _agent_research_service_instance


def reset_agent_research_service() -> None:
    """Resets singleton instance for testing."""
    global _agent_research_service_instance
    _agent_research_service_instance = None
