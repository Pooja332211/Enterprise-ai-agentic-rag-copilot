import time
from typing import Any, Dict, List, Optional
from backend.app.config.settings import Settings, get_settings
from backend.app.models.schemas import (
    ResearchRequest,
    ResearchResponse,
    SearchResultItem,
    VerificationSummary,
)
from backend.app.services.citation_verification_service import (
    CitationVerificationService,
    get_citation_verification_service,
)
from backend.app.services.grounded_generation_service import (
    GroundedGenerationService,
    get_grounded_generation_service,
)
from backend.app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
    get_hybrid_retrieval_service,
)
from backend.app.services.llm_service import BaseLLMService, get_llm_service
from backend.app.utils.logger import logger


class ResearchService:
    """
    Core Agentic RAG orchestrator coordinating the complete research pipeline:
    1. First-stage hybrid candidate retrieval (Dense + BM25 + RRF)
    2. Cross-encoder neural reranking (BGE Reranker)
    3. Context assembly (deduplication & token budget packing)
    4. Grounded answer generation via LLM
    5. Citation mapping and claim verification audit
    """

    def __init__(
        self,
        retrieval_service: Optional[HybridRetrievalService] = None,
        generation_service: Optional[GroundedGenerationService] = None,
        verification_service: Optional[CitationVerificationService] = None,
        llm_service: Optional[BaseLLMService] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or get_hybrid_retrieval_service(self.settings)
        self.generation_service = generation_service or get_grounded_generation_service(
            llm_service=llm_service,
            settings=self.settings,
        )
        self.verification_service = verification_service or get_citation_verification_service(self.settings)
        self.llm_service = llm_service or get_llm_service(self.settings)

    async def execute_research(self, request: ResearchRequest) -> ResearchResponse:
        """
        Executes end-to-end grounded research synthesis for a query.
        """
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}

        # ----------------------------------------------------------------------
        # Stage 1: Retrieval, Reranking & Context Assembly
        # ----------------------------------------------------------------------
        search_res = await self.retrieval_service.search(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            candidate_top_k=request.candidate_top_k,
            rerank=request.rerank,
            min_relevance_threshold=request.min_relevance_threshold,
            document_ids=request.document_ids,
            context_assembly=True,
            context_max_tokens=request.context_max_tokens,
        )

        for k, v in search_res.get("breakdown_ms", {}).items():
            timings[k] = v

        results = search_res.get("results", [])
        evidence_items = [SearchResultItem(**item) for item in results]

        # ----------------------------------------------------------------------
        # Safeguard: Empty Evidence Context Handling
        # ----------------------------------------------------------------------
        if not results:
            total_elapsed = round((time.perf_counter() - t_start) * 1000, 2)
            timings["total_latency_ms"] = total_elapsed

            empty_summary = VerificationSummary(
                status="unverified",
                supported_claims=0,
                partially_supported_claims=0,
                unsupported_claims=0,
                total_claims=0,
                details=[],
            )

            return ResearchResponse(
                query=request.query,
                answer="The available documents do not contain sufficient evidence to answer this query.",
                claims=[],
                citations=[],
                verification=empty_summary,
                evidence=[],
                timings_ms=timings,
                model=self.llm_service.model_name,
                provider=self.llm_service.provider_name,
                grounded=False,
                insufficient_evidence=True,
            )

        # ----------------------------------------------------------------------
        # Stage 2: Grounded Answer & Claim Generation
        # ----------------------------------------------------------------------
        assembled = search_res.get("assembled_context") or {}
        context_text = assembled.get("context_text") or "\n\n".join([r.get("text", "") for r in results])

        gen_output, gen_ms = await self.generation_service.generate_grounded_answer(
            query=request.query,
            context_text=context_text,
            evidence_chunks=results,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        timings["generation_ms"] = gen_ms

        # ----------------------------------------------------------------------
        # Stage 3: Citation Verification & Provenance Mapping
        # ----------------------------------------------------------------------
        verification_summary, citations, verify_ms = self.verification_service.verify_claims(
            claims=gen_output.claims,
            evidence_chunks=results,
        )
        timings["verification_ms"] = verify_ms

        if gen_output.insufficient_evidence:
            citations = []

        total_elapsed = round((time.perf_counter() - t_start) * 1000, 2)
        timings["total_latency_ms"] = total_elapsed

        is_grounded = (
            verification_summary.status in ("verified", "partially_verified")
            and not gen_output.insufficient_evidence
        )

        logger.info(
            f"research_pipeline_completed: query='{request.query[:40]}' grounded={is_grounded} "
            f"status='{verification_summary.status}' claims={len(gen_output.claims)} "
            f"citations={len(citations)} total_ms={total_elapsed}"
        )

        return ResearchResponse(
            query=request.query,
            answer=gen_output.answer,
            claims=gen_output.claims,
            citations=citations,
            verification=verification_summary,
            evidence=evidence_items,
            timings_ms=timings,
            model=self.llm_service.model_name,
            provider=self.llm_service.provider_name,
            grounded=is_grounded,
            insufficient_evidence=gen_output.insufficient_evidence,
        )


_research_service_instance: Optional[ResearchService] = None


def get_research_service(
    settings: Optional[Settings] = None,
    llm_service: Optional[BaseLLMService] = None,
) -> ResearchService:
    """Returns singleton ResearchService instance."""
    global _research_service_instance
    if _research_service_instance is None:
        _research_service_instance = ResearchService(
            settings=settings,
            llm_service=llm_service,
        )
    return _research_service_instance


def reset_research_service() -> None:
    """Resets singleton instance for testing."""
    global _research_service_instance
    _research_service_instance = None
