"""LangGraph state machine for agentic multi-subquery research workflows."""

import asyncio
import time
from typing import Any, Dict, List, Optional
from langgraph.graph import END, START, StateGraph

from backend.app.agents.evidence_aggregator import EvidenceAggregator
from backend.app.agents.query_analyzer import QueryAnalyzer
from backend.app.agents.query_decomposer import QueryDecomposer
from backend.app.agents.state import (
    AgentEvidenceItem,
    AgentState,
    ExecutionTraceItem,
    QueryClassification,
    SubQueryPlan,
    SufficiencyAssessment,
)
from backend.app.agents.sufficiency import SufficiencyEvaluator
from backend.app.config.settings import Settings, get_settings
from backend.app.models.schemas import ClaimItem, VerificationSummary
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
from backend.app.services.reranker_service import BaseReranker, get_reranker_service
from backend.app.observability import metrics_registry, redact_sensitive_data
from backend.app.utils.logger import logger


class ResearchAgentGraph:
    """Builds and compiles the LangGraph state machine for agentic enterprise research."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retrieval_service: Optional[HybridRetrievalService] = None,
        reranker_service: Optional[BaseReranker] = None,
        generation_service: Optional[GroundedGenerationService] = None,
        verification_service: Optional[CitationVerificationService] = None,
        llm_service: Optional[BaseLLMService] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        query_decomposer: Optional[QueryDecomposer] = None,
        evidence_aggregator: Optional[EvidenceAggregator] = None,
        sufficiency_evaluator: Optional[SufficiencyEvaluator] = None,
    ):
        self.settings = settings or get_settings()
        self.llm_service = llm_service or get_llm_service(self.settings)
        self.retrieval_service = retrieval_service or get_hybrid_retrieval_service(self.settings)
        self.reranker_service = reranker_service or get_reranker_service(self.settings)
        self.generation_service = generation_service or get_grounded_generation_service(
            llm_service=self.llm_service, settings=self.settings
        )
        self.verification_service = verification_service or get_citation_verification_service(self.settings)

        self.query_analyzer = query_analyzer or QueryAnalyzer(llm_service=self.llm_service)
        self.query_decomposer = query_decomposer or QueryDecomposer(llm_service=self.llm_service)
        self.evidence_aggregator = evidence_aggregator or EvidenceAggregator(
            context_max_tokens=getattr(self.settings, "CONTEXT_MAX_TOKENS", 2000)
        )
        self.sufficiency_evaluator = sufficiency_evaluator or SufficiencyEvaluator(
            llm_service=self.llm_service,
            min_relevance_threshold=getattr(self.settings, "MIN_RELEVANCE_THRESHOLD", None) or 0.0005,
        )

        self.graph = self._build_graph()

    def _record_trace(
        self,
        state: AgentState,
        node: str,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
    ) -> None:
        """Appends structured audit item to execution trace with secret redaction."""
        trace_list = state.setdefault("execution_trace", [])
        trace_list.append(
            ExecutionTraceItem(
                node=node,
                iteration=state.get("iteration_count", 0),
                input_summary=str(redact_sensitive_data(input_summary[:200])),
                output_summary=str(redact_sensitive_data(output_summary[:200])),
                duration_ms=round(duration_ms, 2),
            ).model_dump()
        )

    # --------------------------------------------------------------------------
    # GRAPH NODES
    # --------------------------------------------------------------------------

    async def node_analyze_query(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Analyzes user query and classifies complexity."""
        t0 = time.perf_counter()
        q = state["original_query"]

        classification = await self.query_analyzer.analyze(q)
        duration = (time.perf_counter() - t0) * 1000

        timings = state.setdefault("timings_ms", {})
        timings["query_analysis_ms"] = round(duration, 2)

        try:
            metrics_registry.agent_iterations.inc(labels={"query_type": classification.query_type})
        except Exception as exc:
            logger.debug(f"Metrics record error in node_analyze_query: {exc}")

        self._record_trace(
            state,
            node="analyze_query",
            input_summary=f"Query: {q}",
            output_summary=f"Type={classification.query_type}, Decompose={classification.needs_decomposition}",
            duration_ms=duration,
        )

        return {
            "query_type": classification.query_type,
            "complexity": classification.complexity,
            "needs_decomposition": classification.needs_decomposition,
            "status": "planned",
        }

    async def node_plan_query(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: Generates sub-queries if decomposition is warranted."""
        t0 = time.perf_counter()
        q = state["original_query"]

        classification = QueryClassification(
            query_type=state.get("query_type", "simple_factual"),
            complexity=state.get("complexity", "low"),
            needs_decomposition=state.get("needs_decomposition", False) if state.get("enable_decomposition", True) else False,
        )

        sub_queries = await self.query_decomposer.decompose(q, classification)
        duration = (time.perf_counter() - t0) * 1000

        timings = state.setdefault("timings_ms", {})
        timings["query_planning_ms"] = round(duration, 2)

        sq_dicts = [sq.model_dump() for sq in sub_queries]
        self._record_trace(
            state,
            node="plan_query",
            input_summary=f"Query: {q}, NeedsDecomposition={classification.needs_decomposition}",
            output_summary=f"Generated {len(sub_queries)} sub-queries: {[s.sub_query_id for s in sub_queries]}",
            duration_ms=duration,
        )

        return {
            "sub_queries": sq_dicts,
            "completed_sub_queries": [],
            "retrieval_results": {},
            "iteration_count": 0,
            "status": "retrieving",
        }

    async def node_retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: Executes retrieval for each sub-query via existing hybrid retrieval."""
        t0 = time.perf_counter()
        sub_queries = state.get("sub_queries", [])
        completed = set(state.get("completed_sub_queries", []))
        results_map = state.get("retrieval_results", {}).copy()

        top_k = state.get("top_k", 5)
        cand_k = state.get("candidate_top_k", 20)

        newly_completed = []
        for sq in sub_queries:
            sq_id = sq["sub_query_id"]
            if sq_id in completed:
                continue

            query_text = sq["sub_query_text"]
            logger.info(f"agent_retrieval: executing sub_query '{sq_id}': {query_text[:50]}")

            search_res = await self.retrieval_service.search(
                query=query_text,
                mode="hybrid",
                top_k=cand_k,
                candidate_top_k=cand_k,
                rerank=False,
                context_assembly=False,
            )

            raw_chunks = search_res.get("results", [])
            results_map[sq_id] = raw_chunks
            newly_completed.append(sq_id)

        duration = (time.perf_counter() - t0) * 1000
        timings = state.setdefault("timings_ms", {})
        timings["retrieval_ms"] = round(timings.get("retrieval_ms", 0.0) + duration, 2)

        all_completed = list(completed.union(newly_completed))
        for _ in newly_completed:
            try:
                metrics_registry.retrieval_calls.inc(labels={"source": "agent_subquery"})
            except Exception as exc:
                logger.debug(f"Metrics record error in node_retrieve: {exc}")

        self._record_trace(
            state,
            node="retrieve",
            input_summary=f"Sub-queries to retrieve: {newly_completed}",
            output_summary=f"Retrieved {sum(len(v) for v in results_map.values())} total candidate chunks",
            duration_ms=duration,
        )

        return {
            "retrieval_results": results_map,
            "completed_sub_queries": all_completed,
            "status": "reranking",
        }

    async def node_rerank(self, state: AgentState) -> Dict[str, Any]:
        """Node 4: Cross-Encoder neural reranking per sub-query."""
        t0 = time.perf_counter()
        results_map = state.get("retrieval_results", {}).copy()
        sub_queries = {sq["sub_query_id"]: sq["sub_query_text"] for sq in state.get("sub_queries", [])}
        top_k = state.get("top_k", 5)
        should_rerank = state.get("rerank", True)

        reranked_map = {}
        if should_rerank:
            for sq_id, chunks in results_map.items():
                if not chunks:
                    reranked_map[sq_id] = []
                    continue
                q_text = sub_queries.get(sq_id, state["original_query"])
                rerank_val = self.reranker_service.rerank(
                    query=q_text,
                    candidates=chunks,
                    top_k=top_k,
                )
                if asyncio.iscoroutine(rerank_val):
                    reranked = await rerank_val
                else:
                    reranked = rerank_val
                reranked_map[sq_id] = reranked
        else:
            for sq_id, chunks in results_map.items():
                reranked_map[sq_id] = chunks[:top_k]

        duration = (time.perf_counter() - t0) * 1000
        timings = state.setdefault("timings_ms", {})
        timings["reranking_ms"] = round(timings.get("reranking_ms", 0.0) + duration, 2)

        self._record_trace(
            state,
            node="rerank",
            input_summary=f"Reranking candidate pools across {len(results_map)} sub-queries",
            output_summary=f"Reranked top {top_k} per sub-query",
            duration_ms=duration,
        )

        return {
            "retrieval_results": reranked_map,
            "status": "aggregating",
        }

    async def node_assemble_evidence(self, state: AgentState) -> Dict[str, Any]:
        """Node 5: Deduplicates and packs evidence with provenance tracking."""
        t0 = time.perf_counter()
        results_map = state.get("retrieval_results", {})
        q = state["original_query"]

        aggregated = self.evidence_aggregator.aggregate_evidence(results_map, parent_query=q)
        duration = (time.perf_counter() - t0) * 1000

        timings = state.setdefault("timings_ms", {})
        timings["evidence_assembly_ms"] = round(timings.get("evidence_assembly_ms", 0.0) + duration, 2)

        agg_dicts = [item.model_dump() for item in aggregated]
        self._record_trace(
            state,
            node="assemble_evidence",
            input_summary=f"Evidence pools from {len(results_map)} sub-queries",
            output_summary=f"Deduplicated to {len(aggregated)} unique evidence chunks",
            duration_ms=duration,
        )

        return {
            "aggregated_evidence": agg_dicts,
            "status": "assessing",
        }

    async def node_assess_sufficiency(self, state: AgentState) -> Dict[str, Any]:
        """Node 6: Assesses if evidence covers query requirements."""
        t0 = time.perf_counter()
        q = state["original_query"]
        classification = QueryClassification(
            query_type=state.get("query_type", "simple_factual"),
            complexity=state.get("complexity", "low"),
            needs_decomposition=state.get("needs_decomposition", False),
        )
        sub_query_plans = [SubQueryPlan(**sq) for sq in state.get("sub_queries", [])]
        aggregated = [AgentEvidenceItem(**item) for item in state.get("aggregated_evidence", [])]
        retrieval_results = state.get("retrieval_results", {})

        assessment = await self.sufficiency_evaluator.evaluate(
            query=q,
            classification=classification,
            sub_queries=sub_query_plans,
            retrieval_results=retrieval_results,
            aggregated_evidence=aggregated,
        )

        duration = (time.perf_counter() - t0) * 1000
        timings = state.setdefault("timings_ms", {})
        timings["sufficiency_check_ms"] = round(timings.get("sufficiency_check_ms", 0.0) + duration, 2)

        new_count = state.get("iteration_count", 0) + 1

        self._record_trace(
            state,
            node="assess_sufficiency",
            input_summary=f"Assessing evidence for {len(sub_query_plans)} sub-queries (Iteration {new_count})",
            output_summary=f"Sufficient={assessment.sufficient} (Confidence={assessment.confidence:.2f}): {assessment.reason}",
            duration_ms=duration,
        )

        return {
            "evidence_sufficient": assessment.sufficient,
            "sufficiency_reason": assessment.reason,
            "missing_information": assessment.missing_information,
            "iteration_count": new_count,
            "status": "assessed",
        }

    async def node_refine_queries(self, state: AgentState) -> Dict[str, Any]:
        """Node 7: Refines queries targeting missing information for iterative retrieval."""
        t0 = time.perf_counter()
        missing = state.get("missing_information", [])
        q = state["original_query"]
        sub_queries = state.get("sub_queries", []).copy()

        new_plans = []
        for idx, item in enumerate(missing, start=len(sub_queries) + 1):
            sq_id = f"SQ-REF-{idx}"
            refined_text = f"{item} details architecture"
            new_plans.append(
                SubQueryPlan(
                    sub_query_id=sq_id,
                    sub_query_text=refined_text,
                    purpose=f"Iterative retrieval for missing information: {item[:30]}",
                    parent_query=q,
                    priority=2,
                ).model_dump()
            )

        sub_queries.extend(new_plans)
        duration = (time.perf_counter() - t0) * 1000

        self._record_trace(
            state,
            node="refine_queries",
            input_summary=f"Missing information: {missing}",
            output_summary=f"Generated {len(new_plans)} refined sub-queries for iteration {state.get('iteration_count')}",
            duration_ms=duration,
        )

        return {
            "sub_queries": sub_queries,
            "status": "retrieving",
        }

    async def node_generate(self, state: AgentState) -> Dict[str, Any]:
        """Node 8: Grounded answer generation strictly using aggregated evidence."""
        t0 = time.perf_counter()
        q = state["original_query"]
        aggregated = [AgentEvidenceItem(**item) for item in state.get("aggregated_evidence", [])]
        context_text = self.evidence_aggregator.format_context_block(aggregated)

        # Convert to format expected by GroundedGenerationService
        evidence_dicts = [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "filename": item.filename,
                "page_number": item.page_number,
                "section_title": item.section_title,
                "text": item.text,
                "score": item.rerank_score or item.retrieval_score,
            }
            for item in aggregated
        ]

        gen_output, gen_ms = await self.generation_service.generate_grounded_answer(
            query=q,
            context_text=context_text,
            evidence_chunks=evidence_dicts,
            temperature=state.get("temperature", 0.0),
            max_tokens=state.get("max_tokens", 2048),
        )

        duration = (time.perf_counter() - t0) * 1000
        timings = state.setdefault("timings_ms", {})
        timings["generation_ms"] = round(duration, 2)

        claims_list = [c.model_dump() for c in gen_output.claims]
        self._record_trace(
            state,
            node="generate",
            input_summary=f"Evidence text ({len(context_text)} chars, {len(aggregated)} chunks)",
            output_summary=f"Generated answer ({len(gen_output.answer)} chars, {len(claims_list)} claims)",
            duration_ms=duration,
        )

        return {
            "final_answer": gen_output.answer,
            "claims": claims_list,
            "generation_result": gen_output.model_dump(),
            "status": "verifying",
        }

    async def node_abstain(self, state: AgentState) -> Dict[str, Any]:
        """Node 9: Formulates clear non-hallucinated refusal when evidence is insufficient."""
        t0 = time.perf_counter()
        q = state["original_query"]
        reason = state.get("sufficiency_reason", "The available documents do not contain sufficient evidence.")

        answer = "I cannot answer this question based on the provided documents. The documents do not contain sufficient evidence."
        duration = (time.perf_counter() - t0) * 1000

        try:
            metrics_registry.abstentions.inc(labels={"reason": "insufficient_evidence"})
        except Exception as exc:
            logger.debug(f"Metrics record error in node_abstain: {exc}")

        self._record_trace(
            state,
            node="abstain",
            input_summary=f"Query: {q}, Reason: {reason}",
            output_summary="Emitted standard non-hallucinated refusal",
            duration_ms=duration,
        )

        return {
            "final_answer": answer,
            "claims": [],
            "generation_result": {"insufficient_evidence": True},
            "status": "abstained",
        }

    async def node_verify_citations(self, state: AgentState) -> Dict[str, Any]:
        """Node 10: Citation audit and claim provenance verification."""
        t0 = time.perf_counter()
        claims_data = state.get("claims", [])
        claims = [ClaimItem(**c) for c in claims_data]
        aggregated = [AgentEvidenceItem(**item) for item in state.get("aggregated_evidence", [])]

        evidence_dicts = [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "filename": item.filename,
                "page_number": item.page_number,
                "section_title": item.section_title,
                "text": item.text,
            }
            for item in aggregated
        ]

        verification_summary, citations, verify_ms = self.verification_service.verify_claims(
            claims=claims,
            evidence_chunks=evidence_dicts,
        )

        duration = (time.perf_counter() - t0) * 1000
        timings = state.setdefault("timings_ms", {})
        timings["citation_verification_ms"] = round(duration, 2)

        try:
            metrics_registry.citation_verification_duration.observe(
                duration / 1000.0,
                labels={"status": verification_summary.status},
            )
            if verification_summary.status not in ("verified", "partially_verified"):
                metrics_registry.citation_failures.inc(labels={"reason": verification_summary.status})
            if verification_summary.unsupported_claims > 0:
                metrics_registry.unsupported_claims.inc(amount=float(verification_summary.unsupported_claims))
        except Exception as exc:
            logger.debug(f"Metrics record error in node_verify_citations: {exc}")

        citations_list = [cit.model_dump() for cit in citations]
        self._record_trace(
            state,
            node="verify_citations",
            input_summary=f"{len(claims)} claims against {len(evidence_dicts)} evidence chunks",
            output_summary=f"Status: {verification_summary.status}, Citations: {len(citations_list)}",
            duration_ms=duration,
        )

        prior_status = state.get("status")
        final_status = "abstained" if prior_status == "abstained" else "completed"

        return {
            "citation_audit": verification_summary.model_dump(),
            "citations": citations_list,
            "status": final_status,
        }

    # --------------------------------------------------------------------------
    # CONDITIONAL EDGES
    # --------------------------------------------------------------------------

    def route_after_assessment(self, state: AgentState) -> str:
        """Explicit conditional routing based on evidence sufficiency and loop bounds."""
        if state.get("evidence_sufficient", False):
            return "generate"

        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 2)

        if iteration < max_iter:
            logger.info(f"agent_routing: iterative retrieval triggered (iteration {iteration} < {max_iter})")
            return "refine_queries"

        logger.info(f"agent_routing: iteration bound reached ({iteration} >= {max_iter}); abstaining")
        return "abstain"

    def _build_graph(self) -> Any:
        """Constructs and compiles the StateGraph workflow."""
        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("analyze_query", self.node_analyze_query)
        builder.add_node("plan_query", self.node_plan_query)
        builder.add_node("retrieve", self.node_retrieve)
        builder.add_node("rerank", self.node_rerank)
        builder.add_node("assemble_evidence", self.node_assemble_evidence)
        builder.add_node("assess_sufficiency", self.node_assess_sufficiency)
        builder.add_node("refine_queries", self.node_refine_queries)
        builder.add_node("generate", self.node_generate)
        builder.add_node("abstain", self.node_abstain)
        builder.add_node("verify_citations", self.node_verify_citations)

        # Linear edges
        builder.add_edge(START, "analyze_query")
        builder.add_edge("analyze_query", "plan_query")
        builder.add_edge("plan_query", "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "assemble_evidence")
        builder.add_edge("assemble_evidence", "assess_sufficiency")

        # Conditional branching
        builder.add_conditional_edges(
            "assess_sufficiency",
            self.route_after_assessment,
            {
                "generate": "generate",
                "refine_queries": "refine_queries",
                "abstain": "abstain",
            },
        )

        # Loop edge
        builder.add_edge("refine_queries", "retrieve")

        # Terminal edges
        builder.add_edge("generate", "verify_citations")
        builder.add_edge("abstain", "verify_citations")
        builder.add_edge("verify_citations", END)

        return builder.compile()

    async def execute(self, state_input: Dict[str, Any]) -> AgentState:
        """Executes the compiled LangGraph agentic research workflow."""
        t_start = time.perf_counter()

        initial_state: AgentState = {
            "original_query": state_input["query"],
            "normalized_query": state_input["query"].strip(),
            "top_k": state_input.get("top_k", 5),
            "candidate_top_k": state_input.get("candidate_top_k", 20),
            "rerank": state_input.get("rerank", True),
            "max_iterations": state_input.get("max_iterations", 2),
            "enable_decomposition": state_input.get("enable_decomposition", True),
            "context_max_tokens": state_input.get("context_max_tokens", 1500),
            "temperature": state_input.get("temperature", 0.0),
            "max_tokens": state_input.get("max_tokens", 2048),
            "execution_trace": [],
            "timings_ms": {},
            "errors": [],
            "status": "started",
        }

        final_state = await self.graph.ainvoke(initial_state)

        total_elapsed = (time.perf_counter() - t_start) * 1000
        timings = final_state.setdefault("timings_ms", {})
        timings["total_pipeline_ms"] = round(total_elapsed, 2)

        return final_state
