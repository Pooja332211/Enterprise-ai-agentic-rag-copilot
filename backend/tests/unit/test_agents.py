"""Unit tests for LangGraph agentic research workflow components."""

import pytest
from typing import Any, Dict, List

from backend.app.agents.evidence_aggregator import EvidenceAggregator
from backend.app.agents.query_analyzer import QueryAnalyzer
from backend.app.agents.query_decomposer import QueryDecomposer
from backend.app.agents.research_graph import ResearchAgentGraph
from backend.app.agents.state import (
    AgentEvidenceItem,
    QueryClassification,
    SubQueryPlan,
    SufficiencyAssessment,
)
from backend.app.agents.sufficiency import SufficiencyEvaluator
from backend.app.config.settings import Settings
from backend.app.services.llm_service import MockLLMService
from backend.app.services.reranker_service import MockRerankerService


# -----------------------------------------------------------------------------
# 1. Query Analyzer Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_analyzer_simple_factual():
    analyzer = QueryAnalyzer(llm_service=MockLLMService())
    result = await analyzer.analyze("What is the primary encryption algorithm used?")
    assert isinstance(result, QueryClassification)
    assert result.query_type in ("simple_factual", "section_lookup", "technical_keyword")
    assert result.needs_decomposition is False
    assert result.complexity in ("low", "medium")


@pytest.mark.asyncio
async def test_query_analyzer_comparison():
    analyzer = QueryAnalyzer(llm_service=MockLLMService())
    result = await analyzer.analyze("Compare the data retention policy of User Accounts vs Audit Logs")
    assert isinstance(result, QueryClassification)
    assert result.query_type == "comparison"
    assert result.needs_decomposition is True
    assert result.complexity in ("medium", "high")


@pytest.mark.asyncio
async def test_query_analyzer_multi_document():
    analyzer = QueryAnalyzer(llm_service=MockLLMService())
    result = await analyzer.analyze(
        "Synthesize the architecture design from the architecture overview and the security whitepaper"
    )
    assert isinstance(result, QueryClassification)
    assert result.query_type == "multi_document"
    assert result.needs_decomposition is True


@pytest.mark.asyncio
async def test_query_analyzer_no_answer_off_topic():
    analyzer = QueryAnalyzer(llm_service=MockLLMService())
    result = await analyzer.analyze("What is the recipe for baking chocolate cake?")
    assert isinstance(result, QueryClassification)
    assert result.query_type == "no_answer"
    assert result.needs_decomposition is False


# -----------------------------------------------------------------------------
# 2. Query Decomposer Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_decomposer_simple_query():
    decomposer = QueryDecomposer(llm_service=MockLLMService())
    classification = QueryClassification(
        query_type="simple_factual",
        complexity="low",
        needs_decomposition=False,
        reason="Simple question",
    )
    plans = await decomposer.decompose("What is the encryption algorithm?", classification)
    assert len(plans) == 1
    assert plans[0].sub_query_id.lower() == "sq-1"
    assert plans[0].sub_query_text == "What is the encryption algorithm?"
    assert plans[0].parent_query == "What is the encryption algorithm?"


@pytest.mark.asyncio
async def test_query_decomposer_comparison_query():
    decomposer = QueryDecomposer(llm_service=MockLLMService())
    query = "Compare user retention vs system audit retention"
    classification = QueryClassification(
        query_type="comparison",
        complexity="medium",
        needs_decomposition=True,
        reason="Comparison requires distinct entity lookups",
    )
    plans = await decomposer.decompose(query, classification)
    assert len(plans) >= 2
    for plan in plans:
        assert isinstance(plan, SubQueryPlan)
        assert plan.sub_query_id.lower().startswith("sq-")
        assert plan.parent_query == query
        assert len(plan.sub_query_text) > 0
        assert len(plan.purpose) > 0


@pytest.mark.asyncio
async def test_query_decomposer_multi_document_query():
    decomposer = QueryDecomposer(llm_service=MockLLMService())
    query = "Analyze security policies across engineering and HR guidelines"
    classification = QueryClassification(
        query_type="multi_document",
        complexity="high",
        needs_decomposition=True,
        reason="Multiple source guidelines",
    )
    plans = await decomposer.decompose(query, classification)
    assert len(plans) >= 2


# -----------------------------------------------------------------------------
# 3. Evidence Aggregator Tests (Deduplication, Provenance, Budget)
# -----------------------------------------------------------------------------

def test_evidence_aggregator_deduplication_and_provenance():
    aggregator = EvidenceAggregator(context_max_tokens=2000, max_chunks=8)

    chunk_a = {
        "chunk_id": "chunk-alpha",
        "document_id": "doc-1",
        "filename": "security_policy.md",
        "page_number": 1,
        "section_title": "Security Controls",
        "text": "AES-256 encryption is mandated for all stored sensitive data.",
        "score": 0.85,
        "rerank_score": 0.95,
    }
    chunk_b = {
        "chunk_id": "chunk-beta",
        "document_id": "doc-2",
        "filename": "audit_policy.md",
        "page_number": 2,
        "section_title": "Audit Logging",
        "text": "Audit logs must be retained for at least 365 calendar days.",
        "score": 0.80,
        "rerank_score": 0.88,
    }
    # chunk_a discovered again by sub-query 2 with lower rerank score
    chunk_a_dupe = {
        "chunk_id": "chunk-alpha",
        "document_id": "doc-1",
        "filename": "security_policy.md",
        "page_number": 1,
        "section_title": "Security Controls",
        "text": "AES-256 encryption is mandated for all stored sensitive data.",
        "score": 0.90,
        "rerank_score": 0.92,
    }

    retrieval_results = {
        "sq-1": [chunk_a],
        "sq-2": [chunk_a_dupe, chunk_b],
    }

    aggregated = aggregator.aggregate_evidence(
        retrieval_results=retrieval_results,
        parent_query="Compare encryption and retention",
    )

    # Verify deduplication: chunk-alpha appears only ONCE
    chunk_ids = [item.chunk_id for item in aggregated]
    assert chunk_ids == ["chunk-alpha", "chunk-beta"]
    assert len(aggregated) == 2

    # Verify provenance tracking: chunk-alpha retains both sub-query IDs
    alpha_item = next(item for item in aggregated if item.chunk_id == "chunk-alpha")
    assert set(alpha_item.discovered_by_sub_queries) == {"sq-1", "sq-2"}
    assert "Query -> SubQuery" in alpha_item.provenance
    assert "sq-1" in alpha_item.provenance
    assert "sq-2" in alpha_item.provenance

    # Verify highest rerank score preserved
    assert alpha_item.rerank_score == 0.95


def test_evidence_aggregator_budget_truncation():
    # Small token budget (20 tokens * 4 chars/token = 80 chars max)
    aggregator = EvidenceAggregator(context_max_tokens=20, max_chunks=8)

    chunk_1 = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "text": "Short text one.",  # ~16 chars
        "score": 0.95,
        "rerank_score": 0.95,
    }
    chunk_2 = {
        "chunk_id": "chunk-2",
        "document_id": "doc-1",
        "text": "This second chunk text is substantially longer and will exceed the tight token budget limit.",
        "score": 0.85,
        "rerank_score": 0.85,
    }

    retrieval_results = {"sq-1": [chunk_1, chunk_2]}
    aggregated = aggregator.aggregate_evidence(
        retrieval_results=retrieval_results,
        parent_query="Budget test",
    )

    # Should retain the top-ranked chunk and truncate the second
    assert len(aggregated) == 1
    assert aggregated[0].chunk_id == "chunk-1"


# -----------------------------------------------------------------------------
# 4. Sufficiency Evaluator Tests
# -----------------------------------------------------------------------------

def test_sufficiency_evaluator_sufficient():
    evaluator = SufficiencyEvaluator(min_relevance_threshold=0.3)
    evidence = [
        AgentEvidenceItem(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="AES-256 encryption is used for data at rest.",
            retrieval_score=0.9,
            rerank_score=0.95,
            discovered_by_sub_queries=["sq-1"],
            provenance="sq-1",
        )
    ]
    sub_plans = [
        SubQueryPlan(
            sub_query_id="sq-1",
            sub_query_text="What encryption algorithm is used?",
            purpose="Check encryption",
            parent_query="What encryption algorithm is used?",
        )
    ]
    classification = QueryClassification(
        query_type="simple_factual",
        complexity="low",
        needs_decomposition=False,
    )
    retrieval_results = {
        "sq-1": [{"chunk_id": "chunk-1", "score": 0.9, "rerank_score": 0.95}]
    }

    assessment = evaluator.evaluate_deterministic(
        query="What encryption algorithm is used?",
        classification=classification,
        sub_queries=sub_plans,
        retrieval_results=retrieval_results,
        aggregated_evidence=evidence,
    )

    assert isinstance(assessment, SufficiencyAssessment)
    assert assessment.sufficient is True
    assert len(assessment.missing_information) == 0


def test_sufficiency_evaluator_insufficient_empty():
    evaluator = SufficiencyEvaluator(min_relevance_threshold=0.5)
    sub_plans = [
        SubQueryPlan(
            sub_query_id="sq-1",
            sub_query_text="Quantum entanglement algorithms?",
            purpose="Check physics",
            parent_query="Quantum entanglement algorithms?",
        )
    ]
    classification = QueryClassification(
        query_type="simple_factual",
        complexity="low",
        needs_decomposition=False,
    )

    assessment = evaluator.evaluate_deterministic(
        query="Quantum entanglement algorithms?",
        classification=classification,
        sub_queries=sub_plans,
        retrieval_results={"sq-1": []},
        aggregated_evidence=[],
    )

    assert isinstance(assessment, SufficiencyAssessment)
    assert assessment.sufficient is False
    assert len(assessment.missing_information) >= 1


# -----------------------------------------------------------------------------
# 5. ResearchAgentGraph Execution & Edge Routing Tests
# -----------------------------------------------------------------------------

class MockRetrievalForGraph:
    """Mock retrieval service delivering configurable chunk dictionaries for agent graph tests."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks

    async def search(self, query: str, top_k: int = 10, **kwargs) -> Dict[str, Any]:
        return {
            "query": query,
            "results": self.chunks[:top_k],
            "total": len(self.chunks),
            "latency_ms": 1.0,
            "breakdown_ms": {},
        }


@pytest.mark.asyncio
async def test_agent_graph_simple_query_execution():
    test_settings = Settings()
    mock_llm = MockLLMService()
    mock_reranker = MockRerankerService()

    sample_chunks = [
        {
            "chunk_id": "chunk-arch-1",
            "document_id": "doc-arch",
            "filename": "architecture.md",
            "chunk_index": 0,
            "page_number": 1,
            "section_title": "Architecture",
            "section_path": "System > Architecture",
            "text": "The service uses event-driven microservices architecture communicating via Kafka.",
            "token_count": 15,
            "score": 0.92,
        }
    ]
    mock_retrieval = MockRetrievalForGraph(sample_chunks)

    graph = ResearchAgentGraph(
        settings=test_settings,
        retrieval_service=mock_retrieval,
        reranker_service=mock_reranker,
        llm_service=mock_llm,
    )

    input_state = {
        "query": "What message broker does the microservices architecture use?",
        "top_k": 3,
        "candidate_top_k": 5,
        "rerank": True,
        "max_iterations": 2,
        "enable_decomposition": True,
    }

    result = await graph.execute(input_state)

    assert result["status"] == "completed"
    assert result["iteration_count"] == 1
    assert result["evidence_sufficient"] is True
    assert len(result["aggregated_evidence"]) >= 1
    assert len(result["final_answer"]) > 0
    assert len(result["execution_trace"]) >= 5

    # Check node order in trace
    executed_nodes = [t["node"] for t in result["execution_trace"]]
    assert "analyze_query" in executed_nodes
    assert "plan_query" in executed_nodes
    assert "retrieve" in executed_nodes
    assert "rerank" in executed_nodes
    assert "assemble_evidence" in executed_nodes
    assert "assess_sufficiency" in executed_nodes
    assert "generate" in executed_nodes
    assert "verify_citations" in executed_nodes


@pytest.mark.asyncio
async def test_agent_graph_multi_subquery_decomposition():
    test_settings = Settings()
    mock_llm = MockLLMService()
    mock_reranker = MockRerankerService()

    sample_chunks = [
        {
            "chunk_id": "c-ret-1",
            "document_id": "doc-policy",
            "filename": "retention_policy.md",
            "chunk_index": 0,
            "page_number": 1,
            "section_title": "User Data Retention",
            "section_path": "Policy > Retention",
            "text": "User account data is retained for 3 years following account closure.",
            "token_count": 15,
            "score": 0.91,
        },
        {
            "chunk_id": "c-ret-2",
            "document_id": "doc-policy",
            "filename": "retention_policy.md",
            "chunk_index": 1,
            "page_number": 2,
            "section_title": "Audit Retention",
            "section_path": "Policy > Retention",
            "text": "Audit security logs are retained for 7 years in immutable storage.",
            "token_count": 15,
            "score": 0.89,
        },
    ]
    mock_retrieval = MockRetrievalForGraph(sample_chunks)

    graph = ResearchAgentGraph(
        settings=test_settings,
        retrieval_service=mock_retrieval,
        reranker_service=mock_reranker,
        llm_service=mock_llm,
    )

    input_state = {
        "query": "Compare user data retention vs audit logs retention",
        "top_k": 3,
        "candidate_top_k": 5,
        "rerank": True,
        "max_iterations": 2,
        "enable_decomposition": True,
    }

    result = await graph.execute(input_state)

    assert result["status"] == "completed"
    assert result["query_type"] == "comparison"
    assert len(result["sub_queries"]) >= 2
    assert len(result["aggregated_evidence"]) >= 1


@pytest.mark.asyncio
async def test_agent_graph_loop_protection_and_abstention():
    """Verifies that an unanswerable query with zero evidence iterates up to max_iterations and cleanly abstains."""
    test_settings = Settings()
    mock_llm = MockLLMService()
    mock_reranker = MockRerankerService()

    # Empty retrieval results simulating zero relevant evidence
    empty_retrieval = MockRetrievalForGraph([])

    graph = ResearchAgentGraph(
        settings=test_settings,
        retrieval_service=empty_retrieval,
        reranker_service=mock_reranker,
        llm_service=mock_llm,
    )

    input_state = {
        "query": "What is the secret recipe for interstellar dark matter propulsion fuel?",
        "top_k": 3,
        "candidate_top_k": 5,
        "rerank": True,
        "max_iterations": 2,  # Strict limit: max 2 iterations
        "enable_decomposition": True,
    }

    result = await graph.execute(input_state)

    # Must abstain after hitting max_iterations
    assert result["status"] == "abstained"
    assert result["iteration_count"] == 2
    assert result["evidence_sufficient"] is False
    assert len(result["aggregated_evidence"]) == 0
    assert "not contain sufficient evidence" in result["final_answer"]

    # Trace must confirm refine_queries was attempted and loop terminated cleanly
    executed_nodes = [t["node"] for t in result["execution_trace"]]
    assert "refine_queries" in executed_nodes
    assert "abstain" in executed_nodes
    assert "verify_citations" in executed_nodes


@pytest.mark.asyncio
async def test_agent_graph_evidence_isolation():
    """Verifies that internal planner reasoning and trace logs NEVER contaminate the evidence context."""
    test_settings = Settings()
    mock_llm = MockLLMService()
    mock_reranker = MockRerankerService()

    chunk_content = "Specific factual document knowledge about system ports."
    sample_chunks = [
        {
            "chunk_id": "chunk-sec-1",
            "document_id": "doc-sec",
            "filename": "security.md",
            "chunk_index": 0,
            "page_number": 1,
            "section_title": "Ports",
            "section_path": "Security > Network",
            "text": chunk_content,
            "token_count": 12,
            "score": 0.94,
        }
    ]
    mock_retrieval = MockRetrievalForGraph(sample_chunks)

    graph = ResearchAgentGraph(
        settings=test_settings,
        retrieval_service=mock_retrieval,
        reranker_service=mock_reranker,
        llm_service=mock_llm,
    )

    input_state = {
        "query": "Which ports are opened for system communication?",
        "top_k": 3,
        "candidate_top_k": 5,
        "rerank": True,
        "max_iterations": 2,
    }

    result = await graph.execute(input_state)

    # Evidence items must only contain document chunk text and metadata
    for ev in result["aggregated_evidence"]:
        # Verify text is purely document content
        assert "Specific factual document knowledge" in ev["text"]
        # Verify planner reasoning is NOT injected into evidence text
        assert "QueryClassification" not in ev["text"]
        assert "execution_trace" not in ev["text"]
        assert "sub_query_id" not in ev["text"]


@pytest.mark.asyncio
async def test_agent_graph_citation_integrity():
    """Verifies that all citations reference valid chunk IDs from the evidence pool."""
    test_settings = Settings()
    mock_llm = MockLLMService()
    mock_reranker = MockRerankerService()

    sample_chunks = [
        {
            "chunk_id": "valid-chunk-id-123",
            "document_id": "doc-security",
            "filename": "security.md",
            "chunk_index": 0,
            "page_number": 4,
            "section_title": "Authentication",
            "section_path": "Security > Auth",
            "text": "OAuth 2.0 with PKCE authorization code grant is required.",
            "token_count": 12,
            "score": 0.96,
        }
    ]
    mock_retrieval = MockRetrievalForGraph(sample_chunks)

    graph = ResearchAgentGraph(
        settings=test_settings,
        retrieval_service=mock_retrieval,
        reranker_service=mock_reranker,
        llm_service=mock_llm,
    )

    result = await graph.execute({"query": "What authentication protocol is required?"})

    assert len(result["citations"]) >= 1
    evidence_chunk_ids = {ev["chunk_id"] for ev in result["aggregated_evidence"]}

    for cit in result["citations"]:
        assert cit["chunk_id"] in evidence_chunk_ids
