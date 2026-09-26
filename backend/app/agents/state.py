"""State definitions for LangGraph agentic research workflows."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


from backend.app.observability.context import get_request_id


class SubQueryPlan(BaseModel):
    """Structured plan for an individual decomposed sub-query."""
    sub_query_id: str
    sub_query_text: str
    purpose: str
    parent_query: str
    expected_document: Optional[str] = None
    priority: int = 1


class ExecutionTraceItem(BaseModel):
    """Structured audit trace for an individual agent graph node execution."""
    node: str
    iteration: int
    input_summary: str
    output_summary: str
    duration_ms: float
    request_id: str = Field(default_factory=get_request_id)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentEvidenceItem(BaseModel):
    """Retrieved evidence chunk enriched with sub-query provenance and discovery tracking."""
    chunk_id: str
    document_id: str
    filename: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    text: str
    retrieval_score: float = 0.0
    rerank_score: Optional[float] = None
    discovered_by_sub_queries: List[str] = Field(default_factory=list)
    provenance: str = ""


class QueryClassification(BaseModel):
    """Classification of query complexity, type, and decomposition requirement."""
    query_type: str  # simple_factual, semantic_lookup, technical_keyword, section_lookup, multi_document, comparison, multi_hop, no_answer
    complexity: str  # low, medium, high
    needs_decomposition: bool = False
    reason: str = ""
    suggested_sub_queries: List[str] = Field(default_factory=list)


class SufficiencyAssessment(BaseModel):
    """Assessment of whether retrieved evidence is sufficient to synthesize a grounded answer."""
    sufficient: bool
    confidence: float = 1.0
    reason: str = ""
    missing_information: List[str] = Field(default_factory=list)
    new_sub_queries: List[SubQueryPlan] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    """Typed state for LangGraph research workflow execution."""
    original_query: str
    normalized_query: str
    query_type: str
    complexity: str
    needs_decomposition: bool

    # Sub-query planning
    sub_queries: List[Dict[str, Any]]
    completed_sub_queries: List[str]

    # Retrieval and aggregation
    retrieval_results: Dict[str, List[Dict[str, Any]]]
    aggregated_evidence: List[Dict[str, Any]]
    evidence_scores: Dict[str, float]

    # Iteration & Sufficiency
    iteration_count: int
    max_iterations: int
    evidence_sufficient: bool
    sufficiency_reason: str
    missing_information: List[str]

    # Final outputs
    generation_result: Optional[Dict[str, Any]]
    citation_audit: Optional[Dict[str, Any]]
    final_answer: str
    claims: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]

    # Audit & Observability
    execution_trace: List[Dict[str, Any]]
    errors: List[str]
    status: str
    timings_ms: Dict[str, float]

    # Request parameters
    top_k: int
    candidate_top_k: int
    rerank: bool
    enable_decomposition: bool
    context_max_tokens: int
    temperature: float
    max_tokens: int
