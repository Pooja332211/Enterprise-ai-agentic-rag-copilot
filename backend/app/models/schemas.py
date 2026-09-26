from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Document entity public representation."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    normalized_filename: str
    mime_type: str
    file_extension: str
    file_size_bytes: int
    file_hash: str
    status: str
    page_count: Optional[int] = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of document entities."""
    items: List[DocumentResponse]
    total: int
    limit: int
    offset: int


class DocumentChunkResponse(BaseModel):
    """Document chunk public representation with rich source metadata."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    character_count: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: datetime


class DocumentChunkListResponse(BaseModel):
    """Paginated list of document chunks for a specific document."""
    document_id: str
    items: List[DocumentChunkResponse]
    total: int
    limit: int
    offset: int


class IngestionJobResponse(BaseModel):
    """Ingestion job lifecycle and progress representation."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    status: str
    stage: str
    progress: float
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class IngestionResponse(BaseModel):
    """Result returned by the document upload endpoint."""
    document: DocumentResponse
    job: Optional[IngestionJobResponse] = None
    is_duplicate: bool = False
    message: str


class ErrorResponse(BaseModel):
    """Standardized API error response format."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------------------------
# Milestone 3: Search & Retrieval Schemas
# ------------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Retrieval query request schema."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query string")
    mode: str = Field("hybrid", description="Retrieval mode: 'hybrid', 'dense', or 'sparse'")
    top_k: int = Field(10, ge=1, le=100, description="Maximum number of chunks to return")
    document_ids: Optional[List[str]] = Field(None, description="Optional document IDs to filter by")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight factor between dense and sparse search")
    fusion_method: str = Field("rrf", description="Fusion method: 'rrf' (default) or 'weighted'")
    candidate_top_k: Optional[int] = Field(None, ge=1, le=100, description="Candidate pool size for first-stage retrieval before reranking")
    rerank: bool = Field(True, description="Whether to apply cross-encoder reranking to retrieved candidates")
    min_relevance_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional minimum score threshold to filter out low-relevance chunks")
    context_assembly: bool = Field(False, description="Whether to assemble structured context with budget management and deduplication")
    context_max_tokens: Optional[int] = Field(None, ge=1, description="Override maximum token budget for context assembly")
    context_max_characters: Optional[int] = Field(None, ge=1, description="Override maximum character budget for context assembly")


class SearchResultItem(BaseModel):
    """Normalized retrieved chunk evidence metadata."""
    chunk_id: str
    document_id: Optional[str] = None
    chunk_index: int = 0
    text: str
    score: float
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    retrieval_sources: List[str]


class AssembledContextResponse(BaseModel):
    """Structured assembled context response with budget management and deduplication."""
    total_chunks: int
    total_tokens: int
    total_characters: int
    budget_tokens_limit: int
    budget_characters_limit: int
    is_truncated: bool
    context_text: str
    packed_chunks: List[SearchResultItem]


class SearchResponse(BaseModel):
    """Search query response format."""
    query: str
    mode: str
    fusion_method: str
    total: int
    latency_ms: float
    breakdown_ms: Dict[str, float]
    reranked: bool = False
    results: List[SearchResultItem]
    assembled_context: Optional[AssembledContextResponse] = None


class IndexRebuildResponse(BaseModel):
    """Result of full index rebuild."""
    message: str
    total_documents: int
    total_chunks: int
    dense_indexed: int
    bm25_indexed: int
    elapsed_seconds: float


class IndexDocumentResponse(BaseModel):
    """Result of single document re-indexing."""
    document_id: str
    chunks_indexed: int
    message: str


class IndexStatsResponse(BaseModel):
    """Statistics for vector and lexical indexes."""
    qdrant: Dict[str, Any]
    bm25: Dict[str, Any]
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int


# ------------------------------------------------------------------------------
# Milestone 5: Grounded Generation, Citations & Research Schemas
# ------------------------------------------------------------------------------

class ClaimItem(BaseModel):
    """Individual factual proposition with referenced evidence chunk IDs."""
    claim_id: str = Field(..., description="Unique claim identifier e.g. 'C1'")
    text: str = Field(..., description="Factual proposition statement")
    evidence_chunk_ids: List[str] = Field(default_factory=list, description="IDs of evidence chunks supporting this claim")


class GroundedGenerationOutput(BaseModel):
    """Strict structured JSON contract expected from LLM generation."""
    answer: str = Field(..., description="Synthesized grounded answer")
    claims: List[ClaimItem] = Field(default_factory=list, description="Extracted factual claims")
    insufficient_evidence: bool = Field(False, description="Flag indicating if available context is insufficient")


class CitationItem(BaseModel):
    """Deterministic citation derived strictly from retrieved evidence metadata."""
    citation_id: str
    chunk_id: str
    document_id: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    score: Optional[float] = None
    formatted_citation: str
    snippet: str


class ClaimVerificationItem(BaseModel):
    """Verification assessment for an individual claim."""
    claim_id: str
    text: str
    status: Literal["supported", "partially_supported", "unsupported"]
    support_score: float
    cited_chunk_ids: List[str]
    verified_chunk_ids: List[str]
    missing_chunk_ids: List[str]
    rationale: str


class VerificationSummary(BaseModel):
    """Aggregate citation and claim verification audit report."""
    status: Literal["verified", "partially_verified", "unverified"]
    supported_claims: int
    partially_supported_claims: int
    unsupported_claims: int
    total_claims: int
    details: List[ClaimVerificationItem]


class ResearchRequest(BaseModel):
    """Agentic RAG research query request schema."""
    query: str = Field(..., min_length=1, max_length=2000, description="Research query or topic")
    mode: str = Field("hybrid", description="Retrieval mode: 'hybrid', 'dense', or 'sparse'")
    top_k: int = Field(5, ge=1, le=50, description="Final number of evidence chunks to assemble")
    candidate_top_k: int = Field(20, ge=1, le=100, description="Candidate pool size for first-stage retrieval")
    rerank: bool = Field(True, description="Whether to apply cross-encoder neural reranking")
    context_max_tokens: Optional[int] = Field(None, ge=1, description="Context budget token limit")
    min_relevance_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score threshold cutoff")
    document_ids: Optional[List[str]] = Field(None, description="Optional document filter")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature override")
    max_tokens: Optional[int] = Field(None, ge=1, description="Max generated tokens override")


class ResearchResponse(BaseModel):
    """Grounded research synthesis response with citation verification audit."""
    query: str
    answer: str
    claims: List[ClaimItem]
    citations: List[CitationItem]
    verification: VerificationSummary
    evidence: List[SearchResultItem]
    timings_ms: Dict[str, float]
    model: str
    provider: str
    grounded: bool
    insufficient_evidence: bool = False


class AgentResearchRequest(BaseModel):
    """Request schema for LangGraph agentic research workflow."""
    query: str = Field(..., min_length=1, max_length=2000, description="Research query or topic")
    top_k: int = Field(5, ge=1, le=50, description="Final number of evidence chunks per sub-query")
    candidate_top_k: int = Field(20, ge=1, le=100, description="Candidate pool size for first-stage retrieval")
    rerank: bool = Field(True, description="Whether to apply cross-encoder neural reranking")
    max_iterations: int = Field(2, ge=1, le=5, description="Maximum iterative retrieval refinement cycles")
    enable_decomposition: bool = Field(True, description="Whether to enable multi-sub-query decomposition")
    context_max_tokens: int = Field(1500, ge=100, le=8000, description="Maximum token budget for assembled evidence")
    temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(2048, ge=1, description="Max generated tokens")


class AgentResearchResponse(BaseModel):
    """Response schema from LangGraph agentic research workflow."""
    query: str
    final_answer: str
    query_type: str
    complexity: str
    sub_queries: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    claims: List[ClaimItem]
    citations: List[CitationItem]
    verification: VerificationSummary
    iterations_used: int
    execution_trace: List[Dict[str, Any]]
    timings_ms: Dict[str, float]
    total_duration_ms: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    grounded: bool
    insufficient_evidence: bool = False
    status: str = "completed"


