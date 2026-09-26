from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class CorpusDocumentInfo(BaseModel):
    """Metadata describing a reference document in the evaluation corpus."""
    filename: str
    format: str
    title: str
    description: str


class BenchmarkQuery(BaseModel):
    """Evaluation query specification with ground-truth labels."""
    query_id: str
    query: str
    category: Literal[
        "factual_lookup",
        "section_lookup",
        "technical_keyword",
        "semantic_paraphrase",
        "multi_document",
        "no_answer",
    ]
    difficulty: Literal["easy", "medium", "hard"]
    relevant_document_filenames: List[str] = Field(default_factory=list)
    relevant_section_paths: List[str] = Field(default_factory=list)
    expected_answer: Optional[str] = None
    expected_abstention: bool = False
    notes: str = ""


class BenchmarkDataset(BaseModel):
    """Root container for a versioned evaluation dataset."""
    version: str
    dataset_name: str
    description: str
    created_at: str
    document_corpus: List[CorpusDocumentInfo] = Field(default_factory=list)
    queries: List[BenchmarkQuery] = Field(default_factory=list)


class RetrievalMetrics(BaseModel):
    """Retrieval ranking quality metrics for a given configuration."""
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    mean_latency_ms: float


class LatencyDistribution(BaseModel):
    """Statistical latency distribution (mean, median, p95)."""
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    count: int


class PipelineLatencyProfile(BaseModel):
    """Breakdown of latencies across all multi-stage research pipeline stages."""
    embedding_latency: LatencyDistribution
    dense_search_latency: LatencyDistribution
    bm25_search_latency: LatencyDistribution
    fusion_latency: LatencyDistribution
    reranking_latency: LatencyDistribution
    context_assembly_latency: LatencyDistribution
    generation_latency: LatencyDistribution
    citation_verification_latency: LatencyDistribution
    total_pipeline_latency: LatencyDistribution


class CitationEvaluationMetrics(BaseModel):
    """Aggregated citation and grounding quality metrics."""
    total_queries_evaluated: int
    total_claims: int
    total_citations: int
    citation_validity_rate: float
    citation_support_rate: float
    citation_coverage_rate: float
    unsupported_claim_rate: float
    partially_supported_rate: float


class AbstentionEvaluationMetrics(BaseModel):
    """Safeguard and hallucination resistance metrics on ungrounded queries."""
    total_no_answer_queries: int
    correct_abstentions: int
    incorrect_answers: int
    unsupported_answers: int
    abstention_accuracy: float


class GenerationEvaluationMetrics(BaseModel):
    """Semantic and lexical answer quality metrics across grounded queries."""
    total_answered_queries: int
    mean_answer_completeness: float
    mean_evidence_grounding_score: float
    mean_reference_token_overlap: float


class ConfigurationAblation(BaseModel):
    """Row in retrieval ablation comparison table."""
    configuration: str
    recall_at_5: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    latency_ms: float


class BenchmarkRunSummary(BaseModel):
    """Complete machine-readable summary of an evaluation run."""
    dataset_version: str
    timestamp: str
    environment: Dict[str, Any]
    total_queries: int
    evaluated_answered_queries: int
    evaluated_no_answer_queries: int
    ablation: List[ConfigurationAblation]
    retrieval_configurations: Dict[str, RetrievalMetrics]
    latency_profile: PipelineLatencyProfile
    generation_metrics: GenerationEvaluationMetrics
    citation_metrics: CitationEvaluationMetrics
    abstention_metrics: AbstentionEvaluationMetrics
    auditable_traces: List[Dict[str, Any]] = Field(default_factory=list)
