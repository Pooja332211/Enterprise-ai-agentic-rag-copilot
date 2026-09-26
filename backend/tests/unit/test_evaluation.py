import math
import pytest
from pathlib import Path

from backend.app.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_latency_percentiles,
    calculate_citation_metrics,
    calculate_abstention_metrics,
)
from backend.app.evaluation.generation_evaluator import (
    compute_token_f1,
    compute_bigram_f1,
    GenerationEvaluator,
)
from backend.app.evaluation.citation_evaluator import CitationEvaluator
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.evaluation.schemas import BenchmarkDataset


def test_calculate_recall_at_k():
    relevant = {"doc1", "doc2", "doc3"}
    retrieved = ["doc1", "doc4", "doc2", "doc5", "doc6"]

    # k=1: only doc1 -> 1/3
    assert math.isclose(calculate_recall_at_k(retrieved, relevant, k=1), 1.0 / 3.0, rel_tol=1e-4)

    # k=3: doc1 and doc2 -> 2/3
    assert math.isclose(calculate_recall_at_k(retrieved, relevant, k=3), 2.0 / 3.0, rel_tol=1e-4)

    # k=5: doc1 and doc2 -> 2/3
    assert math.isclose(calculate_recall_at_k(retrieved, relevant, k=5), 2.0 / 3.0, rel_tol=1e-4)

    # Empty relevant -> 0.0
    assert calculate_recall_at_k(retrieved, set(), k=5) == 0.0

    # k <= 0 -> 0.0
    assert calculate_recall_at_k(retrieved, relevant, k=0) == 0.0


def test_calculate_precision_at_k():
    relevant = {"doc1", "doc2"}
    retrieved = ["doc1", "doc3", "doc2", "doc4"]

    # k=1: doc1 is relevant -> 1/1 = 1.0
    assert calculate_precision_at_k(retrieved, relevant, k=1) == 1.0

    # k=2: doc1 relevant, doc3 not -> 1/2 = 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.5

    # k=3: doc1, doc2 relevant -> 2/3
    assert math.isclose(calculate_precision_at_k(retrieved, relevant, k=3), 2.0 / 3.0, rel_tol=1e-4)

    # k=4: 2/4 = 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=4) == 0.5

    # k <= 0 -> 0.0
    assert calculate_precision_at_k(retrieved, relevant, k=0) == 0.0


def test_calculate_mrr():
    relevant = {"doc_target"}

    # Target at rank 1 -> MRR = 1.0
    assert calculate_mrr(["doc_target", "doc2"], relevant) == 1.0

    # Target at rank 2 -> MRR = 0.5
    assert calculate_mrr(["doc1", "doc_target", "doc3"], relevant) == 0.5

    # Target at rank 4 -> MRR = 0.25
    assert calculate_mrr(["doc1", "doc2", "doc3", "doc_target"], relevant) == 0.25

    # Target not found -> MRR = 0.0
    assert calculate_mrr(["doc1", "doc2"], relevant) == 0.0

    # Target beyond k -> 0.0
    assert calculate_mrr(["doc1", "doc2", "doc_target"], relevant, k=2) == 0.0


def test_calculate_ndcg_at_k():
    relevant = {"doc1", "doc2"}

    # Perfect ranking: both relevant items at rank 1 and 2
    perfect = ["doc1", "doc2", "doc3", "doc4"]
    assert math.isclose(calculate_ndcg_at_k(perfect, relevant, k=2), 1.0, rel_tol=1e-4)
    assert math.isclose(calculate_ndcg_at_k(perfect, relevant, k=5), 1.0, rel_tol=1e-4)

    # Suboptimal ranking: relevant items at rank 2 and 3
    suboptimal = ["doc3", "doc1", "doc2", "doc4"]
    ndcg_sub = calculate_ndcg_at_k(suboptimal, relevant, k=5)
    assert 0.0 < ndcg_sub < 1.0

    # Zero hits
    assert calculate_ndcg_at_k(["doc3", "doc4"], relevant, k=2) == 0.0


def test_calculate_latency_percentiles():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    dist = calculate_latency_percentiles(values)

    assert dist.mean_ms == 30.0
    assert dist.median_ms == 30.0
    assert dist.min_ms == 10.0
    assert dist.max_ms == 50.0
    assert dist.count == 5
    assert dist.p95_ms >= 40.0

    # Empty list
    empty_dist = calculate_latency_percentiles([])
    assert empty_dist.count == 0
    assert empty_dist.mean_ms == 0.0


def test_calculate_citation_metrics():
    sample_audits = [
        {
            "claims": [
                {"claim_text": "Claim 1", "citation_chunk_ids": ["c1"]},
                {"claim_text": "Claim 2", "citation_chunk_ids": ["c2"]},
            ],
            "citations": [
                {"chunk_id": "c1", "document_id": "d1"},
                {"chunk_id": "c2", "document_id": "d1"},
            ],
            "claim_verifications": [
                {"claim_text": "Claim 1", "verification_status": "supported"},
                {"claim_text": "Claim 2", "verification_status": "unsupported"},
            ],
        }
    ]

    metrics = calculate_citation_metrics(sample_audits)
    assert metrics.total_claims == 2
    assert metrics.total_citations == 2
    assert metrics.citation_validity_rate == 1.0
    assert metrics.citation_coverage_rate == 1.0
    assert metrics.citation_support_rate == 0.5
    assert metrics.unsupported_claim_rate == 0.5


def test_calculate_abstention_metrics():
    evals = [
        {"query_id": "q1", "abstained": True, "insufficient_evidence": True, "unsupported_claims_count": 0},
        {"query_id": "q2", "abstained": True, "insufficient_evidence": True, "unsupported_claims_count": 0},
        {"query_id": "q3", "abstained": False, "insufficient_evidence": False, "unsupported_claims_count": 1},
    ]

    metrics = calculate_abstention_metrics(evals)
    assert metrics.total_no_answer_queries == 3
    assert metrics.correct_abstentions == 2
    assert metrics.incorrect_answers == 1
    assert math.isclose(metrics.abstention_accuracy, 2.0 / 3.0, rel_tol=1e-3)


def test_token_and_bigram_f1():
    text_a = "The system provides grounded answers with source citations."
    text_b = "The system provides grounded answers with citations."

    f1 = compute_token_f1(text_a, text_b)
    assert 0.7 < f1 <= 1.0

    bigram_f1 = compute_bigram_f1(text_a, text_b)
    assert 0.5 < bigram_f1 <= 1.0

    # Disjoint text
    disjoint_f1 = compute_token_f1("Apple banana cherry", "Quantum computer physics")
    assert disjoint_f1 == 0.0


def test_benchmark_dataset_validation():
    runner = EvaluationRunner()
    dataset = runner.load_dataset()

    assert isinstance(dataset, BenchmarkDataset)
    assert dataset.version == "1.0.0"
    assert len(dataset.queries) >= 15
    assert len(dataset.document_corpus) == 5

    categories = {q.category for q in dataset.queries}
    expected_categories = {
        "factual_lookup",
        "section_lookup",
        "technical_keyword",
        "semantic_paraphrase",
        "multi_document",
        "no_answer",
    }
    assert expected_categories.issubset(categories)

    # Verify no-answer queries have expected_abstention=True
    for q in dataset.queries:
        if q.category == "no_answer":
            assert q.expected_abstention is True
            assert len(q.relevant_document_filenames) == 0
        else:
            assert q.expected_abstention is False
            assert len(q.relevant_document_filenames) > 0
