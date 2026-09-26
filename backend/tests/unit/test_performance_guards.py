"""Unit tests for performance guards, singleton reuse, warmup, and evaluation gates."""

import pytest

from backend.app.cache import get_embedding_cache, get_reranker_cache
from backend.app.evaluation.gates import EvaluationGate, evaluate_gates
from backend.app.services.embedding_service import get_embedding_service
from backend.app.services.reranker_service import get_reranker_service


def test_singleton_cache_instances():
    emb1 = get_embedding_cache()
    emb2 = get_embedding_cache()
    assert emb1 is emb2

    rerank1 = get_reranker_cache()
    rerank2 = get_reranker_cache()
    assert rerank1 is rerank2


def test_model_warmup_execution():
    reranker = get_reranker_service()
    # Warmup should complete without throwing exceptions
    reranker.warmup()

    emb = get_embedding_service()
    if hasattr(emb, "warmup"):
        emb.warmup()


def test_evaluation_gates_passing_and_failing():
    passing_metrics = {
        "recall_at_5": 0.85,
        "mrr": 0.80,
        "ndcg_at_5": 0.78,
        "grounding_accuracy": 0.95,
        "citation_validity": 0.98,
        "abstention_accuracy": 0.92,
        "p95_latency_ms": 1500.0,
    }
    summary = evaluate_gates(passing_metrics)
    assert summary.all_passed is True
    assert summary.critical_passed is True
    assert summary.passed_gates == summary.total_gates

    failing_metrics = {
        "recall_at_5": 0.50,  # Below 0.80
        "mrr": 0.40,         # Below 0.70
        "grounding_accuracy": 0.80, # Below 0.90
        "citation_validity": 0.85,  # Below 0.95
        "abstention_accuracy": 0.50,
        "p95_latency_ms": 25000.0,
    }
    fail_summary = evaluate_gates(failing_metrics)
    assert fail_summary.all_passed is False
    assert fail_summary.critical_passed is False
    assert fail_summary.failed_gates > 0
