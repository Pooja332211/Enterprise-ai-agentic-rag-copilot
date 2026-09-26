import math
from typing import List, Set, Optional, Dict, Any
from backend.app.evaluation.schemas import (
    LatencyDistribution,
    CitationEvaluationMetrics,
    AbstentionEvaluationMetrics,
)


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Computes Recall@K: proportion of relevant items found in top-K retrieved items.
    
    Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|
    """
    if not relevant_ids:
        return 0.0
    if k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & relevant_ids)
    return float(hits) / float(len(relevant_ids))


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Computes Precision@K: proportion of top-K retrieved items that are relevant.
    
    Precision@K = |Retrieved@K ∩ Relevant| / K
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    
    hits = len(set(top_k) & relevant_ids)
    return float(hits) / float(k)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: Set[str], k: Optional[int] = None) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for a single query.
    Returns 1 / rank of the first relevant item retrieved, or 0.0 if none found.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    
    candidates = retrieved_ids[:k] if k is not None else retrieved_ids
    for rank, item_id in enumerate(candidates, start=1):
        if item_id in relevant_ids:
            return 1.0 / float(rank)
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.
    
    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(K, |Relevant|)} (1 / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K
    """
    if not relevant_ids or k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, item_id in enumerate(top_k, start=1):
        rel = 1.0 if item_id in relevant_ids else 0.0
        if rel > 0.0:
            dcg += rel / math.log2(i + 1)
    
    ideal_hits = min(k, len(relevant_ids))
    if ideal_hits == 0:
        return 0.0
    
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    
    return float(dcg / idcg)


def calculate_latency_percentiles(latencies_ms: List[float]) -> LatencyDistribution:
    """
    Calculates statistical latency summary: mean, median, p95, min, max, count.
    """
    if not latencies_ms:
        return LatencyDistribution(
            mean_ms=0.0,
            median_ms=0.0,
            p95_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            count=0,
        )
    
    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    mean = sum(sorted_l) / float(n)
    
    if n % 2 == 1:
        median = sorted_l[n // 2]
    else:
        median = (sorted_l[n // 2 - 1] + sorted_l[n // 2]) / 2.0
    
    p95_idx = min(n - 1, max(0, int(math.ceil(0.95 * n)) - 1))
    p95 = sorted_l[p95_idx]
    
    return LatencyDistribution(
        mean_ms=round(mean, 2),
        median_ms=round(median, 2),
        p95_ms=round(p95, 2),
        min_ms=round(sorted_l[0], 2),
        max_ms=round(sorted_l[-1], 2),
        count=n,
    )


def calculate_citation_metrics(audits: List[Dict[str, Any]]) -> CitationEvaluationMetrics:
    """
    Computes citation evaluation metrics across evaluated research outputs.
    """
    total_claims = 0
    total_citations = 0
    valid_citations = 0
    claims_with_citations = 0
    supported_claims = 0
    partially_supported_claims = 0
    unsupported_claims = 0

    for audit in audits:
        claims = audit.get("claims", [])
        citations = audit.get("citations", [])
        verifications = audit.get("claim_verifications", [])

        total_citations += len(citations)
        for cit in citations:
            # If citation has a valid chunk_id or was resolved to a chunk
            if cit.get("chunk_id") or cit.get("document_id"):
                valid_citations += 1

        total_claims += len(claims)
        for claim in claims:
            chunk_ids = claim.get("evidence_chunk_ids") or claim.get("citation_chunk_ids", [])
            if chunk_ids:
                claims_with_citations += 1

        for v in verifications:
            status = v.get("status") or v.get("verification_status")
            if status == "supported":
                supported_claims += 1
            elif status == "partially_supported":
                partially_supported_claims += 1
            else:
                unsupported_claims += 1

    validity_rate = (valid_citations / total_citations) if total_citations > 0 else 1.0
    support_rate = (supported_claims / total_claims) if total_claims > 0 else 0.0
    coverage_rate = (claims_with_citations / total_claims) if total_claims > 0 else 0.0
    unsupported_rate = (unsupported_claims / total_claims) if total_claims > 0 else 0.0
    partially_rate = (partially_supported_claims / total_claims) if total_claims > 0 else 0.0

    return CitationEvaluationMetrics(
        total_queries_evaluated=len(audits),
        total_claims=total_claims,
        total_citations=total_citations,
        citation_validity_rate=round(validity_rate, 4),
        citation_support_rate=round(support_rate, 4),
        citation_coverage_rate=round(coverage_rate, 4),
        unsupported_claim_rate=round(unsupported_rate, 4),
        partially_supported_rate=round(partially_rate, 4),
    )


def calculate_abstention_metrics(abstention_evaluations: List[Dict[str, Any]]) -> AbstentionEvaluationMetrics:
    """
    Computes abstention metrics on queries designed to have no answer.
    """
    total = len(abstention_evaluations)
    if total == 0:
        return AbstentionEvaluationMetrics(
            total_no_answer_queries=0,
            correct_abstentions=0,
            incorrect_answers=0,
            unsupported_answers=0,
            abstention_accuracy=1.0,
        )

    correct_abstentions = 0
    incorrect_answers = 0
    unsupported_answers = 0

    for item in abstention_evaluations:
        abstained = item.get("abstained", False)
        insufficient_evidence = item.get("insufficient_evidence", False)
        # If the system explicitly abstained or flagged insufficient evidence
        if abstained or insufficient_evidence:
            correct_abstentions += 1
        else:
            incorrect_answers += 1
            if item.get("unsupported_claims_count", 0) > 0:
                unsupported_answers += 1

    accuracy = correct_abstentions / total

    return AbstentionEvaluationMetrics(
        total_no_answer_queries=total,
        correct_abstentions=correct_abstentions,
        incorrect_answers=incorrect_answers,
        unsupported_answers=unsupported_answers,
        abstention_accuracy=round(accuracy, 4),
    )
