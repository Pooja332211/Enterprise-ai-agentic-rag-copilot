import re
from typing import List, Dict, Any, Optional
from backend.app.evaluation.schemas import GenerationEvaluationMetrics


def _tokenize(text: str) -> List[str]:
    """Tokenize lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_token_f1(prediction: str, reference: str) -> float:
    """Computes token-level precision, recall, and F1 score."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    common = pred_set & ref_set
    if not common:
        return 0.0

    precision = len(common) / float(len(pred_set))
    recall = len(common) / float(len(ref_set))
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def compute_bigram_f1(prediction: str, reference: str) -> float:
    """Computes bigram phrase overlap F1 score."""
    p_tokens = _tokenize(prediction)
    r_tokens = _tokenize(reference)
    if len(p_tokens) < 2 or len(r_tokens) < 2:
        return compute_token_f1(prediction, reference)

    pred_bigrams = set(zip(p_tokens[:-1], p_tokens[1:]))
    ref_bigrams = set(zip(r_tokens[:-1], r_tokens[1:]))
    common = pred_bigrams & ref_bigrams
    if not common:
        return 0.0

    precision = len(common) / float(len(pred_bigrams))
    recall = len(common) / float(len(ref_bigrams))
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


class GenerationEvaluator:
    """
    Evaluates generated research responses against reference ground truth answers
    and assembled context evidence.
    """

    def evaluate_response(
        self,
        generated_answer: str,
        reference_answer: Optional[str],
        retrieved_contexts: List[str],
        claims: List[Dict[str, Any]],
        verifications: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluates a single research response:
        - Completeness & reference overlap (if reference_answer provided)
        - Evidence grounding score based on verified claims
        """
        token_f1 = 0.0
        bigram_f1 = 0.0
        if reference_answer:
            token_f1 = compute_token_f1(generated_answer, reference_answer)
            bigram_f1 = compute_bigram_f1(generated_answer, reference_answer)

        # Grounding score: ratio of supported/partially supported claims
        grounding_score = 0.0
        if verifications:
            supported = sum(
                1 for v in verifications if (v.get("status") or v.get("verification_status")) in ("supported", "partially_supported")
            )
            grounding_score = supported / float(len(verifications))
        elif not claims:
            # Empty claims on abstention
            grounding_score = 1.0

        # Completeness heuristic: minimum length threshold and presence of key factual tokens
        completeness_score = 0.0
        if generated_answer and len(generated_answer.strip()) > 20:
            if reference_answer:
                completeness_score = min(1.0, token_f1 * 1.25)
            else:
                completeness_score = 1.0

        return {
            "token_f1": round(token_f1, 4),
            "bigram_f1": round(bigram_f1, 4),
            "grounding_score": round(grounding_score, 4),
            "completeness_score": round(completeness_score, 4),
        }

    def aggregate_metrics(self, evaluations: List[Dict[str, Any]]) -> GenerationEvaluationMetrics:
        """Aggregates generation evaluation metrics across multiple queries."""
        if not evaluations:
            return GenerationEvaluationMetrics(
                total_answered_queries=0,
                mean_answer_completeness=0.0,
                mean_evidence_grounding_score=0.0,
                mean_reference_token_overlap=0.0,
            )

        n = float(len(evaluations))
        mean_completeness = sum(e["completeness_score"] for e in evaluations) / n
        mean_grounding = sum(e["grounding_score"] for e in evaluations) / n
        mean_overlap = sum(e["token_f1"] for e in evaluations) / n

        return GenerationEvaluationMetrics(
            total_answered_queries=len(evaluations),
            mean_answer_completeness=round(mean_completeness, 4),
            mean_evidence_grounding_score=round(mean_grounding, 4),
            mean_reference_token_overlap=round(mean_overlap, 4),
        )
