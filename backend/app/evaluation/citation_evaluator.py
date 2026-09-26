from typing import List, Dict, Any, Set
from backend.app.evaluation.schemas import CitationEvaluationMetrics
from backend.app.evaluation.metrics import calculate_citation_metrics


class CitationEvaluator:
    """
    Evaluates citation accuracy, validity against retrieved context chunks,
    evidence support rates, and auditable verification traces.
    """

    def audit_query_citations(
        self,
        query_id: str,
        query_text: str,
        retrieved_chunk_ids: Set[str],
        claims: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        verifications: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Creates an auditable trace for a single query research execution:
        query -> claims -> cited chunk IDs -> verification status -> validity.
        """
        claim_audits = []
        verif_by_claim = {
            (v.get("text") or v.get("claim_text", "")): v for v in verifications
        }

        for idx, claim in enumerate(claims):
            c_text = claim.get("text") or claim.get("claim_text", "")
            chunk_ids = claim.get("evidence_chunk_ids") or claim.get("citation_chunk_ids", [])
            v_info = verif_by_claim.get(c_text, {})

            # Check if all cited chunk_ids actually exist in the retrieved pool
            valid_ids = [cid for cid in chunk_ids if cid in retrieved_chunk_ids]
            hallucinated_ids = [cid for cid in chunk_ids if cid not in retrieved_chunk_ids]

            claim_audits.append({
                "claim_index": idx + 1,
                "claim_text": c_text,
                "cited_chunk_ids": chunk_ids,
                "valid_chunk_ids": valid_ids,
                "hallucinated_chunk_ids": hallucinated_ids,
                "has_hallucinated_citations": len(hallucinated_ids) > 0,
                "verification_status": v_info.get("status") or v_info.get("verification_status", "unverified"),
                "support_score": v_info.get("support_score", 0.0),
                "rationale": v_info.get("rationale", ""),
            })

        return {
            "query_id": query_id,
            "query": query_text,
            "claims": claims,
            "citations": citations,
            "claim_verifications": verifications,
            "claim_audits": claim_audits,
        }

    def aggregate_metrics(self, audits: List[Dict[str, Any]]) -> CitationEvaluationMetrics:
        """Aggregates citation metrics across all audited queries."""
        return calculate_citation_metrics(audits)
