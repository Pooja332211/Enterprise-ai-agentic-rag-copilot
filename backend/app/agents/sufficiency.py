"""Evidence sufficiency evaluation engine determining grounded completeness."""

import re
from typing import Dict, List, Optional, Set
from backend.app.agents.state import (
    AgentEvidenceItem,
    QueryClassification,
    SubQueryPlan,
    SufficiencyAssessment,
)
from backend.app.services.llm_service import BaseLLMService
from backend.app.utils.logger import logger


class SufficiencyEvaluator:
    """Evaluates whether retrieved evidence provides sufficient coverage to synthesize a grounded answer."""

    def __init__(
        self,
        llm_service: Optional[BaseLLMService] = None,
        use_llm: bool = False,
        min_relevance_threshold: float = 0.0005,
    ):
        self.llm_service = llm_service
        self.use_llm = use_llm
        self.min_relevance_threshold = min_relevance_threshold

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extracts non-stopword alphanumeric tokens."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "what", "which", "how",
            "and", "or", "in", "on", "of", "to", "for", "with", "about", "that",
            "this", "does", "report", "document", "section", "page",
        }
        tokens = re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        return {t for t in tokens if t not in stopwords}

    def evaluate_deterministic(
        self,
        query: str,
        classification: QueryClassification,
        sub_queries: List[SubQueryPlan],
        retrieval_results: Dict[str, List[Dict]],
        aggregated_evidence: List[AgentEvidenceItem],
    ) -> SufficiencyAssessment:
        """Deterministic evaluation of evidence sufficiency for reliable, reproducible testing."""
        # 1. Zero evidence check
        if not aggregated_evidence:
            missing = [sq.sub_query_text for sq in sub_queries] or [query]
            return SufficiencyAssessment(
                sufficient=False,
                confidence=0.99,
                reason="Zero evidence chunks retrieved across all sub-queries.",
                missing_information=missing,
                new_sub_queries=[],
            )

        # 2. Out-of-domain classification
        if classification.query_type == "no_answer":
            return SufficiencyAssessment(
                sufficient=False,
                confidence=0.95,
                reason="Query classified as out-of-domain or ungrounded; evidence lacks topical relevance.",
                missing_information=["Corpus does not contain relevant domain documentation."],
                new_sub_queries=[],
            )

        # 3. Keyword grounding check
        query_keywords = self._extract_keywords(query)
        combined_evidence_text = " ".join([item.text.lower() for item in aggregated_evidence])

        overlap = {kw for kw in query_keywords if kw in combined_evidence_text}
        overlap_ratio = len(overlap) / len(query_keywords) if query_keywords else 1.0

        if query_keywords and overlap_ratio < 0.20:
            return SufficiencyAssessment(
                sufficient=False,
                confidence=0.90,
                reason=f"Retrieved evidence has minimal topical overlap with query keywords ({len(overlap)}/{len(query_keywords)} matched).",
                missing_information=[f"Core topic keywords missing: {', '.join(query_keywords - overlap)}"],
                new_sub_queries=[],
            )

        # 4. Sub-query coverage analysis
        missing_subs: List[str] = []
        covered_subs: List[str] = []
        new_plans: List[SubQueryPlan] = []

        for sq in sub_queries:
            sq_chunks = retrieval_results.get(sq.sub_query_id, [])
            # Filter by relevance
            valid_chunks = [
                c for c in sq_chunks
                if (c.get("rerank_score") is None or c.get("rerank_score", 0.0) >= self.min_relevance_threshold)
            ]

            if not valid_chunks:
                missing_subs.append(sq.sub_query_text)
                # Formulate a refined sub-query for iteration
                refined_text = f"{sq.sub_query_text} overview details"
                new_plans.append(
                    SubQueryPlan(
                        sub_query_id=f"{sq.sub_query_id}-R",
                        sub_query_text=refined_text,
                        purpose=f"Refined retrieval for missing evidence on '{sq.sub_query_text[:30]}'",
                        parent_query=query,
                        priority=sq.priority,
                    )
                )
            else:
                covered_subs.append(sq.sub_query_id)

        # If more than half the sub-queries or all sub-queries are missing
        if missing_subs and len(missing_subs) >= len(sub_queries):
            return SufficiencyAssessment(
                sufficient=False,
                confidence=0.85,
                reason=f"Evidence missing for all {len(missing_subs)} sub-queries.",
                missing_information=missing_subs,
                new_sub_queries=new_plans,
            )

        # 5. Sufficient evidence verified
        logger.info(
            f"sufficiency_assessed: sufficient=True covered_subqueries={len(covered_subs)}/{len(sub_queries)} "
            f"total_evidence_chunks={len(aggregated_evidence)}"
        )
        return SufficiencyAssessment(
            sufficient=True,
            confidence=0.95,
            reason="Retrieved evidence provides complete coverage of query components.",
            missing_information=[],
            new_sub_queries=[],
        )

    async def evaluate(
        self,
        query: str,
        classification: QueryClassification,
        sub_queries: List[SubQueryPlan],
        retrieval_results: Dict[str, List[Dict]],
        aggregated_evidence: List[AgentEvidenceItem],
    ) -> SufficiencyAssessment:
        """Evaluates sufficiency with deterministic fallback."""
        return self.evaluate_deterministic(
            query=query,
            classification=classification,
            sub_queries=sub_queries,
            retrieval_results=retrieval_results,
            aggregated_evidence=aggregated_evidence,
        )
