import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from backend.app.config.settings import Settings, get_settings
from backend.app.models.schemas import (
    CitationItem,
    ClaimItem,
    ClaimVerificationItem,
    VerificationSummary,
)
from backend.app.utils.logger import logger

# Common English stop words for transparent lexical support verification
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "also", "furthermore", "however", "therefore",
}


class CitationVerificationService:
    """
    Deterministically verifies claim factuality and resolves grounded citations
    against the retrieved evidence chunks.
    Detects hallucinated chunk IDs, measures lexical/entity support, and marks claims
    as 'supported', 'partially_supported', or 'unsupported'.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def verify_claims(
        self,
        claims: List[ClaimItem],
        evidence_chunks: List[Dict[str, Any]],
    ) -> Tuple[VerificationSummary, List[CitationItem], float]:
        """
        Evaluates claims against retrieved evidence chunks.
        Returns:
            - VerificationSummary: Audit details for each claim and overall status.
            - List[CitationItem]: Bound, deterministic citation objects.
            - latency_ms: Execution duration in milliseconds.
        """
        t_start = time.perf_counter()

        chunks_by_id: Dict[str, Dict[str, Any]] = {}
        chunks_by_doc_id: Dict[str, List[Dict[str, Any]]] = {}
        for c in evidence_chunks:
            cid = c.get("chunk_id")
            did = c.get("document_id")
            if cid:
                chunks_by_id[cid] = c
            if did:
                chunks_by_doc_id.setdefault(did, []).append(c)

        verification_details: List[ClaimVerificationItem] = []
        supported_cnt = 0
        partial_cnt = 0
        unsupported_cnt = 0
        cited_valid_chunk_ids: Set[str] = set()

        for claim in claims:
            v_item = self._verify_single_claim(claim, chunks_by_id, chunks_by_doc_id)
            verification_details.append(v_item)

            if v_item.status == "supported":
                supported_cnt += 1
                cited_valid_chunk_ids.update(v_item.verified_chunk_ids)
            elif v_item.status == "partially_supported":
                partial_cnt += 1
                cited_valid_chunk_ids.update(v_item.verified_chunk_ids)
            else:
                unsupported_cnt += 1

        # Determine overall verification status
        total_claims = len(claims)
        if total_claims == 0:
            overall_status = "unverified"
        elif unsupported_cnt == 0 and partial_cnt == 0:
            overall_status = "verified"
        elif supported_cnt > 0:
            overall_status = "partially_verified"
        else:
            overall_status = "unverified"

        summary = VerificationSummary(
            status=overall_status,
            supported_claims=supported_cnt,
            partially_supported_claims=partial_cnt,
            unsupported_claims=unsupported_cnt,
            total_claims=total_claims,
            details=verification_details,
        )

        # Build citations from verified and retrieved chunks
        citations = self._build_citations(cited_valid_chunk_ids, chunks_by_id, evidence_chunks)

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        logger.info(
            f"citation_verification_completed: status='{overall_status}' supported={supported_cnt}/"
            f"{total_claims} unsupported={unsupported_cnt} citations={len(citations)} elapsed_ms={latency_ms}"
        )

        return summary, citations, latency_ms

    def _verify_single_claim(
        self,
        claim: ClaimItem,
        chunks_by_id: Dict[str, Dict[str, Any]],
        chunks_by_doc_id: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> ClaimVerificationItem:
        """
        Verifies individual claim against referenced chunk IDs.
        """
        cited_ids = claim.evidence_chunk_ids
        verified_ids: List[str] = []
        missing_ids: List[str] = []
        doc_map = chunks_by_doc_id or {}

        for cid in cited_ids:
            if cid in chunks_by_id:
                verified_ids.append(cid)
            elif cid in doc_map:
                for dc in doc_map[cid]:
                    chunk_key = dc.get("chunk_id")
                    if chunk_key and chunk_key not in verified_ids:
                        verified_ids.append(chunk_key)
            else:
                missing_ids.append(cid)

        # Case 1: No valid chunks cited or all cited chunk IDs are hallucinated
        if not verified_ids:
            rationale = "No valid retrieved evidence chunks cited."
            if missing_ids:
                rationale += f" Hallucinated or unknown chunk IDs: {missing_ids}."
            return ClaimVerificationItem(
                claim_id=claim.claim_id,
                text=claim.text,
                status="unsupported",
                support_score=0.0,
                cited_chunk_ids=cited_ids,
                verified_chunk_ids=[],
                missing_chunk_ids=missing_ids,
                rationale=rationale,
            )

        # Case 2: Concatenate evidence text from all verified chunks
        evidence_text = " ".join([chunks_by_id[cid].get("text", "") for cid in verified_ids])
        support_score = self._compute_text_support_score(claim.text, evidence_text)

        min_threshold = self.settings.CITATION_MIN_SUPPORT_THRESHOLD
        partial_threshold = self.settings.CITATION_PARTIAL_SUPPORT_THRESHOLD

        if support_score >= min_threshold:
            status = "supported"
            rationale = f"Claim supported by evidence ({round(support_score * 100, 1)}% textual overlap)."
        elif support_score >= partial_threshold:
            status = "partially_supported"
            rationale = f"Claim has partial evidence support ({round(support_score * 100, 1)}% textual overlap)."
        else:
            status = "unsupported"
            rationale = f"Claim lacks textual support in cited chunk ({round(support_score * 100, 1)}% textual overlap)."

        if missing_ids:
            rationale += f" Note: Unknown chunk IDs {missing_ids} were excluded."

        return ClaimVerificationItem(
            claim_id=claim.claim_id,
            text=claim.text,
            status=status,
            support_score=round(support_score, 4),
            cited_chunk_ids=cited_ids,
            verified_chunk_ids=verified_ids,
            missing_chunk_ids=missing_ids,
            rationale=rationale,
        )

    def _compute_text_support_score(self, claim_text: str, evidence_text: str) -> float:
        """
        Computes transparent lexical and entity support between claim text and evidence text.
        Balances significant term overlap with bigram phrase coherence.
        """
        claim_tokens = self._tokenize_significant_terms(claim_text)
        evidence_tokens = self._tokenize_significant_terms(evidence_text)

        if not claim_tokens:
            return 1.0  # Empty claim or stop-words only

        evidence_set = set(evidence_tokens)
        matched_terms = [t for t in claim_tokens if t in evidence_set]
        term_overlap = len(matched_terms) / len(claim_tokens)

        # Bigram phrase overlap
        claim_bigrams = self._extract_bigrams(claim_text)
        evidence_bigrams = set(self._extract_bigrams(evidence_text))

        if claim_bigrams:
            matched_bigrams = [bg for bg in claim_bigrams if bg in evidence_bigrams]
            bigram_overlap = len(matched_bigrams) / len(claim_bigrams)
        else:
            bigram_overlap = term_overlap

        # Weighted combination: 60% term overlap, 40% bigram phrase coherence
        combined_score = 0.60 * term_overlap + 0.40 * bigram_overlap
        return min(max(combined_score, 0.0), 1.0)

    @staticmethod
    def _tokenize_significant_terms(text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if len(w) > 2 and w not in STOPWORDS and not w.isdigit()]

    @staticmethod
    def _extract_bigrams(text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 2:
            return []
        return [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]

    def _build_citations(
        self,
        cited_chunk_ids: Set[str],
        chunks_by_id: Dict[str, Dict[str, Any]],
        all_evidence_chunks: List[Dict[str, Any]],
    ) -> List[CitationItem]:
        """
        Builds standardized, deterministic CitationItem objects.
        If no specific chunks were cited, includes top retrieved chunk for provenance.
        """
        target_ids = cited_chunk_ids
        if not target_ids and all_evidence_chunks:
            # Fallback to top retrieved chunk
            top_id = all_evidence_chunks[0].get("chunk_id")
            if top_id:
                target_ids = {top_id}

        citations: List[CitationItem] = []
        for idx, cid in enumerate(sorted(target_ids), start=1):
            chunk = chunks_by_id.get(cid)
            if not chunk:
                continue

            doc_id = chunk.get("document_id") or "unknown-doc"
            page = chunk.get("page_number")
            sec = chunk.get("section_path") or chunk.get("section_title") or "General"
            score = chunk.get("score")
            text = chunk.get("text", "")

            page_str = f"Page: {page}" if page is not None else "Page: N/A"
            doc_short = doc_id[:8] if len(doc_id) >= 8 else doc_id
            formatted = f"[Doc: {doc_short} | {page_str} | Section: {sec}]"

            snippet = text[:150].strip() + ("..." if len(text) > 150 else "")

            citations.append(
                CitationItem(
                    citation_id=f"CIT-{idx}",
                    chunk_id=cid,
                    document_id=doc_id,
                    page_number=page,
                    section_title=chunk.get("section_title"),
                    section_path=chunk.get("section_path"),
                    score=score,
                    formatted_citation=formatted,
                    snippet=snippet,
                )
            )

        return citations


_citation_service_instance: Optional[CitationVerificationService] = None


def get_citation_verification_service(
    settings: Optional[Settings] = None,
) -> CitationVerificationService:
    """Returns singleton CitationVerificationService instance."""
    global _citation_service_instance
    if _citation_service_instance is None:
        _citation_service_instance = CitationVerificationService(settings=settings)
    return _citation_service_instance


def reset_citation_verification_service() -> None:
    """Resets singleton instance for testing."""
    global _citation_service_instance
    _citation_service_instance = None
