"""Evidence aggregation and provenance tracking engine for multi-sub-query retrieval."""

from typing import Any, Dict, List, Optional
from backend.app.agents.state import AgentEvidenceItem
from backend.app.utils.logger import logger


class EvidenceAggregator:
    """Aggregates, deduplicates, and ranks evidence retrieved across multiple sub-queries."""

    def __init__(self, context_max_tokens: int = 1500, max_chunks: int = 8):
        self.context_max_tokens = context_max_tokens
        self.max_chunks = max_chunks

    def aggregate_evidence(
        self,
        retrieval_results: Dict[str, List[Dict[str, Any]]],
        parent_query: str,
    ) -> List[AgentEvidenceItem]:
        """Combines chunks across all sub-queries, deduplicates, and tracks discovery provenance."""
        chunk_map: Dict[str, Dict[str, Any]] = {}
        discovery_map: Dict[str, List[str]] = {}

        for sq_id, chunks in retrieval_results.items():
            for chunk in chunks:
                cid = chunk.get("chunk_id")
                if not cid:
                    continue

                if cid not in discovery_map:
                    discovery_map[cid] = []
                if sq_id not in discovery_map[cid]:
                    discovery_map[cid].append(sq_id)

                if cid not in chunk_map:
                    chunk_map[cid] = chunk
                else:
                    # Update scores to highest observed
                    existing = chunk_map[cid]
                    ex_score = existing.get("score", 0.0)
                    new_score = chunk.get("score", 0.0)
                    if new_score > ex_score:
                        existing["score"] = new_score

                    ex_rerank = existing.get("rerank_score")
                    new_rerank = chunk.get("rerank_score")
                    if new_rerank is not None:
                        if ex_rerank is None or new_rerank > ex_rerank:
                            existing["rerank_score"] = new_rerank

        if not chunk_map:
            return []

        # Sort aggregated items: by rerank_score if present, else by score
        def sort_key(item: Dict[str, Any]) -> float:
            rerank = item.get("rerank_score")
            if rerank is not None:
                return float(rerank)
            return float(item.get("score", 0.0))

        sorted_chunks = sorted(chunk_map.values(), key=sort_key, reverse=True)

        # Pack within budget
        aggregated_items: List[AgentEvidenceItem] = []
        accumulated_chars = 0
        char_budget = self.context_max_tokens * 4

        for ch in sorted_chunks:
            cid = ch.get("chunk_id", "")
            doc_id = ch.get("document_id", "")
            filename = ch.get("filename") or ch.get("document_title")
            page_no = ch.get("page_number")
            section = ch.get("section_title")
            text = ch.get("text", "")
            sq_discovered = discovery_map.get(cid, [])

            # Provenance chain
            sq_str = ", ".join(sq_discovered)
            provenance = (
                f"Query -> SubQuery [{sq_str}] -> Chunk {cid[:8]} -> "
                f"Doc {filename or doc_id[:8]} (Page {page_no or 'N/A'} | Section: {section or 'General'})"
            )

            chunk_len = len(text)
            if accumulated_chars + chunk_len > char_budget and len(aggregated_items) >= 1:
                # Truncate packing if exceeding budget, ensuring at least 1 top chunk is retained
                break

            evidence_item = AgentEvidenceItem(
                chunk_id=cid,
                document_id=doc_id,
                filename=filename,
                page_number=page_no,
                section_title=section,
                text=text,
                retrieval_score=round(float(ch.get("score", 0.0)), 6),
                rerank_score=round(float(ch.get("rerank_score")), 6) if ch.get("rerank_score") is not None else None,
                discovered_by_sub_queries=sq_discovered,
                provenance=provenance,
            )
            aggregated_items.append(evidence_item)
            accumulated_chars += chunk_len

            if len(aggregated_items) >= self.max_chunks:
                break

        logger.info(
            f"evidence_aggregation_completed: sub_queries={len(retrieval_results)} "
            f"unique_chunks={len(chunk_map)} packed={len(aggregated_items)} chars={accumulated_chars}"
        )
        return aggregated_items

    def format_context_block(self, evidence: List[AgentEvidenceItem]) -> str:
        """Formats aggregated evidence into standardized context text with block headers."""
        blocks: List[str] = []
        for idx, item in enumerate(evidence, start=1):
            sq_tag = ", ".join(item.discovered_by_sub_queries)
            header = (
                f"[Source: {item.filename or item.document_id} | Chunk: {item.chunk_id} | "
                f"SubQuery: {sq_tag} | Page: {item.page_number or 'N/A'} | "
                f"Section: {item.section_title or 'General'} | Score: {item.rerank_score or item.retrieval_score}]"
            )
            blocks.append(f"{header}\n{item.text}")
        return "\n\n".join(blocks)
