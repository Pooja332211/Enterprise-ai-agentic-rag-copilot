import re
from typing import Any, Dict, List, Optional
from backend.app.config.settings import Settings, get_settings
from backend.app.utils.logger import logger


class ContextAssemblyService:
    """
    Assembles, deduplicates, and packs ranked retrieval evidence chunks into
    a strictly budgeted context window suitable for downstream LLM synthesis.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates token count (~4 characters per token heuristic)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def _is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.85) -> bool:
        """Determines if two text passages are near-duplicates using Jaccard word similarity."""
        if text_a == text_b:
            return True
        tokens_a = set(re.findall(r"\b\w+\b", text_a.lower()))
        tokens_b = set(re.findall(r"\b\w+\b", text_b.lower()))
        if not tokens_a or not tokens_b:
            return False
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return (intersection / union) >= threshold

    def assemble_context(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        max_characters: Optional[int] = None,
        deduplicate: bool = True,
    ) -> Dict[str, Any]:
        """
        Packs ranked chunks into a context budget with deduplication and metadata headers.
        """
        token_budget = max_tokens or self.settings.CONTEXT_MAX_TOKENS
        char_budget = max_characters or self.settings.CONTEXT_MAX_CHARACTERS

        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "total_characters": 0,
                "budget_tokens_limit": token_budget,
                "budget_characters_limit": char_budget,
                "is_truncated": False,
                "context_text": "",
                "packed_chunks": [],
            }

        packed_chunks: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        seen_chunk_ids = set()
        seen_texts: List[str] = []

        accumulated_tokens = 0
        accumulated_chars = 0
        is_truncated = False

        for chunk in chunks:
            cid = chunk.get("chunk_id")
            text = chunk.get("text", "").strip()
            if not text:
                continue

            # 1. Deduplication
            if deduplicate:
                if cid and cid in seen_chunk_ids:
                    continue
                if any(self._is_near_duplicate(text, prev) for prev in seen_texts):
                    continue

            # 2. Formatted chunk block with metadata header
            doc_id = chunk.get("document_id", "unknown")
            page_num = chunk.get("page_number")
            page_str = f"Page: {page_num}" if page_num is not None else "Page: N/A"
            sec = chunk.get("section_path") or chunk.get("section_title") or "General"
            score = chunk.get("rerank_score") if chunk.get("rerank_score") is not None else chunk.get("score")
            score_str = f"Score: {score:.4f}" if score is not None else ""

            header = f"[Source: {doc_id} | Chunk: {cid} | {page_str} | Section: {sec} | {score_str}]".strip()
            formatted_block = f"{header}\n{text}"

            block_tokens = self.estimate_tokens(formatted_block)
            block_chars = len(formatted_block)

            # 3. Budget Check
            if (accumulated_tokens + block_tokens > token_budget) or (accumulated_chars + block_chars > char_budget):
                is_truncated = True
                break

            # Pack chunk
            packed_chunks.append(dict(chunk))
            context_blocks.append(formatted_block)
            if cid:
                seen_chunk_ids.add(cid)
            seen_texts.append(text)

            accumulated_tokens += block_tokens
            accumulated_chars += block_chars

        full_context_text = "\n\n---\n\n".join(context_blocks)

        logger.info(
            f"context_assembled: packed={len(packed_chunks)}/{len(chunks)} "
            f"tokens={accumulated_tokens}/{token_budget} chars={accumulated_chars}/{char_budget} "
            f"truncated={is_truncated}"
        )

        return {
            "total_chunks": len(packed_chunks),
            "total_tokens": accumulated_tokens,
            "total_characters": accumulated_chars,
            "budget_tokens_limit": token_budget,
            "budget_characters_limit": char_budget,
            "is_truncated": is_truncated,
            "context_text": full_context_text,
            "packed_chunks": packed_chunks,
        }


_context_assembly_instance: Optional[ContextAssemblyService] = None


def get_context_assembly_service(settings: Optional[Settings] = None) -> ContextAssemblyService:
    """Returns singleton ContextAssemblyService instance."""
    global _context_assembly_instance
    if _context_assembly_instance is None:
        _context_assembly_instance = ContextAssemblyService(settings=settings)
    return _context_assembly_instance
