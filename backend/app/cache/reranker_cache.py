"""Cross-encoder neural reranking score cache keyed by query, chunk ID, and content hash."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from backend.app.cache.memory import InMemoryLRUCache


class RerankerCache:
    """
    Caches expensive cross-encoder joint relevance scores to eliminate redundant
    inference across iterative agent loops and repeated sub-queries.
    """

    def __init__(self, max_size: int = 20000, ttl_seconds: int = 86400):
        self._cache = InMemoryLRUCache(max_size=max_size, default_ttl_seconds=ttl_seconds, name="reranker")

    @staticmethod
    def generate_key(query: str, chunk_id: str, chunk_text: str, model_name: str) -> str:
        """Constructs deterministic SHA-256 cache key."""
        q_norm = " ".join(query.strip().lower().split())
        t_hash = hashlib.sha256(chunk_text.strip().encode("utf-8")).hexdigest()[:16]
        raw = f"{model_name}:{q_norm}:{chunk_id}:{t_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_chunk_id(cand: Dict[str, Any], default_idx: int = 0) -> str:
        cid = cand.get("chunk_id") or cand.get("id")
        if cid:
            return str(cid)
        text = cand.get("text", "")
        if text:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return str(default_idx)

    def get(self, query: str, chunk_id: str, chunk_text: str, model_name: str) -> Optional[float]:
        key = self.generate_key(query, chunk_id, chunk_text, model_name)
        return self._cache.get(key)

    def set(
        self,
        query: str,
        chunk_id: str,
        chunk_text: str,
        model_name: str,
        score: float,
        ttl: Optional[int] = None,
    ) -> None:
        key = self.generate_key(query, chunk_id, chunk_text, model_name)
        self._cache.set(key, float(score), ttl_seconds=ttl)

    def get_batch(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        model_name: str,
    ) -> Tuple[Dict[int, float], List[Tuple[int, Dict[str, Any]]]]:
        """
        Splits candidates into cached hits and un-scored misses.
        Returns:
            hits: Dict mapping candidate index to cached rerank score
            misses: List of (candidate_index, candidate_dict) requiring model inference
        """
        hits: Dict[int, float] = {}
        misses: List[Tuple[int, Dict[str, Any]]] = []

        for idx, cand in enumerate(candidates):
            cid = self._extract_chunk_id(cand, idx)
            text = cand.get("text", "")
            cached_score = self.get(query, cid, text, model_name)
            if cached_score is not None:
                hits[idx] = cached_score
            else:
                misses.append((idx, cand))

        return hits, misses

    def set_batch(
        self,
        query: str,
        scored_items: List[Tuple[Dict[str, Any], float]],
        model_name: str,
        ttl: Optional[int] = None,
    ) -> None:
        for idx, (cand, score) in enumerate(scored_items):
            cid = self._extract_chunk_id(cand, idx)
            text = cand.get("text", "")
            if text:
                self.set(query, cid, text, model_name, score, ttl=ttl)

    def clear(self) -> int:
        return self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        return self._cache.stats()


# Global Singleton Reranker Cache
_reranker_cache_instance = RerankerCache()


def get_reranker_cache() -> RerankerCache:
    """Returns singleton RerankerCache instance."""
    return _reranker_cache_instance
