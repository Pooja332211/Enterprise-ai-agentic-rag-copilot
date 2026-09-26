"""Dedicated dense embedding cache keyed by normalized text, model, and dimension."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from backend.app.cache.memory import InMemoryLRUCache
from backend.app.utils.logger import logger


class EmbeddingCache:
    """
    Embedding cache with batch lookup capability.
    Generates deterministic SHA-256 keys incorporating text, model, and dimension.
    """

    def __init__(self, max_size: int = 20000, ttl_seconds: int = 86400):
        self._cache = InMemoryLRUCache(max_size=max_size, default_ttl_seconds=ttl_seconds, name="embedding")

    @staticmethod
    def generate_key(text: str, model_name: str, dimension: int) -> str:
        """Constructs deterministic SHA-256 cache key."""
        normalized = " ".join(text.strip().split())
        raw_key = f"{model_name}:{dimension}:{normalized}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str, dimension: int) -> Optional[List[float]]:
        """Retrieves cached vector for single text."""
        key = self.generate_key(text, model_name, dimension)
        return self._cache.get(key)

    def set(self, text: str, model_name: str, dimension: int, vector: List[float], ttl: Optional[int] = None) -> None:
        """Stores vector for single text."""
        key = self.generate_key(text, model_name, dimension)
        self._cache.set(key, vector, ttl_seconds=ttl)

    def get_batch(
        self,
        texts: List[str],
        model_name: str,
        dimension: int,
    ) -> Tuple[Dict[int, List[float]], List[Tuple[int, str]]]:
        """
        Performs batch cache lookup.
        Returns:
            hits: Dict mapping original text index to cached vector
            misses: List of (original_index, text) tuples requiring model computation
        """
        hits: Dict[int, List[float]] = {}
        misses: List[Tuple[int, str]] = []

        for idx, text in enumerate(texts):
            key = self.generate_key(text, model_name, dimension)
            cached_vec = self._cache.get(key)
            if cached_vec is not None:
                hits[idx] = cached_vec
            else:
                misses.append((idx, text))

        return hits, misses

    def set_batch(
        self,
        items: List[Tuple[str, List[float]]],
        model_name: str,
        dimension: int,
        ttl: Optional[int] = None,
    ) -> None:
        """Stores batch of text and vector pairs in cache."""
        for text, vector in items:
            self.set(text, model_name, dimension, vector, ttl=ttl)

    def clear(self) -> int:
        """Invalidates all cached embeddings."""
        return self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Returns cache telemetry."""
        return self._cache.stats()


# Global Singleton Embedding Cache
_embedding_cache_instance = EmbeddingCache()


def get_embedding_cache() -> EmbeddingCache:
    """Returns singleton EmbeddingCache instance."""
    return _embedding_cache_instance
