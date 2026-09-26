"""Unit tests for bounded LRU caches, batch lookups, TTL expiry, and lifecycle invalidation."""

import time
import pytest

from backend.app.cache.embedding_cache import EmbeddingCache
from backend.app.cache.invalidation import invalidate_all_caches, invalidate_document_cache
from backend.app.cache.memory import InMemoryLRUCache
from backend.app.cache.reranker_cache import RerankerCache


def test_in_memory_lru_capacity_and_eviction():
    # Cache with capacity 3
    cache = InMemoryLRUCache(max_size=3, name="test_lru")

    cache.set("k1", "val1")
    cache.set("k2", "val2")
    cache.set("k3", "val3")

    assert cache.get("k1") == "val1"
    assert cache.get("k2") == "val2"
    assert cache.get("k3") == "val3"
    assert cache.size == 3

    # Access k1 so k2 becomes least recently used
    _ = cache.get("k1")

    # Insert k4, which must evict k2
    cache.set("k4", "val4")

    assert cache.size == 3
    assert cache.get("k2") is None  # evicted
    assert cache.get("k1") == "val1"
    assert cache.get("k3") == "val3"
    assert cache.get("k4") == "val4"


def test_in_memory_lru_ttl_expiration():
    cache = InMemoryLRUCache(max_size=10, default_ttl_seconds=1, name="test_ttl")

    cache.set("ephemeral", "data", ttl_seconds=1)
    assert cache.get("ephemeral") == "data"

    time.sleep(1.1)
    # Must expire
    assert cache.get("ephemeral") is None


def test_embedding_cache_batch_operations():
    cache = EmbeddingCache(max_size=100)
    cache.clear()

    model = "BAAI/bge-small-en-v1.5"
    dim = 384

    texts = [
        "First document text for embedding",
        "Second document text for embedding",
        "Third document text for embedding",
    ]

    # Initially all misses
    hits, misses = cache.get_batch(texts, model, dim)
    assert len(hits) == 0
    assert len(misses) == 3

    # Simulate computing and storing embeddings
    computed = [
        (texts[0], [0.1] * dim),
        (texts[2], [0.3] * dim),
    ]
    cache.set_batch(computed, model, dim)

    # Next lookup: 0 and 2 should hit, 1 should miss
    hits, misses = cache.get_batch(texts, model, dim)
    assert 0 in hits
    assert 2 in hits
    assert len(misses) == 1
    assert misses[0][0] == 1
    assert misses[0][1] == texts[1]

    stats = cache.stats()
    assert stats["hits"] >= 2
    assert stats["misses"] >= 4
    assert stats["size"] == 2


def test_reranker_cache_batch_operations():
    cache = RerankerCache(max_size=100)
    cache.clear()

    query = "financial revenue"
    candidates = [
        {"chunk_id": "c1", "text": "Q3 financial revenue rose 15 percent."},
        {"chunk_id": "c2", "text": "Workplace cafeteria menu for Friday."},
        {"chunk_id": "c3", "text": "Enterprise cloud revenue breakdown."},
    ]
    model = "BAAI/bge-reranker-base"

    # All misses initially
    hits, misses = cache.get_batch(query, candidates, model)
    assert len(hits) == 0
    assert len(misses) == 3

    # Cache scores for c1 and c3
    scored = [
        (candidates[0], 0.95),
        (candidates[2], 0.88),
    ]
    cache.set_batch(query, scored, model)

    # Re-query: c1 and c3 hit, c2 misses
    hits, misses = cache.get_batch(query, candidates, model)
    assert hits[0] == 0.95
    assert hits[2] == 0.88
    assert len(misses) == 1
    assert misses[0][0] == 1
    assert misses[0][1]["chunk_id"] == "c2"


def test_cache_invalidation():
    emb_cache = EmbeddingCache()
    rerank_cache = RerankerCache()

    emb_cache.set("sample text", "test-model", 384, [0.1, 0.2])
    rerank_cache.set("query", "chunk-1", "sample text", "test-model", 0.9)

    assert emb_cache.get("sample text", "test-model", 384) is not None
    assert rerank_cache.get("query", "chunk-1", "sample text", "test-model") is not None

    # Invalidate document cache
    inv_count = invalidate_document_cache("doc-uuid-123")
    assert inv_count >= 0

    # Invalidate all caches
    cleared = invalidate_all_caches()
    assert cleared["embedding_entries_cleared"] >= 0
    assert cleared["reranker_entries_cleared"] >= 0
