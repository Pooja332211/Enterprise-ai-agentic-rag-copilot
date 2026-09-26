"""Cache invalidation manager for document lifecycle and index updates."""

from typing import Dict
from backend.app.cache.embedding_cache import get_embedding_cache
from backend.app.cache.reranker_cache import get_reranker_cache
from backend.app.utils.logger import logger


def invalidate_all_caches() -> Dict[str, int]:
    """
    Clears all application caches (embeddings and reranking).
    Invoked when documents are deleted or when vector/BM25 indexes are rebuilt.
    """
    emb_cleared = get_embedding_cache().clear()
    rerank_cleared = get_reranker_cache().clear()
    logger.info(f"invalidate_all_caches: cleared {emb_cleared} embeddings, {rerank_cleared} rerank scores.")
    return {
        "embedding_entries_cleared": emb_cleared,
        "reranker_entries_cleared": rerank_cleared,
    }


def invalidate_document_cache(document_id: str) -> int:
    """Invalidates cache entries associated with a deleted document."""
    # Reranker cache stores chunk_id hashes; clearing reranker cache prevents stale chunks
    rerank_cleared = get_reranker_cache().clear()
    logger.info(f"invalidate_document_cache: document_id='{document_id}' cleared {rerank_cleared} reranker entries.")
    return rerank_cleared
