import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Plus
from backend.app.config.settings import Settings, get_settings
from backend.app.utils.logger import logger


class BM25IndexService:
    """
    In-memory BM25 lexical index with disk persistence and full relational rebuildability.
    Uses BM25Plus to guarantee strictly positive IDF scores even on small document corpora.
    Maintains keyword index over document chunks with technical term preservation.
    """

    TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_\-\.]+\b")

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.index_path = Path(self.settings.BM25_INDEX_PATH)
        self._chunks: List[Dict[str, Any]] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Plus] = None
        self._chunk_id_map: Dict[str, int] = {}
        self._load_from_disk_if_exists()

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        Tokenizes text into lowercase tokens while preserving technical identifiers,
        hyphenated words, and alphanumeric codes.
        """
        if not text:
            return []
        matches = cls.TOKEN_PATTERN.findall(text.lower())
        # Filter out empty or pure punctuation tokens
        return [m for m in matches if any(c.isalnum() for c in m)]

    def _build_indexed_text(self, chunk: Dict[str, Any]) -> str:
        """Combines section headers and body text for lexical matching."""
        parts = []
        if chunk.get("section_title"):
            parts.append(chunk["section_title"])
        if chunk.get("section_path"):
            parts.append(chunk["section_path"])
        parts.append(chunk.get("text", ""))
        return " ".join(parts)

    def _rebuild_model(self) -> None:
        """Reconstructs BM25Plus instance from current in-memory chunks."""
        self._chunk_id_map = {c["chunk_id"]: i for i, c in enumerate(self._chunks)}
        self._tokenized_corpus = [
            self.tokenize(self._build_indexed_text(c)) for c in self._chunks
        ]
        if self._tokenized_corpus and any(len(doc) > 0 for doc in self._tokenized_corpus):
            self._bm25 = BM25Plus(self._tokenized_corpus)
        else:
            self._bm25 = None

    def add_chunks(self, chunks: List[Dict[str, Any]], persist: bool = True) -> int:
        """
        Idempotently adds or updates chunks in the BM25 index.
        """
        if not chunks:
            return 0

        # Create copy of current chunks excluding any being updated
        incoming_ids = {c["chunk_id"] for c in chunks}
        updated_chunks = [c for c in self._chunks if c["chunk_id"] not in incoming_ids]
        updated_chunks.extend(chunks)

        self._chunks = updated_chunks
        self._rebuild_model()

        if persist:
            self.save_to_disk()

        logger.info(f"BM25 index updated: {len(chunks)} chunks added/updated. Total chunks: {len(self._chunks)}.")
        return len(chunks)

    def remove_document(self, document_id: str, persist: bool = True) -> int:
        """Removes all chunks belonging to a document from the BM25 index."""
        initial_count = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.get("document_id") != document_id]
        removed = initial_count - len(self._chunks)

        if removed > 0:
            self._rebuild_model()
            if persist:
                self.save_to_disk()
            logger.info(f"Removed {removed} chunks for document '{document_id}' from BM25 index.")
        return removed

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index for the top-K lexical matches.
        Returns results sorted by BM25 score descending.
        """
        if not self._bm25 or not self._chunks:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        query_token_set = set(tokens)

        # Pair scores with chunk metadata
        doc_id_set = set(document_ids) if document_ids else None
        scored_results: List[Dict[str, Any]] = []

        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            # Ensure the document actually contains at least one token from the query
            doc_tokens = self._tokenized_corpus[idx]
            if not query_token_set.intersection(doc_tokens):
                continue

            chunk = self._chunks[idx]
            if doc_id_set and chunk.get("document_id") not in doc_id_set:
                continue

            scored_results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                    "document_id": chunk.get("document_id"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text": chunk.get("text", ""),
                    "page_number": chunk.get("page_number"),
                    "section_title": chunk.get("section_title"),
                    "section_path": chunk.get("section_path"),
                }
            )

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def clear(self, persist: bool = True) -> None:
        """Clears all indexed chunks."""
        self._chunks = []
        self._tokenized_corpus = []
        self._bm25 = None
        self._chunk_id_map = {}
        if persist and self.index_path.exists():
            try:
                os.remove(self.index_path)
            except Exception as exc:
                logger.warning(f"Could not remove BM25 index file: {exc}")
        logger.info("BM25 index cleared.")

    def save_to_disk(self, path: Optional[Path] = None) -> None:
        """Serializes the in-memory chunks state to disk."""
        target_path = path or self.index_path
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                pickle.dump(self._chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved BM25 index with {len(self._chunks)} chunks to '{target_path}'.")
        except Exception as exc:
            logger.error(f"Failed to persist BM25 index to disk at '{target_path}': {exc}")

    def _load_from_disk_if_exists(self) -> None:
        """Loads serialized chunks from disk if index file exists."""
        if not self.index_path.exists():
            return
        try:
            with open(self.index_path, "rb") as f:
                chunks = pickle.load(f)
            if isinstance(chunks, list):
                self._chunks = chunks
                self._rebuild_model()
                logger.info(f"Loaded {len(self._chunks)} chunks into BM25 index from '{self.index_path}'.")
        except Exception as exc:
            logger.warning(f"Failed to load BM25 index from '{self.index_path}': {exc}. Starting fresh.")
            self._chunks = []
            self._rebuild_model()

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the BM25 index."""
        vocab_size = len(self._bm25.idf) if self._bm25 and hasattr(self._bm25, "idf") else 0
        avg_doc_len = self._bm25.avgdl if self._bm25 and hasattr(self._bm25, "avgdl") else 0.0
        return {
            "total_chunks": len(self._chunks),
            "vocabulary_size": vocab_size,
            "average_document_length": round(float(avg_doc_len), 2),
            "index_path": str(self.index_path),
            "is_indexed": self._bm25 is not None,
        }


_bm25_service_instance: Optional[BM25IndexService] = None


def get_bm25_service(settings: Optional[Settings] = None) -> BM25IndexService:
    """Returns singleton BM25 service instance."""
    global _bm25_service_instance
    if _bm25_service_instance is None:
        _bm25_service_instance = BM25IndexService(settings=settings)
    return _bm25_service_instance
