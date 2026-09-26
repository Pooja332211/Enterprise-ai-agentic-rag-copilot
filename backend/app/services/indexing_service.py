import time
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.config.settings import Settings, get_settings
from backend.app.database.models import Document, DocumentChunk
from backend.app.services.bm25_service import BM25IndexService, get_bm25_service
from backend.app.services.embedding_service import BaseEmbeddingService, get_embedding_service
from backend.app.services.vector_service import VectorService, get_vector_service
from backend.app.utils.logger import logger


class IndexingService:
    """
    Unified manager for dense vector (Qdrant) and lexical (BM25) indexes.
    Manages indexing lifecycle: ingestion hook, document removal, and full relational rebuild.
    """

    def __init__(
        self,
        vector_service: Optional[VectorService] = None,
        bm25_service: Optional[BM25IndexService] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.vector_service = vector_service or get_vector_service(self.settings)
        self.bm25_service = bm25_service or get_bm25_service(self.settings)
        self.embedding_service = embedding_service or get_embedding_service(self.settings)

    def _prepare_payloads_and_bm25(self, chunks: List[Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extracts payloads for Qdrant and dictionary format for BM25."""
        payloads: List[Dict[str, Any]] = []
        bm25_chunks: List[Dict[str, Any]] = []

        for c in chunks:
            if isinstance(c, dict):
                cid = str(c.get("id") or c.get("chunk_id"))
                doc_id = str(c.get("document_id"))
                idx = c.get("chunk_index", 0)
                txt = c.get("text", "")
                pg = c.get("page_number")
                sec_t = c.get("section_title")
                sec_p = c.get("section_path")
            else:
                cid = str(getattr(c, "id", None) or getattr(c, "chunk_id", None))
                doc_id = str(getattr(c, "document_id", None))
                idx = getattr(c, "chunk_index", 0)
                txt = getattr(c, "text", "")
                pg = getattr(c, "page_number", None)
                sec_t = getattr(c, "section_title", None)
                sec_p = getattr(c, "section_path", None)

            item = {
                "chunk_id": cid,
                "document_id": doc_id,
                "chunk_index": idx,
                "text": txt,
                "page_number": pg,
                "section_title": sec_t,
                "section_path": sec_p,
            }
            payloads.append(item)
            bm25_chunks.append(item)

        return payloads, bm25_chunks

    def index_document_chunks(
        self,
        document_id: str,
        chunks: List[Any],
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Indexes a list of chunks for a document into both Qdrant and BM25.
        Generates dense embeddings in batches.
        """
        if not chunks:
            return {"dense_indexed": 0, "bm25_indexed": 0}

        t_start = time.perf_counter()
        payloads, bm25_chunks = self._prepare_payloads_and_bm25(chunks)
        chunk_ids = [p["chunk_id"] for p in payloads]
        texts = [p["text"] for p in payloads]

        # 1. Generate Dense Embeddings
        vectors = self.embedding_service.embed_texts(texts)

        # 2. Upsert to Qdrant
        self.vector_service.upsert_chunks(
            chunk_ids=chunk_ids,
            vectors=vectors,
            payloads=payloads,
            collection_name=collection_name,
        )

        # 3. Add to BM25 Lexical Index
        bm25_count = self.bm25_service.add_chunks(bm25_chunks, persist=True)

        elapsed = round(time.perf_counter() - t_start, 3)
        logger.info(
            f"indexing_completed: document_id='{document_id}' chunks={len(chunks)} "
            f"dense={len(vectors)} bm25={bm25_count} elapsed_s={elapsed}"
        )

        return {
            "document_id": document_id,
            "chunks_count": len(chunks),
            "dense_indexed": len(vectors),
            "bm25_indexed": bm25_count,
            "elapsed_seconds": elapsed,
        }

    def remove_document_from_index(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Removes all indexed records for a document from Qdrant and BM25."""
        qdrant_ok = self.vector_service.delete_document_vectors(
            document_id=document_id, collection_name=collection_name
        )
        bm25_removed = self.bm25_service.remove_document(document_id, persist=True)

        logger.info(
            f"index_cleanup_completed: document_id='{document_id}' "
            f"qdrant_deleted={qdrant_ok} bm25_chunks_removed={bm25_removed}"
        )
        return {
            "document_id": document_id,
            "qdrant_deleted": qdrant_ok,
            "bm25_chunks_removed": bm25_removed,
        }

    async def rebuild_all_indexes(
        self,
        db: AsyncSession,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reconstructs both Qdrant collection and BM25 index from scratch
        using persistent DocumentChunk records in PostgreSQL / SQLite.
        """
        t_start = time.perf_counter()
        logger.info("rebuild_all_indexes: starting full index reconstruction...")

        # 1. Reset Indexes
        dim = self.embedding_service.dimension
        self.vector_service.recreate_collection(collection_name=collection_name, vector_size=dim)
        self.bm25_service.clear(persist=True)

        # 2. Fetch all document chunks
        stmt = (
            select(DocumentChunk)
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )
        result = await db.execute(stmt)
        all_chunks = result.scalars().all()

        if not all_chunks:
            return {
                "message": "Index rebuild completed (0 chunks found).",
                "total_documents": 0,
                "total_chunks": 0,
                "dense_indexed": 0,
                "bm25_indexed": 0,
                "elapsed_seconds": round(time.perf_counter() - t_start, 2),
            }

        # 3. Process in batches
        batch_size = self.settings.EMBEDDING_BATCH_SIZE
        dense_count = 0
        bm25_count = 0
        doc_ids = set()

        for i in range(0, len(all_chunks), batch_size):
            chunk_batch = all_chunks[i : i + batch_size]
            payloads, bm25_chunks = self._prepare_payloads_and_bm25(chunk_batch)
            chunk_ids = [p["chunk_id"] for p in payloads]
            texts = [p["text"] for p in payloads]
            for p in payloads:
                doc_ids.add(p["document_id"])

            vectors = self.embedding_service.embed_texts(texts)
            self.vector_service.upsert_chunks(
                chunk_ids=chunk_ids,
                vectors=vectors,
                payloads=payloads,
                collection_name=collection_name,
            )
            dense_count += len(vectors)
            self.bm25_service.add_chunks(bm25_chunks, persist=False)
            bm25_count += len(bm25_chunks)

        self.bm25_service.save_to_disk()
        elapsed = round(time.perf_counter() - t_start, 2)

        logger.info(
            f"rebuild_all_indexes_completed: docs={len(doc_ids)} chunks={len(all_chunks)} "
            f"dense={dense_count} bm25={bm25_count} elapsed_s={elapsed}"
        )

        return {
            "message": "Successfully rebuilt dense vector and BM25 lexical indexes.",
            "total_documents": len(doc_ids),
            "total_chunks": len(all_chunks),
            "dense_indexed": dense_count,
            "bm25_indexed": bm25_count,
            "elapsed_seconds": elapsed,
        }

    async def reindex_document(
        self,
        document_id: str,
        db: AsyncSession,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Re-indexes a single document's chunks from database records."""
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        res = await db.execute(stmt)
        chunks = res.scalars().all()

        if not chunks:
            # Clean up if no chunks exist
            self.remove_document_from_index(document_id, collection_name=collection_name)
            return {
                "document_id": document_id,
                "chunks_indexed": 0,
                "message": f"No chunks found for document '{document_id}'. Any existing index entries removed.",
            }

        self.remove_document_from_index(document_id, collection_name=collection_name)
        result = self.index_document_chunks(document_id, chunks, collection_name=collection_name)
        return {
            "document_id": document_id,
            "chunks_indexed": result["chunks_count"],
            "message": f"Successfully indexed {result['chunks_count']} chunks for document '{document_id}'.",
        }

    def get_index_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Returns consolidated indexing stats across dense Qdrant and lexical BM25."""
        q_stats = self.vector_service.get_collection_stats(collection_name)
        bm_stats = self.bm25_service.get_stats()

        return {
            "qdrant": q_stats,
            "bm25": bm_stats,
            "embedding_provider": self.settings.EMBEDDING_PROVIDER,
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.dimension,
        }


_indexing_service_instance: Optional[IndexingService] = None


def get_indexing_service(settings: Optional[Settings] = None) -> IndexingService:
    """Returns singleton IndexingService instance."""
    global _indexing_service_instance
    if _indexing_service_instance is None:
        _indexing_service_instance = IndexingService(settings=settings)
    return _indexing_service_instance
