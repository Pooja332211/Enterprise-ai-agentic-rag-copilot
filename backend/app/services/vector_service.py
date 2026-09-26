import time
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from backend.app.config.settings import Settings, get_settings
from backend.app.utils.logger import logger


class VectorService:
    """
    Qdrant vector store service abstraction.
    Manages connection, collection lifecycle, and health diagnostics
    with zero coupling between vector implementation and application logic.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client: Optional[QdrantClient] = None
        self._is_fallback_mode: bool = False
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initializes connection to Qdrant with optional local fallback."""
        url = self.settings.resolved_qdrant_url
        api_key = self.settings.QDRANT_API_KEY or None

        # Production / strict remote path
        try:
            client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=self.settings.QDRANT_TIMEOUT_SECONDS,
                check_compatibility=False,
            )
            # Validate connection immediately
            client.get_collections()
            self._client = client
            self._is_fallback_mode = False
            logger.info(f"Connected to remote Qdrant vector database at {url}")
            return
        except Exception as exc:
            if not self.settings.QDRANT_LOCAL_FALLBACK:
                logger.error(f"Failed to connect to Qdrant at {url}: {exc}")
                self._client = None
                return

            logger.warning(
                f"Could not connect to Qdrant at {url} ({exc}). "
                f"Falling back to local in-memory Qdrant client for development/testing."
            )
            try:
                self._client = QdrantClient(":memory:")
                self._is_fallback_mode = True
                logger.info("Initialized local in-memory Qdrant client fallback.")
            except Exception as local_exc:
                logger.error(f"Failed to initialize local Qdrant fallback: {local_exc}")
                self._client = None

    @property
    def client(self) -> QdrantClient:
        """Returns the underlying Qdrant client instance."""
        if self._client is None:
            self._initialize_client()
        if self._client is None:
            raise RuntimeError("Qdrant vector service is unavailable.")
        return self._client

    def is_connected(self) -> bool:
        """Returns True if the vector database is accessible."""
        if self._client is None:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
    ) -> bool:
        """
        Ensures the target vector collection exists.
        Creates it with standard vector parameters if not already present.
        """
        name = collection_name or self.settings.QDRANT_COLLECTION_NAME
        dim = vector_size or self.settings.EMBEDDING_DIMENSION

        try:
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]

            if name not in existing_names:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(
                        size=dim,
                        distance=distance,
                    ),
                )
                logger.info(f"Created Qdrant collection '{name}' with vector size {dim}.")

            # Ensure payload field indexes exist for accelerated filtering
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name="document_id",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name="page_number",
                    field_schema=qmodels.PayloadSchemaType.INTEGER,
                )
            except Exception:
                pass

            return True
        except Exception as exc:
            logger.error(f"Failed to ensure collection '{name}': {exc}")
            return False

    def get_collections(self) -> List[str]:
        """Returns a list of all existing collection names."""
        if not self.is_connected():
            return []
        try:
            res = self.client.get_collections()
            return [c.name for c in res.collections]
        except Exception as exc:
            logger.error(f"Error fetching Qdrant collections: {exc}")
            return []

    def check_health(self) -> Dict[str, Any]:
        """
        Diagnostics probe returning status, collections, latency, and operational mode.
        Guarantees sensitive keys or raw auth strings are never returned.
        """
        start_time = time.perf_counter()

        if self._client is None:
            # Try re-initializing once in case network or container just came up
            self._initialize_client()

        if self._client is None:
            return {
                "status": "unreachable",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "error": "Could not connect to Qdrant service",
                "collections": [],
            }

        try:
            res = self._client.get_collections()
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            collection_names = [c.name for c in res.collections]

            return {
                "status": "connected",
                "latency_ms": latency_ms,
                "collections": collection_names,
                "mode": "in_memory" if self._is_fallback_mode else "remote",
                "fallback": self._is_fallback_mode,
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "status": "unreachable",
                "latency_ms": latency_ms,
                "error": str(exc),
                "collections": [],
            }

    def recreate_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
    ) -> bool:
        """Drops and recreates the target collection."""
        name = collection_name or self.settings.QDRANT_COLLECTION_NAME
        dim = vector_size or self.settings.EMBEDDING_DIMENSION
        try:
            collections = self.get_collections()
            if name in collections:
                self.client.delete_collection(name)
                logger.info(f"Deleted existing Qdrant collection '{name}'.")
            return self.ensure_collection(name, dim, distance)
        except Exception as exc:
            logger.error(f"Failed to recreate collection '{name}': {exc}")
            return False

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        Upserts dense vectors and associated chunk payloads into Qdrant.
        Vector IDs are set directly to the chunk_ids (UUID strings) to guarantee
        1:1 mapping and idempotency during re-indexing.
        """
        if not chunk_ids or not vectors:
            return True

        name = collection_name or self.settings.QDRANT_COLLECTION_NAME
        self.ensure_collection(name, vector_size=len(vectors[0]))

        points = [
            qmodels.PointStruct(
                id=chunk_ids[i],
                vector=vectors[i],
                payload=payloads[i] if i < len(payloads) else {},
            )
            for i in range(len(chunk_ids))
        ]

        try:
            self.client.upsert(collection_name=name, points=points)
            logger.info(f"Upserted {len(points)} vector chunks into Qdrant collection '{name}'.")
            return True
        except Exception as exc:
            logger.error(f"Failed to upsert chunks to Qdrant collection '{name}': {exc}")
            raise

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        collection_name: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches Qdrant for top-K nearest neighbors using cosine similarity.
        Optionally filters results by document_id.
        """
        name = collection_name or self.settings.QDRANT_COLLECTION_NAME
        if not self.is_connected() or name not in self.get_collections():
            return []

        query_filter: Optional[qmodels.Filter] = None
        if document_ids:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchAny(any=document_ids),
                    )
                ]
            )

        try:
            resp = self.client.query_points(
                collection_name=name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )

            results: List[Dict[str, Any]] = []
            for p in resp.points:
                results.append(
                    {
                        "chunk_id": str(p.id),
                        "score": float(p.score),
                        "payload": p.payload or {},
                    }
                )
            return results
        except Exception as exc:
            logger.error(f"Error executing vector search in collection '{name}': {exc}")
            return []

    def delete_document_vectors(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Deletes all vector chunks belonging to a specific document."""
        name = collection_name or self.settings.QDRANT_COLLECTION_NAME
        if not self.is_connected() or name not in self.get_collections():
            return True

        try:
            self.client.delete(
                collection_name=name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Deleted vector chunks for document '{document_id}' from Qdrant collection '{name}'.")
            return True
        except Exception as exc:
            logger.error(f"Failed to delete vectors for document '{document_id}': {exc}")
            return False

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Returns collection metrics such as points count and vector status."""
        name = collection_name or self.settings.QDRANT_COLLECTION_NAME
        if not self.is_connected() or name not in self.get_collections():
            return {"points_count": 0, "status": "not_found", "collection_name": name}

        try:
            info = self.client.get_collection(collection_name=name)
            return {
                "collection_name": name,
                "points_count": info.points_count or 0,
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", info.points_count or 0),
                "status": str(info.status),
                "mode": "in_memory" if self._is_fallback_mode else "remote",
            }
        except Exception as exc:
            logger.error(f"Failed to fetch stats for collection '{name}': {exc}")
            return {"points_count": 0, "status": "error", "error": str(exc), "collection_name": name}

    def close(self) -> None:
        """Closes any open client connections."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            logger.info("Qdrant client connection closed.")


# Singleton instance for dependency injection
_vector_service_instance: Optional[VectorService] = None


def get_vector_service(settings: Optional[Settings] = None) -> VectorService:
    """Returns the cached vector service instance."""
    global _vector_service_instance
    if _vector_service_instance is None:
        _vector_service_instance = VectorService(settings=settings)
    return _vector_service_instance
