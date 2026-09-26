import asyncio
import time
from typing import Any, Dict, List, Literal, Optional
from backend.app.config.settings import Settings, get_settings
from backend.app.services.bm25_service import BM25IndexService, get_bm25_service
from backend.app.services.context_assembly_service import ContextAssemblyService, get_context_assembly_service
from backend.app.services.embedding_service import BaseEmbeddingService, get_embedding_service
from backend.app.services.reranker_service import BaseReranker, get_reranker_service
from backend.app.services.vector_service import VectorService, get_vector_service
from backend.app.utils.logger import logger


class HybridRetrievalService:
    """
    Coordinates dense semantic vector search (Qdrant) and lexical keyword search (BM25),
    fuses candidate result sets using Reciprocal Rank Fusion (RRF) or weighted score blend,
    applies cross-encoder neural reranking, and optionally packs relevant context within budgets.
    """

    def __init__(
        self,
        vector_service: Optional[VectorService] = None,
        bm25_service: Optional[BM25IndexService] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        reranker_service: Optional[BaseReranker] = None,
        context_assembly_service: Optional[ContextAssemblyService] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.vector_service = vector_service or get_vector_service(self.settings)
        self.bm25_service = bm25_service or get_bm25_service(self.settings)
        self.embedding_service = embedding_service or get_embedding_service(self.settings)
        self.reranker_service = reranker_service or get_reranker_service(self.settings)
        self.context_assembly_service = context_assembly_service or get_context_assembly_service(self.settings)

    async def search(
        self,
        query: str,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
        alpha: float = 0.5,
        fusion_method: Literal["rrf", "weighted"] = "rrf",
        candidate_top_k: Optional[int] = None,
        rerank: bool = True,
        min_relevance_threshold: Optional[float] = None,
        context_assembly: bool = False,
        context_max_tokens: Optional[int] = None,
        context_max_characters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes retrieval according to specified search mode, rank fusion strategy,
        optional cross-encoder reranking, and optional context assembly.
        """
        t_start = time.perf_counter()
        breakdown_ms: Dict[str, float] = {}

        if not query or not query.strip():
            return {
                "query": query,
                "mode": mode,
                "fusion_method": fusion_method,
                "total": 0,
                "latency_ms": 0.0,
                "breakdown_ms": breakdown_ms,
                "reranked": False,
                "results": [],
                "assembled_context": None,
            }

        # Determine candidate pool size for first-stage retrieval
        if candidate_top_k is not None and candidate_top_k > 0:
            candidate_k = max(candidate_top_k, top_k)
        elif rerank and self.settings.RERANKER_ENABLED:
            candidate_k = max(top_k * 3, self.settings.RERANKER_CANDIDATE_TOP_K)
        else:
            candidate_k = max(top_k * 3, 20)

        dense_results: List[Dict[str, Any]] = []
        bm25_results: List[Dict[str, Any]] = []

        # 1. Dense Semantic Search
        if mode in ("hybrid", "dense"):
            t_emb_start = time.perf_counter()
            query_vector = self.embedding_service.embed_query(query)
            breakdown_ms["embedding_generation_ms"] = round((time.perf_counter() - t_emb_start) * 1000, 2)

            t_dense_start = time.perf_counter()
            dense_results = self.vector_service.search_vectors(
                query_vector=query_vector,
                top_k=candidate_k,
                document_ids=document_ids,
            )
            breakdown_ms["dense_search_ms"] = round((time.perf_counter() - t_dense_start) * 1000, 2)

        # 2. Lexical BM25 Search
        if mode in ("hybrid", "sparse"):
            t_bm25_start = time.perf_counter()
            bm25_results = self.bm25_service.search(
                query=query,
                top_k=candidate_k,
                document_ids=document_ids,
            )
            breakdown_ms["bm25_search_ms"] = round((time.perf_counter() - t_bm25_start) * 1000, 2)

        # 3. First-Stage Fusion / Mode Execution
        t_fusion_start = time.perf_counter()
        pool_k = candidate_k if (rerank and self.settings.RERANKER_ENABLED) else top_k

        if mode == "dense":
            candidate_results = self._format_dense_only(dense_results, pool_k)
        elif mode == "sparse":
            candidate_results = self._format_sparse_only(bm25_results, pool_k)
        else:  # hybrid
            if fusion_method == "weighted":
                candidate_results = self._weighted_score_fusion(
                    dense_results=dense_results,
                    bm25_results=bm25_results,
                    top_k=pool_k,
                    alpha=alpha,
                )
            else:  # rrf default
                candidate_results = self._reciprocal_rank_fusion(
                    dense_results=dense_results,
                    bm25_results=bm25_results,
                    top_k=pool_k,
                    alpha=alpha,
                    k=self.settings.RETRIEVAL_RRF_K,
                )
        breakdown_ms["fusion_ms"] = round((time.perf_counter() - t_fusion_start) * 1000, 2)

        # Preserve first-stage score
        for item in candidate_results:
            item["retrieval_score"] = item["score"]

        # 4. Cross-Encoder Neural Reranking
        is_reranked = False
        if rerank and self.settings.RERANKER_ENABLED and candidate_results:
            t_rerank_start = time.perf_counter()
            reranked_pool = self.reranker_service.rerank(
                query=query,
                candidates=candidate_results,
                top_k=len(candidate_results),
            )
            breakdown_ms["reranking_ms"] = round((time.perf_counter() - t_rerank_start) * 1000, 2)
            is_reranked = True

            final_pool = []
            for item in reranked_pool:
                r_item = dict(item)
                r_item["score"] = r_item.get("rerank_score", r_item["score"])
                final_pool.append(r_item)
        else:
            final_pool = candidate_results
            for item in final_pool:
                item["rerank_score"] = None

        # 5. Min Relevance Threshold Filtering
        effective_threshold = (
            min_relevance_threshold
            if min_relevance_threshold is not None
            else self.settings.MIN_RELEVANCE_THRESHOLD
        )
        if effective_threshold is not None:
            final_pool = [c for c in final_pool if c["score"] >= effective_threshold]
        elif is_reranked and len(final_pool) > 0 and final_pool[0].get("score", 0.0) >= 0.2:
            cutoff = max(0.02, final_pool[0]["score"] * 0.05)
            filtered = [c for c in final_pool if c.get("score", 0.0) >= cutoff]
            if filtered:
                final_pool = filtered

        # 6. Final Slice and Rank Assignment
        final_results = final_pool[:top_k]
        for rank_idx, item in enumerate(final_results, start=1):
            item["final_rank"] = rank_idx

        # 7. Context Assembly (Budget packing and deduplication)
        assembled_context_data = None
        if context_assembly:
            t_context_start = time.perf_counter()
            assembled_context_data = self.context_assembly_service.assemble_context(
                chunks=final_results,
                max_tokens=context_max_tokens,
                max_characters=context_max_characters,
            )
            breakdown_ms["context_assembly_ms"] = round((time.perf_counter() - t_context_start) * 1000, 2)

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)

        logger.info(
            f"retrieval_completed: mode='{mode}' fusion='{fusion_method}' reranked={is_reranked} "
            f"query='{query[:40]}' results={len(final_results)} total_ms={total_latency}"
        )

        return {
            "query": query,
            "mode": mode,
            "fusion_method": fusion_method,
            "total": len(final_results),
            "latency_ms": total_latency,
            "breakdown_ms": breakdown_ms,
            "reranked": is_reranked,
            "results": final_results,
            "assembled_context": assembled_context_data,
        }

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
        alpha: float,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion:
        RRF_score(d) = (alpha / (k + rank_dense)) + ((1 - alpha) / (k + rank_bm25))
        """
        all_chunks: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        # Collect dense ranks (1-indexed)
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            dense_ranks[cid] = rank
            dense_scores[cid] = item["score"]
            payload = item.get("payload", {})
            if cid not in all_chunks:
                all_chunks[cid] = {
                    "chunk_id": cid,
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": payload.get("text", ""),
                    "page_number": payload.get("page_number"),
                    "section_title": payload.get("section_title"),
                    "section_path": payload.get("section_path"),
                }

        # Collect BM25 ranks (1-indexed)
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            bm25_ranks[cid] = rank
            bm25_scores[cid] = item["score"]
            if cid not in all_chunks:
                all_chunks[cid] = {
                    "chunk_id": cid,
                    "document_id": item.get("document_id"),
                    "chunk_index": item.get("chunk_index", 0),
                    "text": item.get("text", ""),
                    "page_number": item.get("page_number"),
                    "section_title": item.get("section_title"),
                    "section_path": item.get("section_path"),
                }

        # Compute RRF score for every candidate
        fused: List[Dict[str, Any]] = []
        w_dense = alpha
        w_bm25 = 1.0 - alpha

        for cid, data in all_chunks.items():
            r_dense = dense_ranks.get(cid)
            r_bm25 = bm25_ranks.get(cid)

            rrf_score = 0.0
            sources: List[str] = []

            if r_dense is not None:
                rrf_score += w_dense / (k + r_dense)
                sources.append("dense")
            if r_bm25 is not None:
                rrf_score += w_bm25 / (k + r_bm25)
                sources.append("bm25")

            fused.append(
                {
                    **data,
                    "score": round(rrf_score, 6),
                    "dense_score": dense_scores.get(cid),
                    "bm25_score": bm25_scores.get(cid),
                    "dense_rank": r_dense,
                    "bm25_rank": r_bm25,
                    "retrieval_sources": sources,
                }
            )

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

    def _weighted_score_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
        alpha: float,
    ) -> List[Dict[str, Any]]:
        """
        Min-max normalized score fusion:
        Score(d) = alpha * norm(dense) + (1 - alpha) * norm(bm25)
        """
        all_chunks: Dict[str, Dict[str, Any]] = {}
        dense_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            dense_scores[cid] = item["score"]
            dense_ranks[cid] = rank
            payload = item.get("payload", {})
            if cid not in all_chunks:
                all_chunks[cid] = {
                    "chunk_id": cid,
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": payload.get("text", ""),
                    "page_number": payload.get("page_number"),
                    "section_title": payload.get("section_title"),
                    "section_path": payload.get("section_path"),
                }

        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            bm25_scores[cid] = item["score"]
            bm25_ranks[cid] = rank
            if cid not in all_chunks:
                all_chunks[cid] = {
                    "chunk_id": cid,
                    "document_id": item.get("document_id"),
                    "chunk_index": item.get("chunk_index", 0),
                    "text": item.get("text", ""),
                    "page_number": item.get("page_number"),
                    "section_title": item.get("section_title"),
                    "section_path": item.get("section_path"),
                }

        # Min-max normalization helper
        def normalize_map(scores_dict: Dict[str, float]) -> Dict[str, float]:
            if not scores_dict:
                return {}
            vals = list(scores_dict.values())
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return {k: 1.0 for k in scores_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

        norm_dense = normalize_map(dense_scores)
        norm_bm25 = normalize_map(bm25_scores)

        fused: List[Dict[str, Any]] = []
        for cid, data in all_chunks.items():
            s_dense = norm_dense.get(cid, 0.0)
            s_bm25 = norm_bm25.get(cid, 0.0)
            combined = alpha * s_dense + (1.0 - alpha) * s_bm25

            sources: List[str] = []
            if cid in dense_scores:
                sources.append("dense")
            if cid in bm25_scores:
                sources.append("bm25")

            fused.append(
                {
                    **data,
                    "score": round(combined, 6),
                    "dense_score": dense_scores.get(cid),
                    "bm25_score": bm25_scores.get(cid),
                    "dense_rank": dense_ranks.get(cid),
                    "bm25_rank": bm25_ranks.get(cid),
                    "retrieval_sources": sources,
                }
            )

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

    def _format_dense_only(self, dense_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for rank, item in enumerate(dense_results[:top_k], start=1):
            payload = item.get("payload", {})
            formatted.append(
                {
                    "chunk_id": item["chunk_id"],
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": payload.get("text", ""),
                    "score": round(item["score"], 6),
                    "dense_score": item["score"],
                    "bm25_score": None,
                    "dense_rank": rank,
                    "bm25_rank": None,
                    "page_number": payload.get("page_number"),
                    "section_title": payload.get("section_title"),
                    "section_path": payload.get("section_path"),
                    "retrieval_sources": ["dense"],
                }
            )
        return formatted

    def _format_sparse_only(self, bm25_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for rank, item in enumerate(bm25_results[:top_k], start=1):
            formatted.append(
                {
                    "chunk_id": item["chunk_id"],
                    "document_id": item.get("document_id"),
                    "chunk_index": item.get("chunk_index", 0),
                    "text": item.get("text", ""),
                    "score": round(item["score"], 6),
                    "dense_score": None,
                    "bm25_score": item["score"],
                    "dense_rank": None,
                    "bm25_rank": rank,
                    "page_number": item.get("page_number"),
                    "section_title": item.get("section_title"),
                    "section_path": item.get("section_path"),
                    "retrieval_sources": ["bm25"],
                }
            )
        return formatted


_hybrid_service_instance: Optional[HybridRetrievalService] = None


def get_hybrid_retrieval_service(settings: Optional[Settings] = None) -> HybridRetrievalService:
    """Returns singleton HybridRetrievalService instance."""
    global _hybrid_service_instance
    if _hybrid_service_instance is None:
        _hybrid_service_instance = HybridRetrievalService(settings=settings)
    return _hybrid_service_instance


def reset_hybrid_retrieval_service() -> None:
    """Resets the singleton instance (useful for testing)."""
    global _hybrid_service_instance
    _hybrid_service_instance = None

