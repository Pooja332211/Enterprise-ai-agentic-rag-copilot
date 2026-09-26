import time
from typing import List, Dict, Set, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.database.models import Document, DocumentChunk
from backend.app.services.hybrid_retrieval_service import HybridRetrievalService, get_hybrid_retrieval_service
from backend.app.evaluation.schemas import (
    BenchmarkQuery,
    RetrievalMetrics,
    ConfigurationAblation,
)
from backend.app.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
)


class RetrievalEvaluator:
    """
    Evaluates and compares retrieval ranking performance across 4 configurations:
    1. Dense Only
    2. BM25 Only
    3. Hybrid RRF
    4. Hybrid + Cross-Encoder Reranking
    """

    def __init__(self, retrieval_service: Optional[HybridRetrievalService] = None):
        self.retrieval_service = retrieval_service or get_hybrid_retrieval_service()

    async def build_ground_truth_map(
        self, session: AsyncSession, queries: List[BenchmarkQuery]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Builds a map from query_id to relevant chunk_ids and document_ids
        based on ingested documents in the database.
        """
        docs_res = await session.execute(select(Document))
        docs = docs_res.scalars().all()
        doc_by_filename = {d.filename: d for d in docs}

        chunks_res = await session.execute(select(DocumentChunk))
        all_chunks = chunks_res.scalars().all()

        query_ground_truth = {}
        for q in queries:
            if q.expected_abstention:
                query_ground_truth[q.query_id] = {
                    "relevant_doc_ids": set(),
                    "relevant_chunk_ids": set(),
                }
                continue

            target_doc_ids = set()
            for fn in q.relevant_document_filenames:
                if fn in doc_by_filename:
                    target_doc_ids.add(str(doc_by_filename[fn].id))

            relevant_chunk_ids = set()
            for chunk in all_chunks:
                chunk_doc_id = str(chunk.document_id)
                if chunk_doc_id in target_doc_ids:
                    # If specific section paths are requested, check section match
                    if q.relevant_section_paths:
                        c_sec = (chunk.section_path or "").lower()
                        c_title = (chunk.section_title or "").lower()
                        matches_section = any(
                            sp.lower() in c_sec or sp.lower() in c_title
                            for sp in q.relevant_section_paths
                        )
                        if matches_section:
                            relevant_chunk_ids.add(str(chunk.id))
                    else:
                        relevant_chunk_ids.add(str(chunk.id))

            # If section filtering yielded no chunks (e.g. layout variation),
            # fall back to all chunks of the relevant document(s)
            if not relevant_chunk_ids and target_doc_ids:
                for chunk in all_chunks:
                    if str(chunk.document_id) in target_doc_ids:
                        relevant_chunk_ids.add(str(chunk.id))

            query_ground_truth[q.query_id] = {
                "relevant_doc_ids": target_doc_ids,
                "relevant_chunk_ids": relevant_chunk_ids,
            }

        return query_ground_truth

    async def evaluate_configuration(
        self,
        config_name: str,
        mode: str,
        rerank: bool,
        fusion_method: str,
        candidate_top_k: int,
        queries: List[BenchmarkQuery],
        ground_truth_map: Dict[str, Dict[str, Any]],
        top_k: int = 10,
    ) -> RetrievalMetrics:
        """
        Executes all queries under a specific retrieval configuration and computes metrics.
        """
        eval_queries = [q for q in queries if not q.expected_abstention]
        if not eval_queries:
            return RetrievalMetrics(
                recall_at_1=0.0,
                recall_at_3=0.0,
                recall_at_5=0.0,
                recall_at_10=0.0,
                precision_at_1=0.0,
                precision_at_3=0.0,
                precision_at_5=0.0,
                mrr=0.0,
                ndcg_at_5=0.0,
                ndcg_at_10=0.0,
                mean_latency_ms=0.0,
            )

        r1_list, r3_list, r5_list, r10_list = [], [], [], []
        p1_list, p3_list, p5_list = [], [], []
        mrr_list = []
        ndcg5_list, ndcg10_list = [], []
        latencies = []

        for q in eval_queries:
            gt = ground_truth_map.get(q.query_id, {})
            relevant_chunks = gt.get("relevant_chunk_ids", set())

            start_t = time.perf_counter()
            response = await self.retrieval_service.search(
                query=q.query,
                mode=mode,
                fusion_method=fusion_method,
                candidate_top_k=candidate_top_k,
                rerank=rerank,
                top_k=top_k,
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)

            results_list = response.get("results", [])
            retrieved_chunk_ids = [str(r.get("chunk_id")) for r in results_list]
            effective_relevant = relevant_chunks if relevant_chunks else set()

            r1_list.append(calculate_recall_at_k(retrieved_chunk_ids, effective_relevant, k=1))
            r3_list.append(calculate_recall_at_k(retrieved_chunk_ids, effective_relevant, k=3))
            r5_list.append(calculate_recall_at_k(retrieved_chunk_ids, effective_relevant, k=5))
            r10_list.append(calculate_recall_at_k(retrieved_chunk_ids, effective_relevant, k=10))

            p1_list.append(calculate_precision_at_k(retrieved_chunk_ids, effective_relevant, k=1))
            p3_list.append(calculate_precision_at_k(retrieved_chunk_ids, effective_relevant, k=3))
            p5_list.append(calculate_precision_at_k(retrieved_chunk_ids, effective_relevant, k=5))

            mrr_list.append(calculate_mrr(retrieved_chunk_ids, effective_relevant, k=top_k))
            ndcg5_list.append(calculate_ndcg_at_k(retrieved_chunk_ids, effective_relevant, k=5))
            ndcg10_list.append(calculate_ndcg_at_k(retrieved_chunk_ids, effective_relevant, k=10))

        n = float(len(eval_queries))
        return RetrievalMetrics(
            recall_at_1=round(sum(r1_list) / n, 4),
            recall_at_3=round(sum(r3_list) / n, 4),
            recall_at_5=round(sum(r5_list) / n, 4),
            recall_at_10=round(sum(r10_list) / n, 4),
            precision_at_1=round(sum(p1_list) / n, 4),
            precision_at_3=round(sum(p3_list) / n, 4),
            precision_at_5=round(sum(p5_list) / n, 4),
            mrr=round(sum(mrr_list) / n, 4),
            ndcg_at_5=round(sum(ndcg5_list) / n, 4),
            ndcg_at_10=round(sum(ndcg10_list) / n, 4),
            mean_latency_ms=round(sum(latencies) / float(len(latencies)), 2),
        )

    async def run_ablation(
        self,
        queries: List[BenchmarkQuery],
        session: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Executes the 4-way retrieval ablation:
        1. Dense Only
        2. BM25 Only
        3. Hybrid RRF
        4. Hybrid + Cross-Encoder Reranker
        """
        gt_map = await self.build_ground_truth_map(session, queries)

        # 1. Dense Only
        dense_metrics = await self.evaluate_configuration(
            config_name="Dense",
            mode="dense",
            rerank=False,
            fusion_method="rrf",
            candidate_top_k=20,
            queries=queries,
            ground_truth_map=gt_map,
        )

        # 2. BM25 Only
        bm25_metrics = await self.evaluate_configuration(
            config_name="BM25",
            mode="sparse",
            rerank=False,
            fusion_method="rrf",
            candidate_top_k=20,
            queries=queries,
            ground_truth_map=gt_map,
        )

        # 3. Hybrid RRF
        hybrid_metrics = await self.evaluate_configuration(
            config_name="Hybrid RRF",
            mode="hybrid",
            rerank=False,
            fusion_method="rrf",
            candidate_top_k=20,
            queries=queries,
            ground_truth_map=gt_map,
        )

        # 4. Hybrid + Cross-Encoder Reranker
        rerank_metrics = await self.evaluate_configuration(
            config_name="Hybrid + Reranker",
            mode="hybrid",
            rerank=True,
            fusion_method="rrf",
            candidate_top_k=20,
            queries=queries,
            ground_truth_map=gt_map,
        )

        ablation_rows = [
            ConfigurationAblation(
                configuration="Dense",
                recall_at_5=dense_metrics.recall_at_5,
                precision_at_5=dense_metrics.precision_at_5,
                mrr=dense_metrics.mrr,
                ndcg_at_5=dense_metrics.ndcg_at_5,
                latency_ms=dense_metrics.mean_latency_ms,
            ),
            ConfigurationAblation(
                configuration="BM25",
                recall_at_5=bm25_metrics.recall_at_5,
                precision_at_5=bm25_metrics.precision_at_5,
                mrr=bm25_metrics.mrr,
                ndcg_at_5=bm25_metrics.ndcg_at_5,
                latency_ms=bm25_metrics.mean_latency_ms,
            ),
            ConfigurationAblation(
                configuration="Hybrid RRF",
                recall_at_5=hybrid_metrics.recall_at_5,
                precision_at_5=hybrid_metrics.precision_at_5,
                mrr=hybrid_metrics.mrr,
                ndcg_at_5=hybrid_metrics.ndcg_at_5,
                latency_ms=hybrid_metrics.mean_latency_ms,
            ),
            ConfigurationAblation(
                configuration="Hybrid + Reranker",
                recall_at_5=rerank_metrics.recall_at_5,
                precision_at_5=rerank_metrics.precision_at_5,
                mrr=rerank_metrics.mrr,
                ndcg_at_5=rerank_metrics.ndcg_at_5,
                latency_ms=rerank_metrics.mean_latency_ms,
            ),
        ]

        return {
            "configurations": {
                "dense": dense_metrics,
                "bm25": bm25_metrics,
                "hybrid_rrf": hybrid_metrics,
                "hybrid_reranker": rerank_metrics,
            },
            "ablation_table": ablation_rows,
        }
