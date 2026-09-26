"""Empirical investigation of cross-encoder reranker quality versus latency trade-offs."""

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.database.session import AsyncSessionLocal, init_db
from backend.app.evaluation.metrics import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service
from backend.app.services.reranker_service import get_reranker_service


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Computes mean, p50, p95, and p99 percentiles."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        idx = max(0, min(n - 1, math.ceil(p / 100.0 * n) - 1))
        return sorted_vals[idx]

    return {
        "mean": round(sum(sorted_vals) / n, 2),
        "p50": round(percentile(50), 2),
        "p95": round(percentile(95), 2),
        "p99": round(percentile(99), 2),
    }


async def main():
    print("=" * 80)
    print("  EMPIRICAL CROSS-ENCODER RERANKER BOTTLENECK INVESTIGATION")
    print("=" * 80)

    await init_db()
    runner = EvaluationRunner()
    dataset = runner.load_dataset()
    eval_queries = [q for q in dataset.queries if not q.expected_abstention]

    print(f"\n[INFO] Loaded benchmark: {len(dataset.queries)} total, {len(eval_queries)} evaluation queries.")

    async with AsyncSessionLocal() as session:
        await runner.ensure_corpus_ingested(session)
        gt_map = await runner.retrieval_evaluator.build_ground_truth_map(session, eval_queries)

    retrieval_service = get_hybrid_retrieval_service()
    reranker_service = get_reranker_service()
    reranker_service.warmup()

    # Define ablation matrix
    # K_cand in {5, 10, 15, 20}, K_final in {3, 5, 10}
    test_configs = [
        # Baseline: Reranker OFF (pure Hybrid RRF)
        {"name": "Hybrid RRF (No Rerank)", "rerank": False, "cand_k": 20, "top_k": 5},
        {"name": "Hybrid RRF (No Rerank)", "rerank": False, "cand_k": 20, "top_k": 10},
        # Candidate pool ablation with top_k=5
        {"name": "BGE Reranker (K_cand=5)", "rerank": True, "cand_k": 5, "top_k": 5},
        {"name": "BGE Reranker (K_cand=10)", "rerank": True, "cand_k": 10, "top_k": 5},
        {"name": "BGE Reranker (K_cand=15)", "rerank": True, "cand_k": 15, "top_k": 5},
        {"name": "BGE Reranker (K_cand=20)", "rerank": True, "cand_k": 20, "top_k": 5},
        # Final top_k ablation with K_cand=10
        {"name": "BGE Reranker (K_cand=10, K_final=3)", "rerank": True, "cand_k": 10, "top_k": 3},
        {"name": "BGE Reranker (K_cand=10, K_final=10)", "rerank": True, "cand_k": 10, "top_k": 10},
    ]

    all_results = []

    print("\n[INFO] Running reranker empirical matrix...")
    for cfg in test_configs:
        name = cfg["name"]
        rerank = cfg["rerank"]
        cand_k = cfg["cand_k"]
        top_k = cfg["top_k"]

        r1_list, r3_list, r5_list, r10_list = [], [], [], []
        mrr_list, ndcg5_list, ndcg10_list = [], [], []
        latencies_ms = []

        for q in eval_queries:
            gt = gt_map.get(q.query_id, {})
            rel_chunks = gt.get("relevant_chunk_ids", set())

            t0 = time.perf_counter()
            resp = await retrieval_service.search(
                query=q.query,
                mode="hybrid",
                fusion_method="rrf",
                candidate_top_k=cand_k,
                rerank=rerank,
                top_k=top_k,
                context_assembly=False,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat)

            retrieved_cids = [str(r.get("chunk_id")) for r in resp.get("results", [])]
            r1_list.append(calculate_recall_at_k(retrieved_cids, rel_chunks, k=1))
            r3_list.append(calculate_recall_at_k(retrieved_cids, rel_chunks, k=3))
            r5_list.append(calculate_recall_at_k(retrieved_cids, rel_chunks, k=5))
            r10_list.append(calculate_recall_at_k(retrieved_cids, rel_chunks, k=10))

            mrr_list.append(calculate_mrr(retrieved_cids, rel_chunks, k=top_k))
            ndcg5_list.append(calculate_ndcg_at_k(retrieved_cids, rel_chunks, k=5))
            ndcg10_list.append(calculate_ndcg_at_k(retrieved_cids, rel_chunks, k=10))

        perc = compute_percentiles(latencies_ms)
        cfg_summary = {
            "configuration": name,
            "rerank_enabled": rerank,
            "candidate_k": cand_k,
            "final_top_k": top_k,
            "recall_at_1": round(sum(r1_list) / len(r1_list), 4),
            "recall_at_3": round(sum(r3_list) / len(r3_list), 4),
            "recall_at_5": round(sum(r5_list) / len(r5_list), 4),
            "recall_at_10": round(sum(r10_list) / len(r10_list), 4),
            "mrr": round(sum(mrr_list) / len(mrr_list), 4),
            "ndcg_at_5": round(sum(ndcg5_list) / len(ndcg5_list), 4),
            "ndcg_at_10": round(sum(ndcg10_list) / len(ndcg10_list), 4),
            "latencies_ms": perc,
        }
        all_results.append(cfg_summary)

    # Print markdown table
    print("\n" + "=" * 105)
    print("  EMPIRICAL RERANKER INVESTIGATION RESULTS TABLE")
    print("=" * 105)
    header = (
        f"{'Configuration':<34} | {'R@1':<6} | {'R@5':<6} | {'MRR':<6} | "
        f"{'NDCG@5':<7} | {'Mean (ms)':<10} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'p99 (ms)':<9}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        p = r["latencies_ms"]
        print(
            f"{r['configuration']:<34} | "
            f"{r['recall_at_1']:<6.4f} | "
            f"{r['recall_at_5']:<6.4f} | "
            f"{r['mrr']:<6.4f} | "
            f"{r['ndcg_at_5']:<7.4f} | "
            f"{p['mean']:<10.2f} | "
            f"{p['p50']:<9.2f} | "
            f"{p['p95']:<9.2f} | "
            f"{p['p99']:<9.2f}"
        )
    print("=" * 105)

    # Save to disk
    out_dir = Path("data/evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "reranker_investigation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Investigation results saved to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
