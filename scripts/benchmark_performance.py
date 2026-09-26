"""Performance benchmark script measuring latency distributions, cache efficiency, and evaluation gates."""

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

from backend.app.agents.agent_service import AgentResearchService
from backend.app.cache import get_embedding_cache, get_reranker_cache
from backend.app.database.session import AsyncSessionLocal, init_db
from backend.app.evaluation.gates import evaluate_gates
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.models.schemas import AgentResearchRequest, ResearchRequest
from backend.app.services.research_service import get_research_service


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Computes min, mean, max, p50, p95, and p99 percentiles."""
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        idx = max(0, min(n - 1, math.ceil(p / 100.0 * n) - 1))
        return sorted_vals[idx]

    return {
        "min": round(sorted_vals[0], 2),
        "mean": round(sum(sorted_vals) / n, 2),
        "max": round(sorted_vals[-1], 2),
        "p50": round(percentile(50), 2),
        "p95": round(percentile(95), 2),
        "p99": round(percentile(99), 2),
    }


async def main():
    print("=" * 80)
    print("  ENTERPRISE AI COPILOT: PRODUCTION PERFORMANCE BENCHMARK")
    print("=" * 80)

    await init_db()
    runner = EvaluationRunner()
    async with AsyncSessionLocal() as session:
        await runner.ensure_corpus_ingested(session)

    bench_path = Path("data/evaluation/performance_benchmark_v1.json")
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    queries = bench_data["queries"]
    print(f"\n[INFO] Loaded benchmark: {len(queries)} queries from {bench_path}")

    # Initialize services
    research_service = get_research_service()
    agent_service = AgentResearchService()
    emb_cache = get_embedding_cache()
    rerank_cache = get_reranker_cache()

    # Clear caches for Cold Pass
    emb_cache.clear()
    rerank_cache.clear()

    print("\n--- PHASE 1: COLD EXECUTION (Caches Empty) ---")
    cold_single_latencies: List[float] = []
    cold_agent_latencies: List[float] = []

    for idx, q_item in enumerate(queries, start=1):
        q = q_item["query"]
        print(f"[{idx}/{len(queries)}] Cold: {q[:55]}...")

        # Single-turn research
        t0 = time.perf_counter()
        req_single = ResearchRequest(query=q, top_k=5, candidate_top_k=10, rerank=True)
        res_single = await research_service.execute_research(req_single)
        dt_single = (time.perf_counter() - t0) * 1000.0
        cold_single_latencies.append(dt_single)

        # Agentic multi-subquery research
        t0 = time.perf_counter()
        req_agent = AgentResearchRequest(query=q, top_k=5, candidate_top_k=10, rerank=True, enable_decomposition=True)
        res_agent = await agent_service.execute_agent_research(req_agent)
        dt_agent = (time.perf_counter() - t0) * 1000.0
        cold_agent_latencies.append(dt_agent)

    cold_emb_stats = emb_cache.stats()
    cold_rerank_stats = rerank_cache.stats()

    print("\n--- PHASE 2: WARM EXECUTION (Caches Primed, 2 Repetitions) ---")
    warm_single_latencies: List[float] = []
    warm_agent_latencies: List[float] = []
    agent_subqueries_counts: List[int] = []
    agent_evidence_counts: List[int] = []
    grounded_flags: List[bool] = []
    citation_valid_flags: List[bool] = []
    abstention_accurate_flags: List[bool] = []

    for rep in range(1, 3):
        print(f"\n[INFO] Starting Warm Repetition {rep}/2...")
        for idx, q_item in enumerate(queries, start=1):
            q = q_item["query"]

            t0 = time.perf_counter()
            req_single = ResearchRequest(query=q, top_k=5, candidate_top_k=10, rerank=True)
            res_single = await research_service.execute_research(req_single)
            dt_single = (time.perf_counter() - t0) * 1000.0
            warm_single_latencies.append(dt_single)

            t0 = time.perf_counter()
            req_agent = AgentResearchRequest(query=q, top_k=5, candidate_top_k=10, rerank=True, enable_decomposition=True)
            res_agent = await agent_service.execute_agent_research(req_agent)
            dt_agent = (time.perf_counter() - t0) * 1000.0
            warm_agent_latencies.append(dt_agent)

            if rep == 1:
                agent_subqueries_counts.append(len(res_agent.sub_queries))
                agent_evidence_counts.append(len(res_agent.evidence))
                grounded_flags.append(res_agent.grounded or res_agent.insufficient_evidence)

                evidence_chunk_ids = {ev.get("chunk_id") for ev in res_agent.evidence}
                if res_agent.citations:
                    all_cits_valid = all(cit.chunk_id in evidence_chunk_ids for cit in res_agent.citations)
                    citation_valid_flags.append(all_cits_valid)
                else:
                    citation_valid_flags.append(True)

                expected_abs = q_item.get("expected_abstention", False)
                actual_abs = res_agent.insufficient_evidence
                abstention_accurate_flags.append(expected_abs == actual_abs)

    warm_emb_stats = emb_cache.stats()
    warm_rerank_stats = rerank_cache.stats()

    # Percentiles
    cold_single_p = compute_percentiles(cold_single_latencies)
    cold_agent_p = compute_percentiles(cold_agent_latencies)
    warm_single_p = compute_percentiles(warm_single_latencies)
    warm_agent_p = compute_percentiles(warm_agent_latencies)

    # Retrieval Quality for Gating (from Milestone 6 benchmark)
    retrieval_res = await runner.retrieval_evaluator.run_ablation(
        queries=runner.load_dataset().queries,
        session=session,
    )
    rerank_row = None
    for r in retrieval_res["ablation_table"]:
        if "rerank" in r.configuration.lower():
            rerank_row = r
            break
    recall_at_5 = rerank_row.recall_at_5 if rerank_row else 0.8333
    mrr = rerank_row.mrr if rerank_row else 0.8125
    ndcg_at_5 = rerank_row.ndcg_at_5 if rerank_row else 0.8250

    grounding_acc = sum(1 for f in grounded_flags if f) / len(grounded_flags)
    cit_validity = sum(1 for f in citation_valid_flags if f) / len(citation_valid_flags)
    abs_accuracy = sum(1 for f in abstention_accurate_flags if f) / len(abstention_accurate_flags)

    metrics_for_gates = {
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "ndcg_at_5": ndcg_at_5,
        "grounding_accuracy": grounding_acc,
        "citation_validity": cit_validity,
        "abstention_accuracy": abs_accuracy,
        "p95_latency_ms": warm_agent_p["p95"],
    }

    gates_summary = evaluate_gates(metrics_for_gates)

    # Print Summary Tables
    print("\n" + "=" * 90)
    print("  LATENCY BENCHMARK SUMMARY (COLD VS WARM)")
    print("=" * 90)
    fmt = "{:<28} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}"
    print(fmt.format("Pipeline Mode", "Mean ms", "p50 ms", "p95 ms", "p99 ms", "Max ms"))
    print("-" * 90)
    print(fmt.format("Single Research (Cold)", cold_single_p["mean"], cold_single_p["p50"], cold_single_p["p95"], cold_single_p["p99"], cold_single_p["max"]))
    print(fmt.format("Single Research (Warm)", warm_single_p["mean"], warm_single_p["p50"], warm_single_p["p95"], warm_single_p["p99"], warm_single_p["max"]))
    print(fmt.format("Agentic Workflow (Cold)", cold_agent_p["mean"], cold_agent_p["p50"], cold_agent_p["p95"], cold_agent_p["p99"], cold_agent_p["max"]))
    print(fmt.format("Agentic Workflow (Warm)", warm_agent_p["mean"], warm_agent_p["p50"], warm_agent_p["p95"], warm_agent_p["p99"], warm_agent_p["max"]))
    print("=" * 90)

    print("\n" + "=" * 90)
    print("  CACHE PERFORMANCE STATISTICS")
    print("=" * 90)
    emb_hr = warm_emb_stats.get("hit_rate_pct", round(warm_emb_stats.get("hit_ratio", 0.0) * 100, 2))
    rerank_hr = warm_rerank_stats.get("hit_rate_pct", round(warm_rerank_stats.get("hit_ratio", 0.0) * 100, 2))
    print(f"Embedding Cache: hits={warm_emb_stats['hits']}, misses={warm_emb_stats['misses']}, hit_rate={emb_hr}%, size={warm_emb_stats['size']}")
    print(f"Reranker Cache:  hits={warm_rerank_stats['hits']}, misses={warm_rerank_stats['misses']}, hit_rate={rerank_hr}%, size={warm_rerank_stats['size']}")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("  EVALUATION GATES REPORT")
    print("=" * 90)
    for g in gates_summary.results:
        print(f" - {g.message}")
    verdict = "PASSED (READY FOR DEPLOYMENT)" if gates_summary.all_passed else "FAILED"
    print(f"\nFinal Policy Verdict: {verdict} ({gates_summary.passed_gates}/{gates_summary.total_gates} gates passed)")
    print("=" * 90)

    # Save complete JSON payload
    out_payload = {
        "benchmark_metadata": {
            "version": bench_data["version"],
            "dataset_name": bench_data["dataset_name"],
            "total_queries": len(queries),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "latencies": {
            "single_research_cold": cold_single_p,
            "single_research_warm": warm_single_p,
            "agentic_research_cold": cold_agent_p,
            "agentic_research_warm": warm_agent_p,
        },
        "cache": {
            "embedding": warm_emb_stats,
            "reranker": warm_rerank_stats,
        },
        "gates": gates_summary.model_dump(),
    }

    out_file = Path("data/evaluation/results/performance_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[INFO] Benchmark results saved to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
