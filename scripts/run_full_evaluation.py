import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.evaluation.runner import EvaluationRunner


async def main():
    print("=" * 85)
    print("  ENTERPRISE AI COPILOT: MILESTONE 6 COMPREHENSIVE BENCHMARK RUNNER")
    print("  Retrieval Ablation, Latency Profiling, Generation, Citations & Abstention")
    print("=" * 85)

    runner = EvaluationRunner()
    summary = await runner.run_evaluation()

    print("\n[OK] Benchmark execution completed successfully.")
    print(f"     Dataset: {summary.dataset_version} | Timestamp: {summary.timestamp}")
    print(f"     Environment: {summary.environment['os']} {summary.environment['machine']} | Python {summary.environment['python_version']}")
    print(f"     Models: Embedding={summary.environment['embedding_model']} | Reranker={summary.environment['reranker_model']} | LLM={summary.environment['llm_model']}")

    # 1. Retrieval Ablation Table
    print("\n" + "=" * 85)
    print("  1. RETRIEVAL ABLATION TABLE (K=5)")
    print("=" * 85)
    header = f"{'Configuration':<22} | {'Recall@5':<10} | {'Precision@5':<12} | {'MRR':<8} | {'NDCG@5':<8} | {'Mean Latency':<12}"
    print(header)
    print("-" * len(header))
    for row in summary.ablation:
        print(f"{row.configuration:<22} | {row.recall_at_5:<10.4f} | {row.precision_at_5:<12.4f} | {row.mrr:<8.4f} | {row.ndcg_at_5:<8.4f} | {row.latency_ms:<9.2f} ms")

    # 2. Latency Distribution Table
    print("\n" + "=" * 85)
    print("  2. MULTI-STAGE LATENCY PROFILING (ms) [Mean / Median / P95]")
    print("=" * 85)
    l_header = f"{'Pipeline Stage':<28} | {'Mean (ms)':<10} | {'Median (ms)':<12} | {'P95 (ms)':<10} | {'Min (ms)':<9} | {'Max (ms)':<9}"
    print(l_header)
    print("-" * len(l_header))
    stages = [
        ("Embedding Generation", summary.latency_profile.embedding_latency),
        ("Dense Search (Qdrant)", summary.latency_profile.dense_search_latency),
        ("BM25 Search", summary.latency_profile.bm25_search_latency),
        ("RRF Fusion", summary.latency_profile.fusion_latency),
        ("Cross-Encoder Reranking", summary.latency_profile.reranking_latency),
        ("Context Assembly", summary.latency_profile.context_assembly_latency),
        ("LLM Generation", summary.latency_profile.generation_latency),
        ("Citation Verification", summary.latency_profile.citation_verification_latency),
        ("Total Pipeline Latency", summary.latency_profile.total_pipeline_latency),
    ]
    for name, d in stages:
        print(f"{name:<28} | {d.mean_ms:<10.2f} | {d.median_ms:<12.2f} | {d.p95_ms:<10.2f} | {d.min_ms:<9.2f} | {d.max_ms:<9.2f}")

    # 3. Generation & Grounding Quality
    print("\n" + "=" * 85)
    print("  3. GENERATION & EVIDENCE GROUNDING QUALITY")
    print("=" * 85)
    print(f"  Evaluated Answered Queries:       {summary.generation_metrics.total_answered_queries}")
    print(f"  Mean Answer Completeness:         {summary.generation_metrics.mean_answer_completeness:.4f}")
    print(f"  Mean Evidence Grounding Score:    {summary.generation_metrics.mean_evidence_grounding_score:.4f}")
    print(f"  Mean Reference Overlap (Token F1):{summary.generation_metrics.mean_reference_token_overlap:.4f}")

    # 4. Citation Verification Metrics
    print("\n" + "=" * 85)
    print("  4. CITATION AUDIT & ACCURACY METRICS")
    print("=" * 85)
    print(f"  Total Claims Evaluated:           {summary.citation_metrics.total_claims}")
    print(f"  Total Citations Generated:        {summary.citation_metrics.total_citations}")
    print(f"  Citation Validity Rate:           {summary.citation_metrics.citation_validity_rate * 100:.2f}% (Chunk ID verified in evidence)")
    print(f"  Citation Coverage Rate:           {summary.citation_metrics.citation_coverage_rate * 100:.2f}% (Claims with citation)")
    print(f"  Citation Support Rate:            {summary.citation_metrics.citation_support_rate * 100:.2f}% (Supported claims)")
    print(f"  Partially Supported Rate:         {summary.citation_metrics.partially_supported_rate * 100:.2f}%")
    print(f"  Unsupported Claim Rate:           {summary.citation_metrics.unsupported_claim_rate * 100:.2f}%")

    # 5. Abstention & Hallucination Resistance
    print("\n" + "=" * 85)
    print("  5. INSUFFICIENT-EVIDENCE & ABSTENTION ACCURACY")
    print("=" * 85)
    print(f"  No-Answer Queries Tested:         {summary.abstention_metrics.total_no_answer_queries}")
    print(f"  Correct Abstentions:              {summary.abstention_metrics.correct_abstentions}")
    print(f"  Incorrect Answers (Hallucinated): {summary.abstention_metrics.incorrect_answers}")
    print(f"  Abstention Accuracy:              {summary.abstention_metrics.abstention_accuracy * 100:.2f}%")

    # 6. Artifact paths
    print("\n" + "=" * 85)
    print("  MACHINE-READABLE OUTPUT ARTIFACTS GENERATED")
    print("=" * 85)
    print(f"  [JSON] data/evaluation/results/benchmark_summary.json")
    print(f"  [CSV]  data/evaluation/results/ablation_table.csv")
    print(f"  [CSV]  data/evaluation/results/latency_table.csv")
    print(f"  [JSON] data/evaluation/results/citation_audits.json")
    print("=" * 85)


if __name__ == "__main__":
    asyncio.run(main())
