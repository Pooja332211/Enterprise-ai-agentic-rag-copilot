"""Evaluates the LangGraph agentic research workflow on the dedicated agent benchmark."""

import asyncio
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select
from backend.app.agents.agent_service import get_agent_research_service
from backend.app.config.settings import get_settings
from backend.app.database.models import Document
from backend.app.database.session import AsyncSessionLocal, init_db
from backend.app.models.schemas import AgentResearchRequest
from backend.app.services.indexing_service import IndexingService
from backend.app.services.ingestion_service import IngestionService

BENCHMARK_PATH = Path("data/evaluation/agent_benchmark_v1.json")
RESULTS_DIR = Path("data/evaluation/results")
FIXTURES_DIR = Path("backend/tests/fixtures")

TEST_FILES = [
    ("sample.pdf", "application/pdf"),
    ("sample.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("sample.txt", "text/plain"),
    ("sample.md", "text/markdown"),
    ("sample.html", "text/html"),
]


async def ensure_corpus_ingested() -> None:
    """Ensures evaluation fixtures are ingested and indexed in the vector and BM25 stores."""
    settings = get_settings()
    ingestion_service = IngestionService(settings)
    indexing_service = IndexingService(settings=settings)

    await init_db()
    async with AsyncSessionLocal() as session:
        existing_res = await session.execute(select(Document))
        existing_docs = {d.filename: d for d in existing_res.scalars().all()}

        newly_ingested = False
        for filename, mime_type in TEST_FILES:
            file_path = FIXTURES_DIR / filename
            if not file_path.exists():
                continue
            if filename in existing_docs:
                continue

            file_bytes = file_path.read_bytes()
            await ingestion_service.ingest_file(
                filename=filename,
                content=file_bytes,
                mime_type=mime_type,
                db=session,
            )
            newly_ingested = True

        await session.commit()

        print("  Rebuilding vector & BM25 indexes for evaluation corpus...")
        await indexing_service.rebuild_all_indexes(db=session)


async def run_agent_benchmark() -> Dict[str, Any]:
    print("=" * 85)
    print("  ENTERPRISE AI COPILOT: MILESTONE 8 AGENTIC RESEARCH WORKFLOW BENCHMARK")
    print("  LangGraph State Machine: Query Planning, Decomposition, Aggregation & Verification")
    print("=" * 85)

    await ensure_corpus_ingested()

    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_PATH}")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    queries = benchmark_data.get("queries", [])
    print(f"\n  Loaded {len(queries)} evaluation queries from {BENCHMARK_PATH}")

    settings = get_settings()
    agent_service = get_agent_research_service(settings=settings)

    query_results: List[Dict[str, Any]] = []
    decomposition_correct = 0
    sufficiency_correct = 0
    grounded_correct = 0
    abstention_correct = 0
    total_citations_evaluated = 0
    valid_citations_evaluated = 0
    total_subqueries = 0
    total_retrieval_calls = 0

    pipeline_timings: List[float] = []
    retrieval_timings: List[float] = []
    rerank_timings: List[float] = []
    generation_timings: List[float] = []
    verification_timings: List[float] = []

    print("\n  Executing Agentic Research Workflow per query...")
    print("  " + "-" * 81)

    for idx, q_item in enumerate(queries, start=1):
        q_id = q_item["query_id"]
        q_text = q_item["query"]
        category = q_item["category"]
        exp_decomp = q_item["expected_decomposition"]
        exp_abstain = q_item["expected_abstention"]

        req = AgentResearchRequest(
            query=q_text,
            top_k=5,
            candidate_top_k=15,
            rerank=True,
            max_iterations=2,
            enable_decomposition=True,
            temperature=0.0,
        )

        t_start = time.perf_counter()
        resp = await agent_service.execute_agent_research(req)
        elapsed_s = time.perf_counter() - t_start

        # Track metrics
        sub_q_count = len(resp.sub_queries)
        total_subqueries += sub_q_count

        # Count retrieval calls from execution trace
        retrieval_nodes = [t for t in resp.execution_trace if t["node"] == "retrieve"]
        retrieval_calls_this_query = len(retrieval_nodes)
        total_retrieval_calls += retrieval_calls_this_query

        # Decomposition accuracy: check if initial planning performed decomposition
        plan_nodes = [t for t in resp.execution_trace if t["node"] == "plan_query"]
        if plan_nodes:
            is_decomp_actual = "Generated 1 sub-queries" not in plan_nodes[0].get("output_summary", "")
        else:
            is_decomp_actual = sub_q_count > 1
        decomp_match = (is_decomp_actual == exp_decomp)
        if decomp_match:
            decomposition_correct += 1

        # Abstention accuracy: evaluated strictly over unanswerable queries
        if exp_abstain:
            if resp.insufficient_evidence:
                abstention_correct += 1

        # Sufficiency accuracy:
        # For non-abstain queries, evidence was deemed sufficient and answer generated
        # For abstain queries, evidence was deemed insufficient
        suff_match = (not resp.insufficient_evidence) if not exp_abstain else resp.insufficient_evidence
        if suff_match:
            sufficiency_correct += 1

        # Grounding accuracy
        if not exp_abstain:
            if resp.grounded:
                grounded_correct += 1

        # Citation validity
        evidence_chunk_ids = {ev["chunk_id"] for ev in resp.evidence}
        for cit in resp.citations:
            total_citations_evaluated += 1
            if cit.chunk_id in evidence_chunk_ids:
                valid_citations_evaluated += 1

        # Timings
        timings = resp.timings_ms
        pipeline_timings.append(timings.get("total_pipeline_ms", elapsed_s * 1000))
        retrieval_timings.append(timings.get("retrieval_ms", 0.0))
        rerank_timings.append(timings.get("reranking_ms", 0.0))
        generation_timings.append(timings.get("generation_ms", 0.0))
        verification_timings.append(timings.get("citation_verification_ms", 0.0))

        status_str = f"status={resp.status} | subqueries={sub_q_count} | iter={resp.iterations_used} | latency={elapsed_s * 1000:.1f}ms"
        print(f"  [{idx:02d}/{len(queries):02d}] {q_id:<14} | {category:<15} | {status_str}")

        query_results.append({
            "query_id": q_id,
            "query": q_text,
            "category": category,
            "expected_decomposition": exp_decomp,
            "actual_decomposition": is_decomp_actual,
            "sub_queries": resp.sub_queries,
            "expected_abstention": exp_abstain,
            "actual_abstention": resp.insufficient_evidence,
            "status": resp.status,
            "grounded": resp.grounded,
            "iterations_used": resp.iterations_used,
            "evidence_count": len(resp.evidence),
            "claims_count": len(resp.claims),
            "citations_count": len(resp.citations),
            "verification_status": resp.verification.status,
            "timings_ms": timings,
            "final_answer": resp.final_answer,
            "execution_trace": resp.execution_trace,
        })

    # Summary aggregations
    num_queries = len(queries)
    answerable_count = sum(1 for q in queries if not q["expected_abstention"])
    unanswerable_count = sum(1 for q in queries if q["expected_abstention"])

    decomp_acc = decomposition_correct / num_queries if num_queries > 0 else 0.0
    suff_acc = sufficiency_correct / num_queries if num_queries > 0 else 0.0
    abstain_acc = abstention_correct / unanswerable_count if unanswerable_count > 0 else 1.0
    grounding_rate = grounded_correct / answerable_count if answerable_count > 0 else 0.0
    citation_validity = (
        valid_citations_evaluated / total_citations_evaluated if total_citations_evaluated > 0 else 1.0
    )

    avg_pipeline_ms = sum(pipeline_timings) / len(pipeline_timings) if pipeline_timings else 0.0
    avg_retrieval_ms = sum(retrieval_timings) / len(retrieval_timings) if retrieval_timings else 0.0
    avg_rerank_ms = sum(rerank_timings) / len(rerank_timings) if rerank_timings else 0.0
    avg_generation_ms = sum(generation_timings) / len(generation_timings) if generation_timings else 0.0
    avg_verification_ms = sum(verification_timings) / len(verification_timings) if verification_timings else 0.0

    summary_metrics = {
        "dataset_name": benchmark_data.get("dataset_name"),
        "dataset_version": benchmark_data.get("version"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "framework": "LangGraph state machine",
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_provider": settings.LLM_PROVIDER,
        },
        "metrics": {
            "total_queries": num_queries,
            "answerable_queries": answerable_count,
            "unanswerable_queries": unanswerable_count,
            "decomposition_accuracy": round(decomp_acc, 4),
            "sufficiency_accuracy": round(suff_acc, 4),
            "abstention_accuracy": round(abstain_acc, 4),
            "grounding_rate": round(grounding_rate, 4),
            "citation_validity_rate": round(citation_validity, 4),
            "total_citations_evaluated": total_citations_evaluated,
            "valid_citations_evaluated": valid_citations_evaluated,
            "average_subqueries_per_query": round(total_subqueries / num_queries, 2),
            "total_retrieval_calls": total_retrieval_calls,
        },
        "latency_profile_ms": {
            "mean_pipeline_latency": round(avg_pipeline_ms, 2),
            "mean_retrieval_latency": round(avg_retrieval_ms, 2),
            "mean_rerank_latency": round(avg_rerank_ms, 2),
            "mean_generation_latency": round(avg_generation_ms, 2),
            "mean_verification_latency": round(avg_verification_ms, 2),
        },
        "queries": query_results,
    }

    # Print summary tables
    print("\n" + "=" * 85)
    print("  MILESTONE 8 AGENTIC RESEARCH EVALUATION RESULTS")
    print("=" * 85)
    print(f"  Decomposition Accuracy:      {decomp_acc * 100:6.2f}% ({decomposition_correct}/{num_queries})")
    print(f"  Evidence Sufficiency Acc:    {suff_acc * 100:6.2f}% ({sufficiency_correct}/{num_queries})")
    print(f"  Abstention Accuracy:         {abstain_acc * 100:6.2f}% ({abstention_correct}/{unanswerable_count})")
    print(f"  Answer Grounding Rate:       {grounding_rate * 100:6.2f}% ({grounded_correct}/{answerable_count})")
    print(f"  Citation Validity Rate:      {citation_validity * 100:6.2f}% ({valid_citations_evaluated}/{total_citations_evaluated})")
    print(f"  Avg Sub-Queries / Query:     {total_subqueries / num_queries:6.2f}")
    print(f"  Total Retrieval Calls:       {total_retrieval_calls}")
    print("-" * 85)
    print("  Multi-Stage Latency Breakdown (Mean ms):")
    print(f"    Retrieval per SubQuery:    {avg_retrieval_ms:8.2f} ms")
    print(f"    Neural Reranking:          {avg_rerank_ms:8.2f} ms")
    print(f"    Grounded Generation:       {avg_generation_ms:8.2f} ms")
    print(f"    Citation Verification:     {avg_verification_ms:8.2f} ms")
    print(f"    Total Agent Pipeline:      {avg_pipeline_ms:8.2f} ms")
    print("=" * 85)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "agent_benchmark_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"\n[OK] Evaluation metrics written to {out_file}\n")
    return summary_metrics


if __name__ == "__main__":
    asyncio.run(run_agent_benchmark())
