import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.database.session import AsyncSessionLocal, init_db
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.models.schemas import ResearchRequest


async def main():
    print("=" * 75)
    print("  ENTERPRISE AI COPILOT: CITATION AUDIT & VERIFICATION EVALUATION")
    print("=" * 75)

    await init_db()
    runner = EvaluationRunner()
    dataset = runner.load_dataset()

    async with AsyncSessionLocal() as session:
        print("[INFO] Verifying corpus ingestion...")
        await runner.ensure_corpus_ingested(session)

    eval_queries = [q for q in dataset.queries if not q.expected_abstention]
    print(f"\n[INFO] Auditing citations across {len(eval_queries)} research queries...\n")

    audits = []
    for q in eval_queries:
        req = ResearchRequest(
            query=q.query,
            mode="hybrid",
            top_k=3,
            candidate_top_k=15,
            rerank=True,
            context_max_tokens=1500,
        )
        res = await runner.research_service.execute_research(req)
        retrieved_chunk_ids = {str(item.chunk_id) for item in res.evidence}

        audit = runner.citation_evaluator.audit_query_citations(
            query_id=q.query_id,
            query_text=q.query,
            retrieved_chunk_ids=retrieved_chunk_ids,
            claims=[c.model_dump() for c in res.claims],
            citations=[cit.model_dump() for cit in res.citations],
            verifications=[v.model_dump() for v in res.verification.details],
        )
        audits.append(audit)

        print(f"[{q.query_id}] '{q.query[:55]}...'")
        print(f"      Claims: {len(res.claims)} | Citations: {len(res.citations)} | Status: {res.verification.status}")
        for ca in audit["claim_audits"]:
            hallucinated_note = " [HALLUCINATED CHUNK]" if ca["has_hallucinated_citations"] else ""
            print(f"        - Claim: \"{ca['claim_text'][:50]}...\" -> {ca['verification_status']} (score: {ca['support_score']:.2f}){hallucinated_note}")

    metrics = runner.citation_evaluator.aggregate_metrics(audits)
    print("\n" + "=" * 75)
    print("  AGGREGATE CITATION EVALUATION METRICS")
    print("=" * 75)
    print(f"  Total Queries Audited:         {metrics.total_queries_evaluated}")
    print(f"  Total Claims Extracted:        {metrics.total_claims}")
    print(f"  Total Citations Generated:     {metrics.total_citations}")
    print(f"  Citation Validity Rate:        {metrics.citation_validity_rate * 100:.2f}% (Valid chunk ID match)")
    print(f"  Citation Coverage Rate:        {metrics.citation_coverage_rate * 100:.2f}% (Claims with citation)")
    print(f"  Citation Support Rate:         {metrics.citation_support_rate * 100:.2f}% (Fully supported claims)")
    print(f"  Partially Supported Rate:      {metrics.partially_supported_rate * 100:.2f}%")
    print(f"  Unsupported Claim Rate:        {metrics.unsupported_claim_rate * 100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
