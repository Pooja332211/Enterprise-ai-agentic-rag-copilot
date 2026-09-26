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
    print("  ENTERPRISE AI COPILOT: GROUNDED GENERATION EVALUATION")
    print("=" * 75)

    await init_db()
    runner = EvaluationRunner()
    dataset = runner.load_dataset()

    async with AsyncSessionLocal() as session:
        print("[INFO] Verifying corpus ingestion...")
        await runner.ensure_corpus_ingested(session)

    eval_queries = [q for q in dataset.queries if not q.expected_abstention]
    print(f"\n[INFO] Evaluating {len(eval_queries)} grounded benchmark queries...")

    evaluations = []
    print("\n" + "-" * 75)
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
        eval_item = runner.generation_evaluator.evaluate_response(
            generated_answer=res.answer,
            reference_answer=q.expected_answer,
            retrieved_contexts=[e.text for e in res.evidence],
            claims=[c.model_dump() for c in res.claims],
            verifications=[v.model_dump() for v in res.verification.details],
        )
        evaluations.append(eval_item)

        print(f"[{q.query_id}] Query: '{q.query[:50]}...'")
        print(f"      Token F1: {eval_item['token_f1']:.4f} | Bigram F1: {eval_item['bigram_f1']:.4f} | Grounding: {eval_item['grounding_score']:.4f} | Completeness: {eval_item['completeness_score']:.4f}")

    metrics = runner.generation_evaluator.aggregate_metrics(evaluations)
    print("\n" + "=" * 75)
    print("  AGGREGATE GENERATION EVALUATION SUMMARY")
    print("=" * 75)
    print(f"  Total Grounded Queries Evaluated:    {metrics.total_answered_queries}")
    print(f"  Mean Answer Completeness:           {metrics.mean_answer_completeness:.4f}")
    print(f"  Mean Evidence Grounding Score:      {metrics.mean_evidence_grounding_score:.4f}")
    print(f"  Mean Reference Token Overlap (F1):  {metrics.mean_reference_token_overlap:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
