import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.database.session import AsyncSessionLocal, init_db
from backend.app.evaluation.runner import EvaluationRunner


async def main():
    print("=" * 75)
    print("  ENTERPRISE AI COPILOT: RETRIEVAL EVALUATION & ABLATION")
    print("=" * 75)

    await init_db()
    runner = EvaluationRunner()
    dataset = runner.load_dataset()
    print(f"\n[INFO] Loaded benchmark dataset: {dataset.dataset_name} (v{dataset.version})")
    print(f"[INFO] Total queries: {len(dataset.queries)} | Non-abstention queries: {len([q for q in dataset.queries if not q.expected_abstention])}")

    async with AsyncSessionLocal() as session:
        print("[INFO] Verifying corpus ingestion...")
        await runner.ensure_corpus_ingested(session)

        print("\n[INFO] Running 4-way retrieval ablation across benchmark queries...")
        eval_res = await runner.retrieval_evaluator.run_ablation(
            queries=dataset.queries,
            session=session,
        )

        configs = eval_res["configurations"]
        ablation_rows = eval_res["ablation_table"]

        print("\n" + "=" * 85)
        print("  RETRIEVAL ABLATION RESULTS TABLE")
        print("=" * 85)
        header = f"{'Configuration':<22} | {'Recall@1':<8} | {'Recall@5':<8} | {'Recall@10':<9} | {'P@5':<6} | {'MRR':<6} | {'NDCG@5':<7} | {'Latency (ms)':<12}"
        print(header)
        print("-" * len(header))

        for row in ablation_rows:
            key = row.configuration.lower().replace(" ", "_").replace("+_", "")
            c = configs.get(key) or configs.get(row.configuration.lower().replace(" ", "_"))
            if not c:
                # fallback matching
                for k, v in configs.items():
                    if k in row.configuration.lower():
                        c = v
                        break
            r1 = f"{c.recall_at_1:.4f}" if c else "N/A"
            r5 = f"{row.recall_at_5:.4f}"
            r10 = f"{c.recall_at_10:.4f}" if c else "N/A"
            p5 = f"{row.precision_at_5:.4f}"
            mrr = f"{row.mrr:.4f}"
            ndcg5 = f"{row.ndcg_at_5:.4f}"
            lat = f"{row.latency_ms:.2f}"

            print(f"{row.configuration:<22} | {r1:<8} | {r5:<8} | {r10:<9} | {p5:<6} | {mrr:<6} | {ndcg5:<7} | {lat:<12}")

        print("=" * 85)
        print("  RETRIEVAL EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 85)


if __name__ == "__main__":
    asyncio.run(main())
