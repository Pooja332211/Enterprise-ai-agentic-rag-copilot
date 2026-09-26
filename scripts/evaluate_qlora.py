"""CLI script to evaluate Base Model vs QLoRA Model using the Milestone 6 evaluation framework."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.finetuning.evaluator import QLoRAEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("copilot.scripts.evaluate_qlora")


async def main_async(
    base_model: str,
    adapter_path: str,
    benchmark_path: str,
    output_dir: str,
) -> None:
    evaluator = QLoRAEvaluator(
        base_model_name=base_model,
        adapter_path=adapter_path,
        benchmark_path=benchmark_path,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("  BASE MODEL VS QLoRA DOMAIN ADAPTATION EVALUATION")
    print("=" * 80)
    print(f"  Base Model:       {base_model}")
    print(f"  Adapter Path:     {adapter_path}")
    print(f"  Benchmark:        {benchmark_path}")
    print(f"  Output Directory: {output_dir}")
    print("=" * 80 + "\n")

    logger.info("Executing comparative model evaluation on benchmark...")
    comparison = await evaluator.evaluate_models()

    print("\n" + "=" * 80)
    print("  EVALUATION ABLATION TABLE: BASE INSTRUCT VS QLoRA ADAPTED MODEL")
    print("=" * 80)
    print(f"{'Metric':<38} | {'Base Model':<18} | {'QLoRA Model':<20}")
    print("-" * 80)
    for row in comparison.comparison_table:
        metric = row["metric"]
        base = row["base_model"]
        qlora = row["qlora_model"]
        print(f"{metric:<38} | {base:<18} | {qlora:<20}")
    print("=" * 80)
    print(f"  Notes: {comparison.notes}")
    print(f"  Artifacts saved to: {output_dir}/\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs QLoRA Model.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier")
    parser.add_argument("--adapter-path", default="data/finetuning/checkpoints/final_adapter", help="Path to adapter checkpoint")
    parser.add_argument("--benchmark-path", default="data/evaluation/benchmark_v1.json", help="Path to evaluation benchmark")
    parser.add_argument("--output-dir", default="data/finetuning/results", help="Directory for output tables")
    args = parser.parse_args()

    asyncio.run(main_async(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        benchmark_path=args.benchmark_path,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
