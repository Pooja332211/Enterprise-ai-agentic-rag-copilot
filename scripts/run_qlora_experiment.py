"""Master CLI orchestrator for Milestone 7: QLoRA Domain Adaptation Experiment.

Executes the full experimental pipeline:
1. Instruction dataset generation & serialization
2. Strict data leakage verification against Milestone 6 benchmark
3. Hardware diagnostics & QLoRA training execution/validation
4. Comparative Base Model vs QLoRA Model evaluation
5. Machine-readable artifact serialization
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.finetuning.config import get_finetuning_settings
from backend.app.finetuning.dataset_generator import build_finetuning_dataset
from backend.app.finetuning.evaluator import QLoRAEvaluator
from backend.app.finetuning.leakage_detector import LeakageDetector, DataLeakageError
from backend.app.finetuning.trainer import QLoRATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("copilot.scripts.run_qlora_experiment")


async def run_experiment(
    output_dir: str = "data/finetuning",
    benchmark_path: str = "data/evaluation/benchmark_v1.json",
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> None:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    results_dir = dest / "results"
    checkpoints_dir = dest / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  MILESTONE 7: QLoRA DOMAIN ADAPTATION EXPERIMENT")
    print("=" * 80)
    print(f"  Base Model:       {base_model}")
    print(f"  Data Directory:   {output_dir}")
    print(f"  Benchmark Path:   {benchmark_path}")
    print(f"  Results Path:     {results_dir}")
    print("=" * 80 + "\n")

    # ---------------------------------------------------------
    # PHASE 1: Data Preparation & Leakage Protection
    # ---------------------------------------------------------
    print(">>> PHASE 1: GENERATING INSTRUCTION DATASET & LEAKAGE AUDIT")
    dataset = build_finetuning_dataset()
    detector = LeakageDetector(benchmark_path=benchmark_path)
    leakage_report = detector.validate_dataset(dataset.examples, raise_on_violation=True)

    train_path = dest / "train.jsonl"
    val_path = dest / "validation.jsonl"

    train_examples = [ex for ex in dataset.examples if ex.split == "train"]
    val_examples = [ex for ex in dataset.examples if ex.split == "validation"]

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(ex.model_dump_json() + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(ex.model_dump_json() + "\n")

    print(f"  [OK] Dataset generated: {len(dataset.examples)} total examples (Train: {len(train_examples)}, Val: {len(val_examples)})")
    print(f"  [OK] Dataset SHA-256:   {leakage_report['dataset_sha256']}")
    print(f"  [OK] Leakage Check:     PASSED (0 violations detected across {leakage_report['benchmark_queries_checked']} benchmark queries)\n")

    # ---------------------------------------------------------
    # PHASE 2: Hardware Readiness & Training Attempt
    # ---------------------------------------------------------
    print(">>> PHASE 2: HARDWARE READINESS & QLoRA TRAINING ATTEMPT")
    settings = get_finetuning_settings()
    config = settings.to_qlora_config()
    config.base_model = base_model
    config.train_data_path = str(train_path)
    config.eval_data_path = str(val_path)
    config.output_dir = str(checkpoints_dir)

    trainer = QLoRATrainer(config=config)
    hw = trainer.hardware

    print(f"  - CUDA Available:    {hw.cuda_available}")
    print(f"  - Device Count:      {hw.device_count} ({', '.join(hw.device_names) if hw.device_names else 'None'})")
    print(f"  - VRAM:              {hw.vram_gb:.2f} GB")
    print(f"  - System RAM:        {hw.system_ram_gb:.2f} GB")
    print(f"  - bitsandbytes:      {'Installed' if hw.bitsandbytes_available else 'Missing'}")
    print(f"  - peft:              {'Installed' if hw.peft_available else 'Missing'}")
    print(f"  - QLoRA Compatible:  {hw.is_qlora_compatible}")

    train_metadata = trainer.train(raise_on_incompatible=False)
    if train_metadata.training_status == "completed":
        print(f"  [OK] Training completed: Loss={train_metadata.final_loss}, Steps={train_metadata.total_steps}")
    else:
        print(f"  [!] QLoRA Training Status: '{train_metadata.training_status}'")
        print("      Reason: Hardware requires an NVIDIA CUDA GPU with minimum 6 GB VRAM.")
        print("      Metadata saved: data/finetuning/checkpoints/run_metadata.json\n")

    # ---------------------------------------------------------
    # PHASE 3: Comparative Evaluation
    # ---------------------------------------------------------
    print(">>> PHASE 3: COMPARATIVE EVALUATION (BASE VS ADAPTED MODEL)")
    evaluator = QLoRAEvaluator(
        base_model_name=base_model,
        adapter_path=f"{checkpoints_dir}/final_adapter",
        benchmark_path=benchmark_path,
        output_dir=str(results_dir),
    )
    comparison = await evaluator.evaluate_models()

    print("\n" + "=" * 80)
    print("  MODEL ABLATION COMPARISON TABLE (MILESTONE 6 BENCHMARK)")
    print("=" * 80)
    print(f"{'Metric':<38} | {'Base Model':<18} | {'QLoRA Model':<20}")
    print("-" * 80)
    for row in comparison.comparison_table:
        print(f"{row['metric']:<38} | {row['base_model']:<18} | {row['qlora_model']:<20}")
    print("=" * 80)
    print(f"  Status Note: {comparison.notes}\n")

    # ---------------------------------------------------------
    # PHASE 4: Export Summary & Verification
    # ---------------------------------------------------------
    print(">>> PHASE 4: ARTIFACT EXPORT & VERIFICATION")
    print(f"  [Artifact] Dataset Metadata:   data/finetuning/dataset_metadata.json")
    print(f"  [Artifact] Train Split:        {train_path}")
    print(f"  [Artifact] Validation Split:   {val_path}")
    print(f"  [Artifact] Training Run Meta:  {checkpoints_dir}/run_metadata.json")
    print(f"  [Artifact] Ablation Table CSV: {results_dir}/model_ablation_table.csv")
    print(f"  [Artifact] Evaluation JSON:    {results_dir}/qlora_evaluation_summary.json")
    print("\n[OK] Milestone 7 QLoRA Domain Adaptation experiment workflow completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="Run Milestone 7 QLoRA domain adaptation experiment.")
    parser.add_argument("--output-dir", default="data/finetuning", help="Output directory")
    parser.add_argument("--benchmark-path", default="data/evaluation/benchmark_v1.json", help="Path to benchmark")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier")
    args = parser.parse_args()

    try:
        asyncio.run(run_experiment(
            output_dir=args.output_dir,
            benchmark_path=args.benchmark_path,
            base_model=args.base_model,
        ))
    except DataLeakageError as err:
        logger.error("FATAL: Data leakage detected during experiment run:\n%s", "\n".join(err.violations))
        sys.exit(1)


if __name__ == "__main__":
    main()
