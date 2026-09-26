"""CLI script to generate, validate, and serialize domain adaptation datasets for QLoRA."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.finetuning.dataset_generator import build_finetuning_dataset
from backend.app.finetuning.leakage_detector import LeakageDetector, DataLeakageError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("copilot.scripts.prepare_finetuning_data")


def prepare_data(
    output_dir: str = "data/finetuning",
    benchmark_path: str = "data/evaluation/benchmark_v1.json",
) -> None:
    """Generates instruction datasets, checks for data leakage, and serializes train/val JSONL files."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Generating enterprise fine-tuning dataset...")
    dataset = build_finetuning_dataset()

    logger.info("Running strict leakage detection against benchmark: %s", benchmark_path)
    detector = LeakageDetector(benchmark_path=benchmark_path)
    report = detector.validate_dataset(dataset.examples, raise_on_violation=True)

    logger.info(
        "Leakage check passed! %d examples validated, 0 violations detected against %d benchmark queries.",
        report["total_examples"],
        report["benchmark_queries_checked"],
    )

    # Separate into train and validation
    train_examples = [ex for ex in dataset.examples if ex.split == "train"]
    val_examples = [ex for ex in dataset.examples if ex.split == "validation"]

    train_path = dest / "train.jsonl"
    val_path = dest / "validation.jsonl"
    meta_path = dest / "dataset_metadata.json"

    # Write train.jsonl
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(ex.model_dump_json() + "\n")

    # Write validation.jsonl
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(ex.model_dump_json() + "\n")

    # Write dataset metadata
    metadata = {
        "dataset_name": dataset.dataset_name,
        "version": dataset.version,
        "created_at": dataset.created_at,
        "description": dataset.description,
        "source_corpus": dataset.source_corpus,
        "total_examples": len(dataset.examples),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "train_file": str(train_path),
        "validation_file": str(val_path),
        "dataset_sha256": report["dataset_sha256"],
        "leakage_check": {
            "status": "PASSED",
            "benchmark_queries_checked": report["benchmark_queries_checked"],
            "violations": 0,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("  ENTERPRISE FINETUNING DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"  Dataset Version:     {dataset.version}")
    print(f"  Total Examples:      {len(dataset.examples)}")
    print(f"  Training Split:      {len(train_examples)} examples -> {train_path}")
    print(f"  Validation Split:    {len(val_examples)} examples -> {val_path}")
    print(f"  Dataset SHA-256:     {report['dataset_sha256']}")
    print(f"  Leakage Validation:  PASSED (0 violations against {report['benchmark_queries_checked']} benchmark queries)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and validate fine-tuning datasets for QLoRA.")
    parser.add_argument("--output-dir", default="data/finetuning", help="Output directory for JSONL files")
    parser.add_argument("--benchmark-path", default="data/evaluation/benchmark_v1.json", help="Path to evaluation benchmark")
    args = parser.parse_args()

    try:
        prepare_data(output_dir=args.output_dir, benchmark_path=args.benchmark_path)
    except DataLeakageError as err:
        logger.error("FATAL: Data leakage detected during dataset preparation:\n%s", "\n".join(err.violations))
        sys.exit(1)
