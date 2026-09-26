"""CLI script to execute or validate QLoRA fine-tuning for enterprise domain adaptation."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.finetuning.config import get_finetuning_settings
from backend.app.finetuning.schemas import QLoRAConfig
from backend.app.finetuning.trainer import QLoRATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("copilot.scripts.train_qlora")


def run_training(
    base_model: str,
    output_dir: str,
    train_data: str,
    eval_data: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
) -> None:
    """Executes QLoRA training with hardware verification and graceful error handling."""
    config = QLoRAConfig(
        base_model=base_model,
        output_dir=output_dir,
        train_data_path=train_data,
        eval_data_path=eval_data,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    trainer = QLoRATrainer(config=config)
    hw = trainer.hardware

    print("\n" + "=" * 75)
    print("  QLoRA DOMAIN ADAPTATION TRAINING ENGINE")
    print("=" * 75)
    print(f"  Base Model:          {config.base_model}")
    print(f"  Target Checkpoint:   {config.output_dir}")
    print(f"  LoRA Rank (r):       {config.lora_r} | Alpha: {config.lora_alpha} | Dropout: {config.lora_dropout}")
    print(f"  Quantization:        {config.quantization_bits}-bit NF4 with double quantization")
    print(f"  Epochs / Batch Size: {config.num_epochs} epochs | Batch size: {config.batch_size}")
    print("-" * 75)
    print("  HARDWARE DIAGNOSTICS:")
    print(f"  - CUDA Available:    {hw.cuda_available}")
    print(f"  - Devices Detected:  {hw.device_count} ({', '.join(hw.device_names) if hw.device_names else 'None'})")
    print(f"  - GPU VRAM:          {hw.vram_gb:.2f} GB")
    print(f"  - System RAM:        {hw.system_ram_gb:.2f} GB")
    print(f"  - PyTorch Version:   {hw.pytorch_version}")
    print(f"  - bitsandbytes:      {'Installed' if hw.bitsandbytes_available else 'Missing'}")
    print(f"  - peft:              {'Installed' if hw.peft_available else 'Missing'}")
    print(f"  - QLoRA Compatible:  {hw.is_qlora_compatible}")
    print("=" * 75 + "\n")

    if not hw.is_qlora_compatible:
        print("[!] QLoRA TRAINING CANNOT PROCEED ON CURRENT HARDWARE:")
        for r in hw.incompatibility_reasons:
            print(f"    - {r}")
        print("\n[ACTIONABLE GPU EXECUTION GUIDE]")
        print("    1. Provision a host with an NVIDIA CUDA-enabled GPU (minimum 6 GB VRAM, e.g. T4, A10G, RTX 3080/4090).")
        print("    2. Verify NVIDIA drivers and CUDA toolkit: run 'nvidia-smi'.")
        print("    3. Install required GPU quantization packages:")
        print("       pip install torch bitsandbytes peft accelerate trl")
        print("    4. Execute training:")
        print("       python scripts/train_qlora.py\n")

        metadata = trainer.train(raise_on_incompatible=False)
        print(f"Recorded experiment state: status='{metadata.training_status}' in {config.output_dir}/run_metadata.json\n")
        # Exit with informative non-zero code to notify caller of hardware limitation
        sys.exit(2)

    logger.info("Hardware compatible. Starting training execution...")
    metadata = trainer.train(raise_on_incompatible=False)

    if metadata.training_status == "completed":
        print("\n" + "=" * 75)
        print("  QLoRA TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 75)
        print(f"  Run ID:        {metadata.run_id}")
        print(f"  Final Loss:    {metadata.final_loss}")
        print(f"  Total Steps:   {metadata.total_steps}")
        print(f"  Duration:      {metadata.training_duration_seconds:.2f}s")
        print(f"  Adapter Saved: {metadata.checkpoint_path}")
        print("=" * 75 + "\n")
        sys.exit(0)
    else:
        print(f"\n[!] Training halted with status '{metadata.training_status}': {metadata.error_message}\n")
        sys.exit(1)


if __name__ == "__main__":
    settings = get_finetuning_settings()
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter for domain-adapted grounded generation.")
    parser.add_argument("--base-model", default=settings.BASE_MODEL, help="Base model identifier")
    parser.add_argument("--output-dir", default=settings.OUTPUT_DIR, help="Checkpoint output directory")
    parser.add_argument("--train-data", default=settings.TRAIN_DATA, help="Path to train.jsonl")
    parser.add_argument("--eval-data", default=settings.EVAL_DATA, help="Path to validation.jsonl")
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=settings.LORA_R, help="LoRA rank r")
    parser.add_argument("--lora-alpha", type=int, default=settings.LORA_ALPHA, help="LoRA alpha scaling factor")

    args = parser.parse_args()
    run_training(
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_data=args.train_data,
        eval_data=args.eval_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
