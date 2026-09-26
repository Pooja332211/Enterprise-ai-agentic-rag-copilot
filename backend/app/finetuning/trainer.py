"""QLoRA fine-tuning pipeline orchestration with hardware verification and reproducibility controls."""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.finetuning.config import check_gpu_readiness, get_finetuning_settings
from backend.app.finetuning.schemas import (
    FinetuningExample,
    HardwareStatus,
    QLoRAConfig,
    TrainingRunMetadata,
)

logger = logging.getLogger("copilot.finetuning.trainer")


class HardwareIncompatibilityError(Exception):
    """Raised when QLoRA training is executed on incompatible hardware (e.g. CPU or non-CUDA GPU)."""
    def __init__(self, message: str, status: HardwareStatus):
        super().__init__(message)
        self.status = status


class QLoRATrainer:
    """Orchestrates 4-bit QLoRA fine-tuning for domain-specific grounded generation."""

    def __init__(self, config: Optional[QLoRAConfig] = None):
        self.config = config or get_finetuning_settings().to_qlora_config()
        self.hardware = check_gpu_readiness()

    def format_prompt(self, example: FinetuningExample) -> str:
        """Formats an instruction example using standard ChatML instruction format."""
        return (
            f"<|im_start|>system\n{example.system_instruction}<|im_end|>\n"
            f"<|im_start|>user\nContext Evidence:\n{example.context_evidence}\n\n"
            f"Question:\n{example.query}<|im_end|>\n"
            f"<|im_start|>assistant\n{example.grounded_answer}<|im_end|>"
        )

    def load_jsonl_dataset(self, file_path: str) -> List[FinetuningExample]:
        """Loads and parses fine-tuning examples from a JSONL file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Fine-tuning dataset file not found: {file_path}")

        examples: List[FinetuningExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    data = json.loads(clean)
                    examples.append(FinetuningExample(**data))
                except Exception as exc:
                    raise ValueError(f"Malformed JSONL at line {line_no} in {file_path}: {exc}")
        return examples

    def save_run_metadata(self, metadata: TrainingRunMetadata, output_path: Optional[str] = None) -> str:
        """Serializes training experiment metadata to JSON."""
        dest = Path(output_path or f"{self.config.output_dir}/run_metadata.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))
        return str(dest)

    def train(self, raise_on_incompatible: bool = False) -> TrainingRunMetadata:
        """Executes the QLoRA training pipeline.

        If hardware is incompatible (no CUDA GPU / missing bitsandbytes), raises
        HardwareIncompatibilityError if raise_on_incompatible=True, or returns
        a structured failure metadata record detailing actionable GPU execution steps.
        """
        run_id = f"qlora_run_{int(time.time())}"
        start_time = time.time()

        # Step 1: Check hardware compatibility
        if not self.hardware.is_qlora_compatible:
            reasons_str = "\n".join(f"  - {r}" for r in self.hardware.incompatibility_reasons)
            msg = (
                f"QLoRA Training Halted: Incompatible hardware environment.\n"
                f"Diagnostics:\n{reasons_str}\n\n"
                f"Actionable Resolution for GPU Execution:\n"
                f"1. Provision a host with an NVIDIA CUDA-enabled GPU (e.g. AWS g5.xlarge with A10G, or RunPod RTX 4090/A5000).\n"
                f"2. Ensure CUDA Toolkit 11.8 or 12.1+ is installed.\n"
                f"3. Install GPU quantization packages: pip install torch bitsandbytes peft accelerate trl.\n"
                f"4. Re-run: python scripts/train_qlora.py"
            )
            logger.warning(msg)

            metadata = TrainingRunMetadata(
                run_id=run_id,
                config=self.config,
                hardware=self.hardware,
                training_status="failed_hardware_incompatibility",
                training_duration_seconds=round(time.time() - start_time, 2),
                error_message=msg,
            )
            self.save_run_metadata(metadata)

            if raise_on_incompatible:
                raise HardwareIncompatibilityError(msg, self.hardware)
            return metadata

        # Step 2: Compatible GPU execution path
        try:
            import torch  # type: ignore
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                Trainer,
                TrainingArguments,
            )

            # Set reproducible random seeds
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.random_seed)

            logger.info("Loading training and validation datasets...")
            train_examples = self.load_jsonl_dataset(self.config.train_data_path)
            val_examples = self.load_jsonl_dataset(self.config.eval_data_path)

            logger.info("Initializing 4-bit BitsAndBytesConfig...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype, torch.bfloat16),
                bnb_4bit_use_double_quant=self.config.use_double_quant,
            )

            logger.info("Loading base tokenizer and model: %s", self.config.base_model)
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                padding_side="right",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            model = get_peft_model(model, lora_config)

            # Tokenize prompts
            train_texts = [self.format_prompt(ex) for ex in train_examples]
            val_texts = [self.format_prompt(ex) for ex in val_examples]

            def tokenize_fn(texts: List[str]):
                return tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                )

            train_encodings = tokenize_fn(train_texts)
            val_encodings = tokenize_fn(val_texts)

            class TextDataset(torch.utils.data.Dataset):
                def __init__(self, encodings):
                    self.encodings = encodings

                def __len__(self):
                    return len(self.encodings["input_ids"])

                def __getitem__(self, idx):
                    item = {k: v[idx] for k, v in self.encodings.items()}
                    item["labels"] = item["input_ids"].clone()
                    return item

            train_ds = TextDataset(train_encodings)
            val_ds = TextDataset(val_encodings)

            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                lr_scheduler_type=self.config.lr_scheduler_type,
                logging_steps=5,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                fp16=(self.config.bnb_4bit_compute_dtype == "float16"),
                bf16=(self.config.bnb_4bit_compute_dtype == "bfloat16"),
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
            )

            logger.info("Starting QLoRA training loop...")
            train_result = trainer.train()

            # Save LoRA adapter checkpoint
            checkpoint_dir = Path(self.config.output_dir) / "final_adapter"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            duration = round(time.time() - start_time, 2)
            metadata = TrainingRunMetadata(
                run_id=run_id,
                config=self.config,
                hardware=self.hardware,
                num_train_examples=len(train_examples),
                num_eval_examples=len(val_examples),
                training_status="completed",
                training_duration_seconds=duration,
                total_steps=train_result.global_step,
                final_loss=round(float(train_result.training_loss), 4),
                checkpoint_path=str(checkpoint_dir),
            )
            self.save_run_metadata(metadata)
            return metadata

        except Exception as exc:
            duration = round(time.time() - start_time, 2)
            err_msg = f"QLoRA training failed during execution: {exc}"
            logger.error(err_msg, exc_info=True)
            metadata = TrainingRunMetadata(
                run_id=run_id,
                config=self.config,
                hardware=self.hardware,
                training_status="failed",
                training_duration_seconds=duration,
                error_message=err_msg,
            )
            self.save_run_metadata(metadata)
            return metadata
