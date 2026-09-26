"""Configuration and hardware inspection for QLoRA fine-tuning experiments."""

import os
import platform
import psutil
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from backend.app.finetuning.schemas import HardwareStatus, QLoRAConfig


class FinetuningSettings(BaseSettings):
    """Environment-driven settings for QLoRA experiments."""
    model_config = SettingsConfigDict(
        env_prefix="QLORA_",
        env_file=".env",
        extra="ignore",
    )

    BASE_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
    OUTPUT_DIR: str = "data/finetuning/checkpoints"
    TRAIN_DATA: str = "data/finetuning/train.jsonl"
    EVAL_DATA: str = "data/finetuning/validation.jsonl"
    RESULTS_DIR: str = "data/finetuning/results"
    BENCHMARK_PATH: str = "data/evaluation/benchmark_v1.json"

    MAX_SEQ_LENGTH: int = 1024
    BATCH_SIZE: int = 2
    GRADIENT_ACCUMULATION_STEPS: int = 4
    LEARNING_RATE: float = 2e-4
    NUM_EPOCHS: int = 3
    WARMUP_RATIO: float = 0.03
    WEIGHT_DECAY: float = 0.01

    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    QUANTIZATION_BITS: int = 4
    RANDOM_SEED: int = 42

    def to_qlora_config(self) -> QLoRAConfig:
        """Construct a QLoRAConfig dataclass from settings."""
        return QLoRAConfig(
            base_model=self.BASE_MODEL,
            output_dir=self.OUTPUT_DIR,
            train_data_path=self.TRAIN_DATA,
            eval_data_path=self.EVAL_DATA,
            max_seq_length=self.MAX_SEQ_LENGTH,
            batch_size=self.BATCH_SIZE,
            gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.LEARNING_RATE,
            num_epochs=self.NUM_EPOCHS,
            warmup_ratio=self.WARMUP_RATIO,
            weight_decay=self.WEIGHT_DECAY,
            lora_r=self.LORA_R,
            lora_alpha=self.LORA_ALPHA,
            lora_dropout=self.LORA_DROPOUT,
            quantization_bits=self.QUANTIZATION_BITS,
            random_seed=self.RANDOM_SEED,
        )


def check_gpu_readiness() -> HardwareStatus:
    """Inspects the local runtime environment for CUDA GPU availability and QLoRA compatibility.

    Returns a HardwareStatus object with complete diagnostics and actionable incompatibility reasons.
    """
    reasons: List[str] = []
    cuda_available = False
    device_count = 0
    device_names: List[str] = []
    vram_gb = 0.0
    torch_version = "not_installed"
    bitsandbytes_available = False
    peft_available = False

    # Check system RAM
    try:
        vm = psutil.virtual_memory()
        system_ram_gb = round(vm.total / (1024 ** 3), 2)
    except Exception:
        system_ram_gb = 0.0

    # Check PyTorch & CUDA
    try:
        import torch  # type: ignore
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_names.append(torch.cuda.get_device_name(i))
            if device_count > 0:
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = round(vram_bytes / (1024 ** 3), 2)
        else:
            reasons.append("CUDA is not available in PyTorch runtime.")
    except Exception as exc:
        reasons.append(f"PyTorch CUDA inspection error: {exc}")

    if not cuda_available:
        if "CUDA is not available in PyTorch runtime." not in reasons:
            reasons.append("No NVIDIA CUDA GPU detected on system.")

    # Check bitsandbytes
    try:
        import bitsandbytes  # type: ignore
        bitsandbytes_available = True
    except ImportError:
        reasons.append("Package 'bitsandbytes' is not installed (required for 4-bit NF4 quantization).")
    except Exception as exc:
        reasons.append(f"bitsandbytes loading error: {exc}")

    # Check peft
    try:
        import peft  # type: ignore
        peft_available = True
    except ImportError:
        reasons.append("Package 'peft' is not installed (required for LoRA adapter management).")
    except Exception as exc:
        reasons.append(f"peft loading error: {exc}")

    # VRAM adequacy check (minimum 6 GB VRAM for 1.5B 4-bit QLoRA)
    if cuda_available and vram_gb < 6.0:
        reasons.append(f"Insufficient GPU VRAM: {vram_gb} GB detected (minimum 6.0 GB required for 4-bit QLoRA).")

    is_compatible = (
        cuda_available
        and (vram_gb >= 6.0)
        and bitsandbytes_available
        and peft_available
    )

    return HardwareStatus(
        gpu_available=cuda_available,
        cuda_available=cuda_available,
        device_count=device_count,
        device_names=device_names,
        vram_gb=vram_gb,
        system_ram_gb=system_ram_gb,
        pytorch_version=torch_version,
        bitsandbytes_available=bitsandbytes_available,
        peft_available=peft_available,
        is_qlora_compatible=is_compatible,
        incompatibility_reasons=reasons,
    )


@lru_cache()
def get_finetuning_settings() -> FinetuningSettings:
    """Cached factory for finetuning settings."""
    return FinetuningSettings()
