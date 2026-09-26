"""Fine-tuning package for QLoRA domain adaptation experiments.

Provides dataset management, strict data leakage detection, QLoRA training
pipeline orchestration, and comparative evaluation against base models.
"""

from backend.app.finetuning.schemas import (
    FinetuningExample,
    InstructionDataset,
    QLoRAConfig,
    HardwareStatus,
    TrainingRunMetadata,
    AblationComparison,
)
from backend.app.finetuning.config import FinetuningSettings, get_finetuning_settings
from backend.app.finetuning.dataset_generator import build_finetuning_dataset
from backend.app.finetuning.leakage_detector import LeakageDetector, DataLeakageError
from backend.app.finetuning.trainer import QLoRATrainer
from backend.app.finetuning.evaluator import QLoRAEvaluator

__all__ = [
    "FinetuningExample",
    "InstructionDataset",
    "QLoRAConfig",
    "HardwareStatus",
    "TrainingRunMetadata",
    "AblationComparison",
    "FinetuningSettings",
    "get_finetuning_settings",
    "build_finetuning_dataset",
    "LeakageDetector",
    "DataLeakageError",
    "QLoRATrainer",
    "QLoRAEvaluator",
]
