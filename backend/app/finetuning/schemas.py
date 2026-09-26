"""Data contracts and Pydantic schemas for QLoRA fine-tuning and domain adaptation."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HardwareStatus(BaseModel):
    """Snapshot of system hardware capabilities for QLoRA training."""
    gpu_available: bool = False
    cuda_available: bool = False
    device_count: int = 0
    device_names: List[str] = Field(default_factory=list)
    vram_gb: float = 0.0
    system_ram_gb: float = 0.0
    pytorch_version: str = "unknown"
    bitsandbytes_available: bool = False
    peft_available: bool = False
    is_qlora_compatible: bool = False
    incompatibility_reasons: List[str] = Field(default_factory=list)


class CitationReference(BaseModel):
    """Document citation reference attached to an instruction example."""
    document_filename: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    evidence_text: str


class FinetuningExample(BaseModel):
    """Single instruction-tuning example for domain adaptation."""
    example_id: str
    system_instruction: str
    query: str
    context_evidence: str
    grounded_answer: str
    citations: List[CitationReference] = Field(default_factory=list)
    is_abstention: bool = False
    split: str = "train"  # 'train' or 'validation'
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InstructionDataset(BaseModel):
    """Container for the version-controlled instruction dataset."""
    dataset_name: str = "Enterprise AI Copilot Fine-tuning Dataset"
    version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = "Domain-adaptation instruction dataset for enterprise grounded RAG."
    source_corpus: List[str] = Field(default_factory=list)
    train_count: int = 0
    validation_count: int = 0
    examples: List[FinetuningExample] = Field(default_factory=list)


class QLoRAConfig(BaseModel):
    """Hyperparameter and hardware configuration for QLoRA fine-tuning."""
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "data/finetuning/checkpoints"
    train_data_path: str = "data/finetuning/train.jsonl"
    eval_data_path: str = "data/finetuning/validation.jsonl"
    max_seq_length: int = 1024
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    quantization_bits: int = 4
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    random_seed: int = 42


class TrainingRunMetadata(BaseModel):
    """Metadata recorded for every QLoRA training experiment."""
    run_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: QLoRAConfig
    hardware: HardwareStatus
    dataset_version: str = "1.0.0"
    num_train_examples: int = 0
    num_eval_examples: int = 0
    training_status: str = "pending"  # 'completed', 'failed_hardware_incompatibility', 'pending_gpu'
    training_duration_seconds: float = 0.0
    total_steps: int = 0
    final_loss: Optional[float] = None
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None


class AblationComparison(BaseModel):
    """Head-to-head comparison between base instruct model and QLoRA adapted model."""
    dataset_version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    base_model_name: str
    adapted_model_name: str
    adapted_model_available: bool = False
    comparison_table: List[Dict[str, Any]] = Field(default_factory=list)
    notes: str = ""
