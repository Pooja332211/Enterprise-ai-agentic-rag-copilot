"""Unit tests for Milestone 7: QLoRA Domain Adaptation Experiment.

Tests cover:
- Finetuning dataset schemas and JSONL parsing
- Train / validation split separation
- Strict leakage detection and contamination rejection
- QLoRA configuration and hyperparameter validation
- Prompt formatting for instruction tuning
- Hardware inspection and compatibility guards
- Experiment metadata and ablation comparison serialization
"""

import json
import pytest
from pathlib import Path
from typing import List

from backend.app.finetuning.config import check_gpu_readiness, get_finetuning_settings
from backend.app.finetuning.dataset_generator import build_finetuning_dataset
from backend.app.finetuning.leakage_detector import (
    DataLeakageError,
    LeakageDetector,
)
from backend.app.finetuning.schemas import (
    AblationComparison,
    CitationReference,
    FinetuningExample,
    HardwareStatus,
    InstructionDataset,
    QLoRAConfig,
    TrainingRunMetadata,
)
from backend.app.finetuning.trainer import (
    HardwareIncompatibilityError,
    QLoRATrainer,
)


def test_finetuning_example_schema():
    """Verifies FinetuningExample schema creation and serialization."""
    example = FinetuningExample(
        example_id="ft_test_01",
        system_instruction="System prompt",
        query="What is the test question?",
        context_evidence="Test evidence text.",
        grounded_answer="This is a test answer [Doc: test.pdf].",
        citations=[
            CitationReference(
                document_filename="test.pdf",
                page_number=1,
                section_title="Intro",
                evidence_text="Test evidence text.",
            )
        ],
        is_abstention=False,
        split="train",
    )
    assert example.example_id == "ft_test_01"
    assert len(example.citations) == 1
    assert example.citations[0].document_filename == "test.pdf"

    # Test round-trip JSON serialization
    serialized = example.model_dump_json()
    parsed = FinetuningExample(**json.loads(serialized))
    assert parsed.query == example.query
    assert parsed.grounded_answer == example.grounded_answer


def test_dataset_generator_split_and_separation():
    """Verifies that generated fine-tuning dataset maintains strict 80/20 train/validation split."""
    dataset = build_finetuning_dataset()
    assert isinstance(dataset, InstructionDataset)
    assert len(dataset.examples) == 30
    assert dataset.train_count == 24
    assert dataset.validation_count == 6

    # Verify unique example IDs
    ids = [ex.example_id for ex in dataset.examples]
    assert len(ids) == len(set(ids)), "Example IDs must be unique"

    # Verify all examples have non-empty query and context
    for ex in dataset.examples:
        assert len(ex.query.strip()) > 0
        assert len(ex.context_evidence.strip()) > 0
        assert len(ex.grounded_answer.strip()) > 0
        assert ex.split in ("train", "validation")


def test_leakage_detector_clean_dataset():
    """Verifies that clean dataset passes leakage detection with zero violations."""
    dataset = build_finetuning_dataset()
    detector = LeakageDetector("data/evaluation/benchmark_v1.json")
    report = detector.validate_dataset(dataset.examples, raise_on_violation=True)

    assert report["is_clean"] is True
    assert report["violations_count"] == 0
    assert report["total_examples"] == 30
    assert report["benchmark_queries_checked"] == 18
    assert len(report["dataset_sha256"]) == 64


def test_leakage_detector_catches_exact_query():
    """Verifies that LeakageDetector rejects an example containing an exact benchmark query."""
    detector = LeakageDetector("data/evaluation/benchmark_v1.json")

    # Construct contaminated example using exact query from q_fact_01
    leaked_example = FinetuningExample(
        example_id="ft_leaked_01",
        system_instruction="system",
        query="What is the accuracy of Hybrid RAG reported in the benchmark results table?",
        context_evidence="some context",
        grounded_answer="some answer",
        split="train",
    )

    violations = detector.inspect_example(leaked_example)
    assert len(violations) > 0
    assert any("Exact query match" in v for v in violations)

    # validate_dataset should raise DataLeakageError
    with pytest.raises(DataLeakageError) as exc_info:
        detector.validate_dataset([leaked_example], raise_on_violation=True)
    assert len(exc_info.value.violations) > 0


def test_leakage_detector_catches_verbatim_answer():
    """Verifies that LeakageDetector rejects an example containing verbatim benchmark answer."""
    detector = LeakageDetector("data/evaluation/benchmark_v1.json")

    # Construct contaminated example with expected answer from q_fact_02
    leaked_example = FinetuningExample(
        example_id="ft_leaked_02",
        system_instruction="system",
        query="Tell me about the metrics in the docx report",
        context_evidence="evidence",
        grounded_answer="The NDCG@10 score reported is 0.89 according to the table.",
        split="train",
    )

    violations = detector.inspect_example(leaked_example)
    assert len(violations) > 0
    assert any("benchmark answer" in v for v in violations)


def test_qlora_config_validation():
    """Verifies QLoRAConfig hyperparameters and default settings."""
    config = QLoRAConfig()
    assert config.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.quantization_bits == 4
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.lora_r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert "q_proj" in config.target_modules
    assert "v_proj" in config.target_modules
    assert config.random_seed == 42


def test_hardware_readiness_inspection():
    """Verifies that check_gpu_readiness returns a structured HardwareStatus."""
    hw = check_gpu_readiness()
    assert isinstance(hw, HardwareStatus)
    assert isinstance(hw.gpu_available, bool)
    assert isinstance(hw.cuda_available, bool)
    assert isinstance(hw.system_ram_gb, float)
    assert isinstance(hw.incompatibility_reasons, list)


def test_trainer_prompt_formatting():
    """Verifies ChatML prompt formatting for causal LM fine-tuning."""
    trainer = QLoRATrainer()
    example = FinetuningExample(
        example_id="ft_fmt_01",
        system_instruction="You are an enterprise research assistant.",
        query="What is the architecture?",
        context_evidence="Hybrid dense and sparse search.",
        grounded_answer="The architecture uses hybrid search.",
        split="train",
    )
    prompt = trainer.format_prompt(example)
    assert "<|im_start|>system\nYou are an enterprise research assistant.<|im_end|>" in prompt
    assert "<|im_start|>user\nContext Evidence:\nHybrid dense and sparse search.\n\nQuestion:\nWhat is the architecture?<|im_end|>" in prompt
    assert "<|im_start|>assistant\nThe architecture uses hybrid search.<|im_end|>" in prompt


def test_trainer_hardware_guard():
    """Verifies that QLoRATrainer safely handles incompatible hardware."""
    trainer = QLoRATrainer()
    if not trainer.hardware.is_qlora_compatible:
        # Without raising, should return metadata indicating failure reason
        metadata = trainer.train(raise_on_incompatible=False)
        assert metadata.training_status == "failed_hardware_incompatibility"
        assert metadata.error_message is not None

        # With raising, should raise HardwareIncompatibilityError
        with pytest.raises(HardwareIncompatibilityError):
            trainer.train(raise_on_incompatible=True)


def test_experiment_metadata_and_ablation_serialization():
    """Verifies that TrainingRunMetadata and AblationComparison serialize cleanly to JSON."""
    hw = HardwareStatus(gpu_available=False, cuda_available=False, system_ram_gb=8.0)
    config = QLoRAConfig()
    metadata = TrainingRunMetadata(
        run_id="run_test_123",
        config=config,
        hardware=hw,
        training_status="failed_hardware_incompatibility",
        error_message="Requires NVIDIA GPU with CUDA.",
    )
    meta_json = metadata.model_dump_json()
    assert "run_test_123" in meta_json
    assert "failed_hardware_incompatibility" in meta_json

    comparison = AblationComparison(
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        adapted_model_name="Qwen/Qwen2.5-1.5B-Instruct + QLoRA",
        adapted_model_available=False,
        comparison_table=[
            {
                "metric": "Evidence Grounding Score",
                "base_model": "1.0000",
                "qlora_model": "N/A (Pending GPU training)",
                "description": "Evidence grounding",
            }
        ],
        notes="Test comparison notes",
    )
    comp_json = comparison.model_dump_json()
    assert "Evidence Grounding Score" in comp_json
    assert "Pending GPU training" in comp_json
