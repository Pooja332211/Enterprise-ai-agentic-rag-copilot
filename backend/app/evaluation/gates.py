"""Evaluation gates policy enforcing quality and latency thresholds in CI/CD and release pipelines."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationGate(BaseModel):
    """Definition of a single evaluation gate rule."""
    name: str
    metric_key: str
    operator: str  # '>=', '<=', '>', '<', '=='
    threshold: float
    description: str
    critical: bool = True


class EvaluationGateResult(BaseModel):
    """Result of evaluating a specific metric against its threshold."""
    gate_name: str
    metric_key: str
    operator: str
    threshold: float
    actual_value: float
    passed: bool
    critical: bool
    message: str


class EvaluationReportSummary(BaseModel):
    """Overall evaluation gates verdict."""
    all_passed: bool
    critical_passed: bool
    total_gates: int
    passed_gates: int
    failed_gates: int
    results: List[EvaluationGateResult]


# Default Production Quality and Performance Gates
DEFAULT_GATES: List[EvaluationGate] = [
    EvaluationGate(
        name="Retrieval Recall@5",
        metric_key="recall_at_5",
        operator=">=",
        threshold=0.80,
        description="Candidate retrieval must find relevant chunks in top 5 for >= 80% of benchmark queries.",
        critical=True,
    ),
    EvaluationGate(
        name="Retrieval MRR",
        metric_key="mrr",
        operator=">=",
        threshold=0.70,
        description="Mean Reciprocal Rank must be >= 0.70.",
        critical=True,
    ),
    EvaluationGate(
        name="Retrieval NDCG@5",
        metric_key="ndcg_at_5",
        operator=">=",
        threshold=0.70,
        description="Normalized Discounted Cumulative Gain at rank 5 must be >= 0.70.",
        critical=False,
    ),
    EvaluationGate(
        name="Answer Grounding Accuracy",
        metric_key="grounding_accuracy",
        operator=">=",
        threshold=0.90,
        description="At least 90% of answers must be strictly grounded in verified evidence.",
        critical=True,
    ),
    EvaluationGate(
        name="Citation Validity Rate",
        metric_key="citation_validity",
        operator=">=",
        threshold=0.95,
        description="At least 95% of claims must map cleanly to retrieved chunk provenance.",
        critical=True,
    ),
    EvaluationGate(
        name="Abstention Accuracy",
        metric_key="abstention_accuracy",
        operator=">=",
        threshold=0.90,
        description="Agent must cleanly refuse unanswerable/out-of-domain queries without hallucination.",
        critical=True,
    ),
    EvaluationGate(
        name="Agent p95 Latency",
        metric_key="p95_latency_ms",
        operator="<=",
        threshold=12000.0,
        description="Agent multi-hop research p95 latency must remain under 12,000ms on CPU.",
        critical=False,
    ),
]


def evaluate_gates(
    metrics: Dict[str, Any],
    custom_gates: Optional[List[EvaluationGate]] = None,
) -> EvaluationReportSummary:
    """
    Evaluates computed system metrics against defined gates.
    Returns structured pass/fail report for continuous integration.
    """
    gates = custom_gates if custom_gates is not None else DEFAULT_GATES
    results: List[EvaluationGateResult] = []

    for gate in gates:
        val = metrics.get(gate.metric_key)
        if val is None:
            # Metric missing from evaluation payload
            results.append(
                EvaluationGateResult(
                    gate_name=gate.name,
                    metric_key=gate.metric_key,
                    operator=gate.operator,
                    threshold=gate.threshold,
                    actual_value=0.0,
                    passed=False,
                    critical=gate.critical,
                    message=f"Metric '{gate.metric_key}' was not found in evaluation results.",
                )
            )
            continue

        actual_val = float(val)
        passed = False
        if gate.operator == ">=":
            passed = actual_val >= gate.threshold
        elif gate.operator == "<=":
            passed = actual_val <= gate.threshold
        elif gate.operator == ">":
            passed = actual_val > gate.threshold
        elif gate.operator == "<":
            passed = actual_val < gate.threshold
        elif gate.operator == "==":
            passed = abs(actual_val - gate.threshold) < 1e-6

        status_word = "PASSED" if passed else "FAILED"
        msg = (
            f"[{status_word}] {gate.name}: actual={actual_val:.4f} {gate.operator} "
            f"threshold={gate.threshold:.4f}"
        )

        results.append(
            EvaluationGateResult(
                gate_name=gate.name,
                metric_key=gate.metric_key,
                operator=gate.operator,
                threshold=gate.threshold,
                actual_value=actual_val,
                passed=passed,
                critical=gate.critical,
                message=msg,
            )
        )

    all_passed = all(r.passed for r in results)
    critical_passed = all(r.passed for r in results if r.critical)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    return EvaluationReportSummary(
        all_passed=all_passed,
        critical_passed=critical_passed,
        total_gates=len(results),
        passed_gates=passed_count,
        failed_gates=failed_count,
        results=results,
    )
