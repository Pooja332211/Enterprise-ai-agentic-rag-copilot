"""Leakage protection engine ensuring strict isolation between fine-tuning and evaluation data."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from backend.app.finetuning.schemas import FinetuningExample


class DataLeakageError(Exception):
    """Raised when data contamination or evaluation benchmark leakage is detected."""
    def __init__(self, message: str, violations: List[str]):
        super().__init__(message)
        self.violations = violations


class LeakageDetector:
    """Validates that instruction tuning examples do not contaminate the evaluation benchmark."""

    def __init__(self, benchmark_path: str = "data/evaluation/benchmark_v1.json"):
        self.benchmark_path = Path(benchmark_path)
        self.benchmark_queries: List[Dict[str, Any]] = []
        self.benchmark_hashes: Set[str] = set()
        self.benchmark_query_tokens: List[Tuple[str, Set[str], Optional[str]]] = []
        self._load_benchmark()

    def _normalize_text(self, text: str) -> str:
        """Lowercases, removes punctuation, and normalizes whitespace."""
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(clean.split())

    def _tokenize(self, text: str) -> Set[str]:
        """Splits normalized text into a set of distinct word tokens."""
        return set(self._normalize_text(text).split())

    def _compute_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Computes Jaccard token similarity coefficient."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def _load_benchmark(self) -> None:
        """Loads and pre-indexes the evaluation benchmark queries."""
        if not self.benchmark_path.exists():
            # If benchmark does not exist (e.g. running in isolated test fixture), leave empty
            return

        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data.get("queries", [])
        self.benchmark_queries = queries

        for q in queries:
            qid = q.get("query_id", "")
            qtext = q.get("query", "")
            expected_ans = q.get("expected_answer")
            tokens = self._tokenize(qtext)
            self.benchmark_query_tokens.append((qid, tokens, expected_ans))
            h = hashlib.sha256(self._normalize_text(qtext).encode("utf-8")).hexdigest()
            self.benchmark_hashes.add(h)

    def compute_dataset_hash(self, examples: List[FinetuningExample]) -> str:
        """Computes deterministic SHA-256 hash over all instruction examples."""
        hasher = hashlib.sha256()
        for ex in sorted(examples, key=lambda x: x.example_id):
            hasher.update(ex.example_id.encode("utf-8"))
            hasher.update(ex.query.encode("utf-8"))
            hasher.update(ex.grounded_answer.encode("utf-8"))
        return hasher.hexdigest()

    def inspect_example(self, example: FinetuningExample, jaccard_threshold: float = 0.70) -> List[str]:
        """Checks a single fine-tuning example for contamination against the benchmark.

        Returns a list of violation descriptions (empty if clean).
        """
        violations: List[str] = []
        norm_query = self._normalize_text(example.query)
        query_hash = hashlib.sha256(norm_query.encode("utf-8")).hexdigest()

        # 1. Exact query hash match
        if query_hash in self.benchmark_hashes:
            violations.append(
                f"Example '{example.example_id}': Exact query match detected with an evaluation benchmark query!"
            )

        ex_tokens = self._tokenize(example.query)
        ex_ans_norm = self._normalize_text(example.grounded_answer)

        for qid, bm_tokens, expected_ans in self.benchmark_query_tokens:
            # 2. High Jaccard similarity (potential paraphrase or near-duplicate)
            sim = self._compute_jaccard(ex_tokens, bm_tokens)
            if sim >= jaccard_threshold:
                violations.append(
                    f"Example '{example.example_id}': Query token similarity {sim:.2f} >= {jaccard_threshold:.2f} "
                    f"with benchmark query '{qid}'!"
                )

            # 3. Direct answer leakage for answerable queries
            if expected_ans and not example.is_abstention:
                norm_expected = self._normalize_text(expected_ans)
                # If expected benchmark answer is substantial and verbatim inside the training answer
                if len(norm_expected) > 20 and norm_expected in ex_ans_norm:
                    violations.append(
                        f"Example '{example.example_id}': Verbatim benchmark answer for '{qid}' found in training answer!"
                    )

        return violations

    def validate_dataset(
        self,
        examples: List[FinetuningExample],
        jaccard_threshold: float = 0.70,
        raise_on_violation: bool = True,
    ) -> Dict[str, Any]:
        """Validates an entire dataset of instruction examples for leakage.

        If raise_on_violation is True, raises DataLeakageError upon finding any contamination.
        """
        all_violations: List[str] = []
        unique_queries: Set[str] = set()
        duplicate_queries: List[str] = []

        for ex in examples:
            # Check internal duplicate queries within training set
            norm_q = self._normalize_text(ex.query)
            if norm_q in unique_queries:
                duplicate_queries.append(ex.example_id)
            else:
                unique_queries.add(norm_q)

            # Check leakage against benchmark
            violations = self.inspect_example(ex, jaccard_threshold=jaccard_threshold)
            all_violations.extend(violations)

        if duplicate_queries:
            all_violations.append(
                f"Internal duplicate queries detected in dataset: {duplicate_queries}"
            )

        dataset_hash = self.compute_dataset_hash(examples)

        report = {
            "is_clean": len(all_violations) == 0,
            "total_examples": len(examples),
            "benchmark_queries_checked": len(self.benchmark_query_tokens),
            "violations_count": len(all_violations),
            "violations": all_violations,
            "dataset_sha256": dataset_hash,
        }

        if not report["is_clean"] and raise_on_violation:
            raise DataLeakageError(
                f"Data leakage detected! Found {len(all_violations)} violations contaminating evaluation benchmark.",
                violations=all_violations,
            )

        return report
