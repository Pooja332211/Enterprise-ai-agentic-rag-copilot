"""Native, dependency-free Prometheus-compatible metrics registry and exposition formatter."""

import threading
from typing import Dict, List, Optional, Tuple


class Counter:
    """Prometheus-compatible Counter metric (monotonically increasing)."""

    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = tuple(label_names or [])
        self._values: Dict[Tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increments counter by amount (must be non-negative)."""
        if amount < 0:
            raise ValueError("Counter increments must be non-negative.")
        key = self._get_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Returns current counter value for label combination."""
        key = self._get_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _get_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        labels = labels or {}
        return tuple(str(labels.get(name, "")) for name in self.label_names)

    def collect(self) -> List[str]:
        """Formats counter entries in Prometheus exposition format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            if not self._values:
                # Output 0.0 default if no labels
                if not self.label_names:
                    lines.append(f"{self.name} 0.0")
            for key, val in sorted(self._values.items()):
                if self.label_names:
                    label_str = ",".join(f'{k}="{v}"' for k, v in zip(self.label_names, key))
                    lines.append(f"{self.name}{{{label_str}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return lines


class Gauge:
    """Prometheus-compatible Gauge metric (arbitrary up/down values)."""

    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = tuple(label_names or [])
        self._values: Dict[Tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Sets gauge to value."""
        key = self._get_key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increments gauge."""
        key = self._get_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrements gauge."""
        key = self._get_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Gets gauge value."""
        key = self._get_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _get_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        labels = labels or {}
        return tuple(str(labels.get(name, "")) for name in self.label_names)

    def collect(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            if not self._values and not self.label_names:
                lines.append(f"{self.name} 0.0")
            for key, val in sorted(self._values.items()):
                if self.label_names:
                    label_str = ",".join(f'{k}="{v}"' for k, v in zip(self.label_names, key))
                    lines.append(f"{self.name}{{{label_str}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return lines


class Histogram:
    """Prometheus-compatible Histogram metric tracking sample distributions into cumulative buckets."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = tuple(label_names or [])
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[Tuple[str, ...], int] = {}
        self._sums: Dict[Tuple[str, ...], float] = {}
        self._bucket_counts: Dict[Tuple[str, ...], Dict[float, int]] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Records an observed sample in seconds or units."""
        key = self._get_key(labels)
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            self._sums[key] = self._sums.get(key, 0.0) + float(value)

            if key not in self._bucket_counts:
                self._bucket_counts[key] = {b: 0 for b in self.buckets}

            for b in self.buckets:
                if value <= b:
                    self._bucket_counts[key][b] += 1

    def _get_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        labels = labels or {}
        return tuple(str(labels.get(name, "")) for name in self.label_names)

    def collect(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for key in sorted(self._counts.keys()):
                base_labels = []
                if self.label_names:
                    base_labels = [f'{k}="{v}"' for k, v in zip(self.label_names, key)]

                # Cumulative buckets
                b_counts = self._bucket_counts.get(key, {})
                cumulative = 0
                for b in self.buckets:
                    cumulative = b_counts.get(b, 0)
                    lbls = base_labels + [f'le="{b}"']
                    lines.append(f"{self.name}_bucket{{{','.join(lbls)}}} {cumulative}")

                # +Inf bucket
                total_cnt = self._counts.get(key, 0)
                inf_lbls = base_labels + ['le="+Inf"']
                lines.append(f"{self.name}_bucket{{{','.join(inf_lbls)}}} {total_cnt}")

                # Sum and count
                lbl_suffix = f"{{{','.join(base_labels)}}}" if base_labels else ""
                lines.append(f"{self.name}_sum{lbl_suffix} {self._sums.get(key, 0.0):.6f}")
                lines.append(f"{self.name}_count{lbl_suffix} {total_cnt}")
        return lines


class PrometheusRegistry:
    """Central metrics registry aggregating application counters, histograms, and gauges."""

    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        # HTTP Request metrics
        self.requests_total = self.register_counter(
            "copilot_requests_total",
            "Total number of HTTP requests processed by endpoint and status.",
            ["endpoint", "method", "status"],
        )
        self.request_duration = self.register_histogram(
            "copilot_request_duration_seconds",
            "Total HTTP request processing duration in seconds.",
            ["endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self.request_errors = self.register_counter(
            "copilot_request_errors_total",
            "Total count of unhandled or 5xx application errors.",
            ["endpoint", "error_type"],
        )

        # Pipeline stage latencies
        self.retrieval_duration = self.register_histogram(
            "copilot_retrieval_duration_seconds",
            "Retrieval execution latency in seconds.",
            ["mode"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self.reranker_duration = self.register_histogram(
            "copilot_reranker_duration_seconds",
            "Cross-encoder reranking latency in seconds.",
            ["model"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        )
        self.generation_duration = self.register_histogram(
            "copilot_generation_duration_seconds",
            "LLM answer generation latency in seconds.",
            ["provider"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self.citation_verification_duration = self.register_histogram(
            "copilot_citation_verification_duration_seconds",
            "Citation verification and audit latency in seconds.",
            ["status"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
        )

        # Agent workflow metrics
        self.agent_iterations = self.register_counter(
            "copilot_agent_iterations_total",
            "Total agent loop iterations executed.",
            ["query_type"],
        )
        self.retrieval_calls = self.register_counter(
            "copilot_retrieval_calls_total",
            "Total underlying retrieval calls executed across workflows.",
            ["source"],
        )
        self.abstentions = self.register_counter(
            "copilot_abstentions_total",
            "Total number of clean abstentions on ungrounded/insufficient queries.",
            ["reason"],
        )

        # Quality & citation metrics
        self.citation_failures = self.register_counter(
            "copilot_citation_failures_total",
            "Total citations that failed validation or chunk mapping.",
            ["reason"],
        )
        self.unsupported_claims = self.register_counter(
            "copilot_unsupported_claims_total",
            "Total claims marked unsupported during citation audit.",
        )

        # Cache metrics
        self.cache_hits = self.register_counter(
            "copilot_cache_hits_total",
            "Total cache hits for deterministic operations.",
            ["cache_type"],
        )
        self.cache_misses = self.register_counter(
            "copilot_cache_misses_total",
            "Total cache misses for deterministic operations.",
            ["cache_type"],
        )
        self.cache_evictions = self.register_counter(
            "copilot_cache_evictions_total",
            "Total cache entries evicted due to TTL or size capacity.",
            ["cache_type"],
        )
        self.cache_invalidations = self.register_counter(
            "copilot_cache_invalidations_total",
            "Total cache entries invalidated due to document/index lifecycle changes.",
            ["cache_type"],
        )
        self.cache_size = self.register_gauge(
            "copilot_cache_size",
            "Current number of entries stored in cache.",
            ["cache_type"],
        )

    def register_counter(self, name: str, description: str, label_names: Optional[List[str]] = None) -> Counter:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description, label_names)
            return self._metrics[name]

    def register_gauge(self, name: str, description: str, label_names: Optional[List[str]] = None) -> Gauge:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description, label_names)
            return self._metrics[name]

    def register_histogram(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description, label_names, buckets)
            return self._metrics[name]

    def generate_exposition(self) -> str:
        """Renders complete registry in Prometheus text exposition format."""
        output_lines: List[str] = []
        with self._lock:
            for metric in self._metrics.values():
                output_lines.extend(metric.collect())
                output_lines.append("")
        return "\n".join(output_lines) + "\n"

    def reset(self) -> None:
        """Resets all metrics to clean state (primarily for isolated test fixtures)."""
        with self._lock:
            self._metrics.clear()
            self._register_default_metrics()


# Global Singleton Registry
_metrics_registry = PrometheusRegistry()


def get_metrics_registry() -> PrometheusRegistry:
    """Returns singleton PrometheusRegistry instance."""
    return _metrics_registry


metrics_registry = _metrics_registry
