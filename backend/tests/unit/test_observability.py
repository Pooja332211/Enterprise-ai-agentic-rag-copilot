"""Unit tests for production observability: correlation IDs, redaction, timers, and metrics."""

import time
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend.app.main import app
from backend.app.observability.context import (
    bind_request_context,
    clear_request_context,
    get_request_id,
    redact_sensitive_data,
    set_request_id,
)
from backend.app.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    PrometheusRegistry,
    metrics_registry,
)
from backend.app.observability.timing import Timer, timed_block


def test_request_id_lifecycle():
    clear_request_context()
    # 1. Fallback generation
    req_id = get_request_id()
    assert req_id is not None
    assert len(req_id) > 10

    # 2. Explicit set
    custom_id = "test-req-abc-123"
    assigned = set_request_id(custom_id)
    assert assigned == custom_id
    assert get_request_id() == custom_id

    # 3. Context manager scoping
    scoped_id = "test-scoped-999"
    with bind_request_context(scoped_id):
        assert get_request_id() == scoped_id
    # Reverts to prior
    assert get_request_id() == custom_id


def test_redact_sensitive_data():
    # String redaction
    assert redact_sensitive_data("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9") == "[REDACTED]"
    assert redact_sensitive_data("sk-ant-api03-abcdefghijklmnop1234567890") == "[REDACTED]"

    # Dictionary redaction
    payload = {
        "user": "alice",
        "api_key": "secret_api_key_value_12345",
        "Authorization": "Bearer token12345678901234567890",
        "nested": {
            "password": "super_secret_pw",
            "normal_field": "visible_data",
        },
        "items": [
            {"access_token": "token_abc1234567890"},
            {"label": "public_tag"},
        ],
    }

    sanitized = redact_sensitive_data(payload)
    assert sanitized["user"] == "alice"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["Authorization"] == "[REDACTED]"
    assert sanitized["nested"]["password"] == "[REDACTED]"
    assert sanitized["nested"]["normal_field"] == "visible_data"
    assert sanitized["items"][0]["access_token"] == "[REDACTED]"
    assert sanitized["items"][1]["label"] == "public_tag"

    # Pydantic model redaction
    class AuthModel(BaseModel):
        token: str
        name: str

    model_inst = AuthModel(token="my_token_value_12345", name="bob")
    redacted_model = redact_sensitive_data(model_inst)
    assert redacted_model["token"] == "[REDACTED]"
    assert redacted_model["name"] == "bob"


def test_timing_utilities():
    timer = Timer()
    timer.start()
    time.sleep(0.01)
    elapsed_ms = timer.stop()
    assert elapsed_ms >= 5.0

    with timed_block() as t:
        time.sleep(0.01)
    assert t.elapsed_ms >= 5.0


def test_prometheus_registry_and_metrics():
    registry = PrometheusRegistry()

    # Counter
    c = registry.register_counter("test_counter", "Test counter metric", ["status"])
    c.inc(labels={"status": "200"})
    c.inc(labels={"status": "200"}, amount=2.0)
    c.inc(labels={"status": "500"})

    # Gauge
    g = registry.register_gauge("test_gauge", "Test gauge metric", ["type"])
    g.set(42.0, labels={"type": "memory"})

    # Histogram
    h = registry.register_histogram("test_hist", "Test histogram metric", ["path"], buckets=(0.05, 0.1, 0.5))
    h.observe(0.02, labels={"path": "/api"})
    h.observe(0.08, labels={"path": "/api"})
    h.observe(0.40, labels={"path": "/api"})

    exposition = registry.generate_exposition()

    assert "# HELP test_counter Test counter metric" in exposition
    assert "# TYPE test_counter counter" in exposition
    assert 'test_counter{status="200"} 3.0' in exposition
    assert 'test_counter{status="500"} 1.0' in exposition

    assert "# HELP test_gauge Test gauge metric" in exposition
    assert "# TYPE test_gauge gauge" in exposition
    assert 'test_gauge{type="memory"} 42.0' in exposition

    assert "# HELP test_hist Test histogram metric" in exposition
    assert "# TYPE test_hist histogram" in exposition
    assert 'test_hist_bucket{path="/api",le="0.05"} 1' in exposition
    assert 'test_hist_bucket{path="/api",le="0.1"} 2' in exposition
    assert 'test_hist_bucket{path="/api",le="+Inf"} 3' in exposition
    assert 'test_hist_count{path="/api"} 3' in exposition


def test_fastapi_metrics_endpoint_and_headers():
    client = TestClient(app)

    # Call /health with explicit X-Request-ID
    req_id = "test-custom-trace-uuid-42"
    resp = client.get("/health", headers={"X-Request-ID": req_id})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == req_id
    assert "X-Process-Time" in resp.headers

    # Call /metrics
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "text/plain" in metrics_resp.headers.get("content-type", "")
    assert "copilot_requests_total" in metrics_resp.text
    assert "copilot_request_duration_seconds" in metrics_resp.text
