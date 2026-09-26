"""Unit tests for Milestone 10 security hardening, path traversal protection, and prompt isolation."""

import pytest
from pydantic import ValidationError
from backend.app.models.schemas import SearchRequest, ResearchRequest, AgentResearchRequest
from backend.app.observability.context import redact_sensitive_data
from backend.app.services.grounded_generation_service import GROUNDED_SYSTEM_PROMPT
from backend.app.services.storage_service import LocalStorageService


def test_path_traversal_filename_sanitization():
    """Verify that path traversal sequences in filenames are sanitized."""
    raw_names = [
        "../../etc/passwd",
        "..\\..\\windows\\system32\\cmd.exe",
        "/etc/shadow",
        "nested/path/to/file.pdf",
        "valid_doc.md",
    ]
    for name in raw_names:
        sanitized = LocalStorageService.sanitize_filename(name)
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ".." not in sanitized


def test_storage_rejects_path_traversal_resolution(tmp_path):
    """Verify that attempting to access a path outside upload directory raises ValueError."""
    class MockSettings:
        UPLOAD_DIR = str(tmp_path / "uploads")

    storage = LocalStorageService(settings=MockSettings())
    with pytest.raises(ValueError, match="Security error: Invalid storage path"):
        storage._resolve_and_verify_path("../../../etc/passwd")


def test_system_prompt_contains_security_isolation_notice():
    """Verify that grounded generation system prompt contains strict untrusted context notice."""
    assert "<context>" in GROUNDED_SYSTEM_PROMPT
    assert "SECURITY NOTICE" in GROUNDED_SYSTEM_PROMPT
    assert "untrusted data" in GROUNDED_SYSTEM_PROMPT


def test_query_input_length_bounding():
    """Verify that excessively large queries exceeding 2000 characters are rejected."""
    oversized_query = "x" * 2001

    with pytest.raises(ValidationError):
        SearchRequest(query=oversized_query)

    with pytest.raises(ValidationError):
        ResearchRequest(query=oversized_query)

    with pytest.raises(ValidationError):
        AgentResearchRequest(query=oversized_query)


def test_sensitive_data_redaction():
    """Verify that sensitive keys, passwords, and tokens are redacted."""
    sensitive_dict = {
        "api_key": "sk-1234567890abcdef",
        "authorization": "Bearer secret-token-xyz",
        "password": "production_password_123",
        "user_query": "What is the policy?",
    }
    redacted = redact_sensitive_data(sensitive_dict)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["authorization"] == "[REDACTED]"
    assert redacted["password"] == "[REDACTED]"
    assert redacted["user_query"] == "What is the policy?"
