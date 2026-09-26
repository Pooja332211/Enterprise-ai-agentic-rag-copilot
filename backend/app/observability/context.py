"""Request context and correlation ID management using ContextVar with secret redaction."""

import re
import uuid
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Union

# Context variable for thread/async-safe request correlation ID
_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Patterns for sensitive tokens and credentials
SENSITIVE_KEY_PATTERNS = {
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "bearer",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "private_key",
    "client_secret",
}

REGEX_SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{15,}"),
    re.compile(r"sk-[a-zA-Z0-9_\-]{15,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key[\s:=]+)['\"]?[a-zA-Z0-9_\-]{16,}['\"]?"),
    re.compile(r"(?i)(password[\s:=]+)['\"]?[^\s\"',]+['\"]?"),
]


def get_request_id() -> str:
    """Returns the current request correlation ID or generates a fallback ID."""
    req_id = _request_id_ctx.get()
    if not req_id:
        req_id = str(uuid.uuid4())
        _request_id_ctx.set(req_id)
    return req_id


def get_correlation_id() -> str:
    """Alias for get_request_id for enterprise correlation tracking."""
    return get_request_id()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Sets the request correlation ID for the current async task context."""
    val = request_id.strip() if request_id and request_id.strip() else str(uuid.uuid4())
    _request_id_ctx.set(val)
    return val


def clear_request_context() -> None:
    """Resets the request context for the current task."""
    _request_id_ctx.set(None)


def bind_request_context(request_id: Optional[str] = None):
    """Context manager for setting request context within a code block."""
    class _ContextScope:
        def __init__(self, req_id: Optional[str]):
            self._prev: Optional[str] = None
            self._target = req_id

        def __enter__(self):
            self._prev = _request_id_ctx.get()
            return set_request_id(self._target)

        def __exit__(self, exc_type, exc_val, exc_tb):
            _request_id_ctx.set(self._prev)

    return _ContextScope(request_id)


def redact_sensitive_data(data: Any) -> Any:
    """
    Recursively redacts API keys, bearer tokens, passwords, and authorization headers
    from strings, dictionaries, lists, and nested objects.
    """
    if data is None:
        return None

    if isinstance(data, str):
        sanitized = data
        for pattern in REGEX_SENSITIVE_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized

    if isinstance(data, dict):
        redacted_dict: Dict[str, Any] = {}
        for k, v in data.items():
            k_lower = str(k).lower().replace("-", "_")
            if any(sensitive in k_lower for sensitive in SENSITIVE_KEY_PATTERNS):
                redacted_dict[k] = "[REDACTED]"
            else:
                redacted_dict[k] = redact_sensitive_data(v)
        return redacted_dict

    if isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]

    if isinstance(data, tuple):
        return tuple(redact_sensitive_data(item) for item in data)

    if hasattr(data, "model_dump") and callable(getattr(data, "model_dump")):
        return redact_sensitive_data(data.model_dump())

    return data
