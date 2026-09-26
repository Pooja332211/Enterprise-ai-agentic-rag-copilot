"""Structured audit event models and logging utilities with secret redaction."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.observability.context import get_correlation_id, redact_sensitive_data
from backend.app.utils.logger import logger


class AuditEvent(BaseModel):
    """Base schema for structured audit events."""
    event_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str = Field(default_factory=get_correlation_id)
    duration_ms: Optional[float] = None
    status: str = "success"
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def emit_audit_event(
    event_type: str,
    status: str = "success",
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AuditEvent:
    """Creates, redacts, and logs a structured enterprise audit event."""
    safe_metadata = redact_sensitive_data(metadata or {})
    event = AuditEvent(
        event_type=event_type,
        status=status,
        duration_ms=round(duration_ms, 3) if duration_ms is not None else None,
        error=str(error) if error else None,
        metadata=safe_metadata,
    )
    log_msg = f"[AUDIT] {event.event_type} status={event.status} req_id={event.correlation_id}"
    if event.duration_ms is not None:
        log_msg += f" latency={event.duration_ms}ms"
    if event.error:
        log_msg += f" error={event.error}"

    logger.info(log_msg)
    return event
