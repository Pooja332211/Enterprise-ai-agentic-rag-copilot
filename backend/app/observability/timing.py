"""Reusable latency timing utilities, context managers, and function decorators."""

import functools
import inspect
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


class Timer:
    """High-resolution wall-clock timer capturing start, stop, and millisecond duration."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self._start_perf: Optional[float] = None
        self._stop_perf: Optional[float] = None
        self.start_timestamp: Optional[str] = None
        self.end_timestamp: Optional[str] = None
        self.duration_ms: float = 0.0

    def start(self) -> "Timer":
        """Starts timer."""
        self._start_perf = time.perf_counter()
        self.start_timestamp = datetime.now(timezone.utc).isoformat()
        return self

    def stop(self) -> float:
        """Stops timer and calculates duration in milliseconds."""
        if self._start_perf is None:
            return 0.0
        self._stop_perf = time.perf_counter()
        self.end_timestamp = datetime.now(timezone.utc).isoformat()
        self.duration_ms = round((self._stop_perf - self._start_perf) * 1000.0, 3)
        return self.duration_ms

    @property
    def elapsed_ms(self) -> float:
        """Returns current elapsed milliseconds without stopping."""
        if self._start_perf is None:
            return 0.0
        current = self._stop_perf or time.perf_counter()
        return round((current - self._start_perf) * 1000.0, 3)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes timing metadata to structured dictionary."""
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "start_time": self.start_timestamp,
            "end_time": self.end_timestamp,
        }


@contextmanager
def timed_block(name: str = "operation", metrics_callback: Optional[Callable[[str, float], None]] = None):
    """
    Context manager measuring duration of an execution block.
    Yields Timer instance and invokes optional metric callback upon exit.
    """
    timer = Timer(name=name)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
        if metrics_callback is not None:
            try:
                metrics_callback(name, timer.duration_ms)
            except Exception:
                pass


def timed(name: Optional[str] = None, metrics_callback: Optional[Callable[[str, float], None]] = None):
    """
    Decorator for synchronous and asynchronous functions capturing latency.
    Attaches '_last_duration_ms' to function object or passes timing to callback.
    """
    def decorator(fn: Callable):
        op_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    async_wrapper._last_duration_ms = duration_ms
                    if metrics_callback:
                        metrics_callback(op_name, duration_ms)

            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    sync_wrapper._last_duration_ms = duration_ms
                    if metrics_callback:
                        metrics_callback(op_name, duration_ms)

            return sync_wrapper

    return decorator
