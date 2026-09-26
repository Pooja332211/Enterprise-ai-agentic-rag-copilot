"""Abstract interface for application caching layer."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseCache(ABC):
    """Abstract interface defining the cache contract for deterministic operations."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieves value for key, returning None if absent or expired."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Stores value for key with optional expiration in seconds."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Deletes key from cache, returning True if key existed."""
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clears all entries, returning count of evicted items."""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Returns cache telemetry (hits, misses, size, evictions)."""
        pass
