"""Thread-safe bounded in-memory LRU cache with TTL expiration and Prometheus metrics integration."""

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from backend.app.cache.base import BaseCache
from backend.app.observability.metrics import get_metrics_registry


class InMemoryLRUCache(BaseCache):
    """
    Thread-safe, bounded Least-Recently-Used (LRU) cache with per-item TTL expiration.
    Automatically increments Prometheus metrics for hits, misses, evictions, and size.
    """

    def __init__(self, max_size: int = 10000, default_ttl_seconds: int = 3600, name: str = "default"):
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self.name = name
        # Storage: key -> (value, expiration_timestamp_or_None)
        self._cache: OrderedDict[str, Tuple[Any, Optional[float]]] = OrderedDict()
        self._lock = threading.RLock()

        # Telemetry counters
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._invalidations = 0

        self._metrics = get_metrics_registry()
        self._sync_gauge()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def _sync_gauge(self) -> None:
        if self._metrics:
            self._metrics.cache_size.set(float(len(self._cache)), labels={"cache_type": self.name})

    def get(self, key: str) -> Optional[Any]:
        """Retrieves an item from cache. Returns None on cache miss or expiration."""
        now = time.time()
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                if self._metrics:
                    self._metrics.cache_misses.inc(labels={"cache_type": self.name})
                return None

            value, expiry = self._cache[key]
            if expiry is not None and now > expiry:
                # Expired
                del self._cache[key]
                self._evictions += 1
                self._misses += 1
                self._sync_gauge()
                if self._metrics:
                    self._metrics.cache_evictions.inc(labels={"cache_type": self.name})
                    self._metrics.cache_misses.inc(labels={"cache_type": self.name})
                return None

            # Move to most recently used position
            self._cache.move_to_end(key)
            self._hits += 1
            if self._metrics:
                self._metrics.cache_hits.inc(labels={"cache_type": self.name})
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Stores item with optional TTL, evicting oldest item if capacity is reached."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = time.time() + ttl if ttl and ttl > 0 else None

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (value, expiry)
                return

            if len(self._cache) >= self.max_size:
                # Evict oldest item (first item in OrderedDict)
                self._cache.popitem(last=False)
                self._evictions += 1
                if self._metrics:
                    self._metrics.cache_evictions.inc(labels={"cache_type": self.name})

            self._cache[key] = (value, expiry)
            self._sync_gauge()

    def delete(self, key: str) -> bool:
        """Deletes item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._invalidations += 1
                self._sync_gauge()
                if self._metrics:
                    self._metrics.cache_invalidations.inc(labels={"cache_type": self.name})
                return True
            return False

    def delete_by_prefix(self, prefix: str) -> int:
        """Invalidates all cache keys starting with prefix."""
        with self._lock:
            to_delete = [k for k in self._cache if k.startswith(prefix)]
            for k in to_delete:
                del self._cache[k]
            if to_delete:
                self._invalidations += len(to_delete)
                self._sync_gauge()
                if self._metrics:
                    self._metrics.cache_invalidations.inc(len(to_delete), labels={"cache_type": self.name})
            return len(to_delete)

    def clear(self) -> int:
        """Evicts all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._invalidations += count
            self._sync_gauge()
            if self._metrics and count > 0:
                self._metrics.cache_invalidations.inc(count, labels={"cache_type": self.name})
            return count

    def stats(self) -> Dict[str, Any]:
        """Returns diagnostic telemetry dictionary."""
        with self._lock:
            total = self._hits + self._misses
            ratio = self._hits / total if total > 0 else 0.0
            return {
                "name": self.name,
                "size": len(self._cache),
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "total_lookups": total,
                "hit_ratio": round(ratio, 4),
                "hit_rate_pct": round(ratio * 100.0, 2),
                "evictions": self._evictions,
                "invalidations": self._invalidations,
            }
