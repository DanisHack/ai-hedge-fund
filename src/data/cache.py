"""Simple in-memory cache for market data API responses."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional


class DataCache:
    """In-memory cache with TTL support.

    Keys are composite tuples, e.g. ("prices", "AAPL", "2024-01-01", "2024-12-31").
    """

    def __init__(self, default_ttl_minutes: int = 60):
        self._store: dict[tuple, tuple[Any, datetime]] = {}
        self._default_ttl = timedelta(minutes=default_ttl_minutes)

    def get(self, *key_parts: str) -> Optional[Any]:
        """Return cached value if present and not expired, else None."""
        key = key_parts
        if key in self._store:
            value, expires_at = self._store[key]
            if datetime.now() < expires_at:
                return value
            del self._store[key]
        return None

    def set(self, *key_parts_and_value: Any, ttl_minutes: Optional[int] = None) -> None:
        """Cache a value. Last argument is the value; preceding args form the key.

        Usage: cache.set("prices", "AAPL", "2024-01-01", "2024-12-31", data)
        """
        *key_parts, value = key_parts_and_value
        key = tuple(key_parts)
        ttl = timedelta(minutes=ttl_minutes) if ttl_minutes else self._default_ttl
        self._store[key] = (value, datetime.now() + ttl)

    def clear(self) -> None:
        """Clear all cached data."""
        self._store.clear()


# Module-level singleton
_cache = DataCache()


def get_cache() -> DataCache:
    """Return the module-level cache singleton."""
    return _cache
