import hashlib
import threading
import time
from collections import OrderedDict


class AnswerCache:
    def __init__(self, max_entries=256, ttl_seconds=300):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0.")

        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(query, router):
        normalized_query = " ".join(query.casefold().split())
        value = f"{router}\0{normalized_query}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def get(self, query, router):
        key = self.make_key(query, router)
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, answer = entry
            if expires_at <= now:
                del self._entries[key]
                self._misses += 1
                return None

            self._entries.move_to_end(key)
            self._hits += 1
            return answer.copy()

    def set(self, query, router, answer):
        key = self.make_key(query, router)
        with self._lock:
            self._entries[key] = (
                time.monotonic() + self.ttl_seconds,
                answer.copy(),
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def stats(self):
        with self._lock:
            request_count = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "requests": request_count,
                "hit_rate_percent": (
                    round(self._hits / request_count * 100, 1)
                    if request_count
                    else 0.0
                ),
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "scope": "in-memory; counters and entries reset on API restart",
            }
