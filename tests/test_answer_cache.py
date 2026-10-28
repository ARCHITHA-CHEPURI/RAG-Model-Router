import unittest
from unittest.mock import patch

from api.answer_cache import AnswerCache


class AnswerCacheTests(unittest.TestCase):
    def test_cache_hit_returns_a_copy_and_tracks_hit_rate(self):
        cache = AnswerCache(max_entries=2, ttl_seconds=30)
        answer = {"answer": "RAG answer", "sources": ["rag.pdf"]}
        cache.set("What is RAG?", "ml_router", answer)

        cached = cache.get(" what is   rag? ", "ml_router")
        cached["answer"] = "Changed by caller"

        self.assertEqual(
            cache.get("What is RAG?", "ml_router")["answer"],
            "RAG answer",
        )
        self.assertEqual(
            cache.stats(),
            {
                "hits": 2,
                "misses": 0,
                "requests": 2,
                "hit_rate_percent": 100.0,
                "entries": 1,
                "max_entries": 2,
                "ttl_seconds": 30,
                "scope": (
                    "in-memory; counters and entries reset on API restart"
                ),
            },
        )

    def test_cache_is_namespaced_by_router_and_evicts_oldest_entry(self):
        cache = AnswerCache(max_entries=1, ttl_seconds=30)
        cache.set("What is RAG?", "ml_router", {"answer": "ML"})

        self.assertIsNone(cache.get("What is RAG?", "rule_router"))
        cache.set("What is Python?", "ml_router", {"answer": "Python"})

        self.assertIsNone(cache.get("What is RAG?", "ml_router"))
        self.assertEqual(cache.get("What is Python?", "ml_router")["answer"], "Python")
        self.assertEqual(cache.stats()["entries"], 1)

    def test_expired_entries_are_not_returned(self):
        cache = AnswerCache(max_entries=1, ttl_seconds=0.001)
        with patch("api.answer_cache.time.monotonic", side_effect=[10.0, 11.0]):
            cache.set("What is RAG?", "ml_router", {"answer": "RAG"})
            expired = cache.get("What is RAG?", "ml_router")

        self.assertIsNone(expired)
        self.assertEqual(cache.stats()["misses"], 1)
        self.assertEqual(cache.stats()["entries"], 0)

    def test_invalid_cache_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            AnswerCache(max_entries=0)
        with self.assertRaises(ValueError):
            AnswerCache(ttl_seconds=0)


if __name__ == "__main__":
    unittest.main()
