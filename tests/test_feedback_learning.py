import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from api.feedback_store import (
    get_model_quality_preference,
    record_feedback,
    register_request,
)
from llm.llm_router import LLMRouter


class FeedbackLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "feedback.sqlite3"
        self.request_index = 0
        self.addCleanup(self.temp_dir.cleanup)

    def record_ratings(self, model, ratings, router="ml_router"):
        for rating in ratings:
            request_id = f"{model}-{self.request_index}"
            self.request_index += 1
            register_request(
                self.database_path,
                request_id,
                router,
                model,
            )
            record_feedback(
                self.database_path,
                SimpleNamespace(
                    request_id=request_id,
                    rating=rating,
                    comment=None,
                ),
            )

    def test_quality_preference_requires_enough_ratings_and_clear_gap(self):
        self.record_ratings("small-model", [5, 5, 4, 5])
        self.record_ratings("large-model", [3, 3, 3, 3, 3])
        self.assertIsNone(
            get_model_quality_preference(
                self.database_path,
                "small-model",
                "large-model",
            )
        )

        self.record_ratings("small-model", [5])
        self.assertEqual(
            get_model_quality_preference(
                self.database_path,
                "small-model",
                "large-model",
            ),
            "small",
        )

    def test_quality_preference_requires_a_minimum_rating_difference(self):
        self.record_ratings("small-model", [4, 4, 4, 4, 4])
        self.record_ratings("large-model", [4, 4, 4, 4, 4])

        self.assertIsNone(
            get_model_quality_preference(
                self.database_path,
                "small-model",
                "large-model",
            )
        )

    def test_non_ml_router_ratings_do_not_influence_ml_routing(self):
        self.record_ratings("small-model", [5, 5, 5, 5, 5], router="base_router")
        self.record_ratings("large-model", [1, 1, 1, 1, 1], router="rule_router")

        self.assertIsNone(
            get_model_quality_preference(
                self.database_path,
                "small-model",
                "large-model",
            )
        )

    def test_uncertain_router_uses_feedback_preference(self):
        preference_provider = Mock(return_value="small")
        router = LLMRouter(
            feedback_preference_provider=preference_provider,
        )
        with patch.object(
            router.classifier,
            "predict_proba",
            return_value=np.array([[0.55, 0.45]]),
        ):
            selected, confidence = router.choose_model("unseen subject")

        self.assertEqual(selected, "small")
        self.assertEqual(confidence, 0.55)
        preference_provider.assert_called_once_with(
            "qwen2.5:0.5b",
            "llama3.2:1b",
        )

    def test_user_ratings_change_only_the_uncertain_model_choice(self):
        self.record_ratings("qwen2.5:0.5b", [5, 5, 5, 5, 5])
        self.record_ratings("llama3.2:1b", [3, 3, 3, 3, 3])
        router = LLMRouter(
            feedback_preference_provider=lambda small, large: (
                get_model_quality_preference(
                    self.database_path,
                    small,
                    large,
                )
            )
        )

        with patch.object(
            router.classifier,
            "predict_proba",
            return_value=np.array([[0.55, 0.45]]),
        ):
            uncertain_selection, _ = router.choose_model("unseen subject")
        with patch.object(
            router.classifier,
            "predict_proba",
            return_value=np.array([[0.8, 0.2]]),
        ):
            confident_selection, _ = router.choose_model("What is RAG?")

        self.assertEqual(uncertain_selection, "small")
        self.assertEqual(confident_selection, "large")

    def test_uncertain_router_keeps_large_fallback_without_feedback(self):
        router = LLMRouter()
        with patch.object(
            router.classifier,
            "predict_proba",
            return_value=np.array([[0.55, 0.45]]),
        ):
            selected, confidence = router.choose_model("unseen subject")

        self.assertEqual(selected, "large")
        self.assertEqual(confidence, 0.55)


if __name__ == "__main__":
    unittest.main()
