import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.app import ANSWER_CACHE
from api.app import PROJECT_ROOT, _resolve_scored_results_path, app
from llm.ollama_client import ModelResponse
from llm.router_strategies import BaseRouter, RuleRouter


class MetricsApiTests(unittest.TestCase):
    def setUp(self):
        ANSWER_CACHE.clear()
        self.client = TestClient(app)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results_path = Path(self.temp_dir.name) / "scored.csv"
        self.feedback_path = Path(self.temp_dir.name) / "feedback.sqlite3"
        self.patch_results_path = patch(
            "api.app.SCORED_RESULTS_PATH",
            self.results_path,
        )
        self.patch_results_path.start()
        self.addCleanup(self.patch_results_path.stop)
        self.patch_feedback_path = patch(
            "api.app.FEEDBACK_DATABASE_PATH",
            self.feedback_path,
        )
        self.patch_feedback_path.start()
        self.addCleanup(self.patch_feedback_path.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def write_results(self):
        fields = [
            "run_id",
            "timestamp",
            "query_id",
            "configuration",
            "query_type",
            "selected_model",
            "generation_seconds",
            "total_seconds",
            "auto_correct",
            "manual_correct",
        ]
        rows = [
            {
                "run_id": "run-1",
                "timestamp": "2026-10-26T10:00:00",
                "query_id": "Q1",
                "configuration": "base_router",
                "query_type": "factual",
                "selected_model": "llama3.2:1b",
                "generation_seconds": "10",
                "total_seconds": "10",
                "auto_correct": "1",
                "manual_correct": "",
            },
            {
                "run_id": "run-1",
                "timestamp": "2026-10-26T10:00:00",
                "query_id": "Q1",
                "configuration": "rule_router",
                "query_type": "factual",
                "selected_model": "qwen2.5:0.5b",
                "generation_seconds": "5",
                "total_seconds": "5",
                "auto_correct": "1",
                "manual_correct": "",
            },
        ]
        with self.results_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def issue_answer(self, router="ml_router"):
        client = MagicMock()
        client.generate_response.return_value = ModelResponse(
            text="RAG combines retrieval and generation.",
            model="qwen2.5:0.5b",
            input_tokens=12,
            output_tokens=7,
            cost_usd=None,
            latency_seconds=0.02,
        )
        llm_router = SimpleNamespace(
            clients={"small": client, "large": client},
            model_names={
                "small": "qwen2.5:0.5b",
                "large": "llama3.2:1b",
            },
            choose_model=lambda query: ("small", 0.81),
        )
        retrieval_pipeline = SimpleNamespace(
            search=lambda query, top_k: [
                {
                    "text": "RAG combines retrieval and generation.",
                    "metadata": {"document_name": "rag.pdf"},
                }
            ]
        )
        with patch(
            "api.app._get_generation_components",
            return_value=(
                retrieval_pipeline,
                llm_router,
                BaseRouter(),
                RuleRouter(),
            ),
        ):
            return self.client.post(
                "/api/answer",
                json={"query": "What is RAG?", "router": router},
            )

    def test_health_endpoint_does_not_require_benchmark_data(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_scored_results_path_supports_project_relative_and_absolute_paths(self):
        project_relative = _resolve_scored_results_path(
            "evaluation/technical_benchmark_reviewed.csv"
        )
        absolute = _resolve_scored_results_path(
            str(self.results_path.resolve())
        )

        self.assertEqual(
            project_relative,
            PROJECT_ROOT / "evaluation" / "technical_benchmark_reviewed.csv",
        )
        self.assertEqual(absolute, self.results_path.resolve())

    def test_metrics_endpoint_reports_latest_run(self):
        self.write_results()

        response = self.client.get("/api/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "run-1")
        self.assertEqual(
            response.json()["comparison_baseline"],
            "base_router",
        )

    def test_run_list_reports_available_benchmark_runs(self):
        self.write_results()

        response = self.client.get("/api/runs")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"runs": ["run-1"]})

    def test_metrics_endpoint_returns_not_found_for_unknown_run(self):
        self.write_results()

        response = self.client.get("/api/metrics?run_id=missing")

        self.assertEqual(response.status_code, 404)

    def test_metrics_endpoint_explains_when_scored_data_is_missing(self):
        response = self.client.get("/api/metrics")

        self.assertEqual(response.status_code, 503)
        self.assertIn("Scored benchmark data is unavailable", response.json()["detail"])

    def test_feedback_is_persisted_and_aggregated(self):
        answer = self.issue_answer()
        self.assertEqual(answer.status_code, 200)
        answer_data = answer.json()

        submission = {
            "request_id": answer_data["request_id"],
            "rating": 5,
            "comment": "Helpful answer",
        }
        response = self.client.post("/api/feedback", json=submission)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["request_id"], answer_data["request_id"])
        self.assertEqual(response.json()["router"], "ml_router")
        self.assertEqual(response.json()["model"], "qwen2.5:0.5b")

        summary = self.client.get("/api/feedback/summary")

        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["responses"], 1)
        self.assertEqual(summary.json()["requests_issued"], 1)
        self.assertEqual(summary.json()["average_rating"], 5.0)
        self.assertEqual(summary.json()["positive_rate_percent"], 100.0)
        self.assertEqual(
            summary.json()["by_router_and_model"][0]["router"],
            "ml_router",
        )

    def test_feedback_rejects_duplicate_request_ids(self):
        answer = self.issue_answer()
        self.assertEqual(answer.status_code, 200)
        submission = {
            "request_id": answer.json()["request_id"],
            "rating": 4,
        }

        first = self.client.post("/api/feedback", json=submission)
        duplicate = self.client.post("/api/feedback", json=submission)

        self.assertEqual(first.status_code, 201)
        self.assertEqual(duplicate.status_code, 409)

        summary = self.client.get("/api/feedback/summary")
        self.assertEqual(summary.json()["responses"], 1)

    def test_feedback_rejects_ids_not_issued_by_answer_endpoint(self):
        response = self.client.post(
            "/api/feedback",
            json={"request_id": "not-issued", "rating": 4},
        )

        self.assertEqual(response.status_code, 404)

    def test_feedback_validates_rating_and_rejects_extra_query_text(self):
        invalid_rating = self.client.post(
            "/api/feedback",
            json={
                "request_id": "not-issued",
                "rating": 6,
            },
        )
        raw_query = self.client.post(
            "/api/feedback",
            json={
                "request_id": "not-issued",
                "rating": 4,
                "query": "Do not store this query",
            },
        )

        self.assertEqual(invalid_rating.status_code, 422)
        self.assertEqual(raw_query.status_code, 422)

    def test_feedback_summary_is_empty_before_first_submission(self):
        response = self.client.get("/api/feedback/summary")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["responses"], 0)
        self.assertIsNone(response.json()["average_rating"])
        self.assertEqual(response.json()["by_router_and_model"], [])
        self.assertFalse(self.feedback_path.exists())

    def test_answer_endpoint_returns_a_rating_request_id(self):
        response = self.issue_answer(router="rule_router")

        self.assertEqual(response.status_code, 200)
        answer = response.json()
        self.assertTrue(answer["request_id"])
        self.assertEqual(answer["router"], "rule_router")
        self.assertEqual(answer["model"], "qwen2.5:0.5b")
        self.assertEqual(answer["sources"], ["rag.pdf"])
        self.assertEqual(answer["input_tokens"], 12)
        self.assertEqual(answer["output_tokens"], 7)
        self.assertIsNone(answer["cost_usd"])

    def test_repeat_answer_hits_cache_and_issues_a_new_feedback_id(self):
        client = MagicMock()
        client.generate_response.return_value = ModelResponse(
            text="RAG combines retrieval and generation.",
            model="qwen2.5:0.5b",
            input_tokens=12,
            output_tokens=7,
            cost_usd=None,
            latency_seconds=0.02,
        )
        retrieval_pipeline = MagicMock(
            search=MagicMock(
                return_value=[
                    {
                        "text": "RAG combines retrieval and generation.",
                        "metadata": {"document_name": "rag.pdf"},
                    }
                ]
            )
        )
        llm_router = SimpleNamespace(
            clients={"small": client, "large": client},
            model_names={
                "small": "qwen2.5:0.5b",
                "large": "llama3.2:1b",
            },
            choose_model=lambda query: ("small", 0.81),
        )
        with patch(
            "api.app._get_generation_components",
            return_value=(
                retrieval_pipeline,
                llm_router,
                BaseRouter(),
                RuleRouter(),
            ),
        ):
            first = self.client.post(
                "/api/answer",
                json={"query": "What is RAG?", "router": "ml_router"},
            )
            second = self.client.post(
                "/api/answer",
                json={"query": "  WHAT is RAG?  ", "router": "ml_router"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        first_data = first.json()
        second_data = second.json()
        self.assertFalse(first_data["cache_hit"])
        self.assertTrue(second_data["cache_hit"])
        self.assertNotEqual(first_data["request_id"], second_data["request_id"])
        self.assertEqual(second_data["answer"], first_data["answer"])
        self.assertEqual(second_data["retrieval_seconds"], 0.0)
        self.assertEqual(second_data["generation_seconds"], 0.0)
        self.assertEqual(second_data["input_tokens"], 0)
        self.assertEqual(second_data["output_tokens"], 0)
        self.assertEqual(second_data["cost_usd"], 0.0)
        retrieval_pipeline.search.assert_called_once_with("What is RAG?", top_k=3)
        client.generate_response.assert_called_once()

        second_feedback = self.client.post(
            "/api/feedback",
            json={"request_id": second_data["request_id"], "rating": 5},
        )
        self.assertEqual(second_feedback.status_code, 201)

    def test_cache_is_separate_for_each_router_strategy(self):
        with patch(
            "api.app._get_generation_components",
            return_value=(
                SimpleNamespace(
                    search=lambda query, top_k: [
                        {
                            "text": "RAG combines retrieval and generation.",
                            "metadata": {},
                        }
                    ]
                ),
                SimpleNamespace(
                    clients={
                        "small": SimpleNamespace(
                            generate_response=lambda prompt: ModelResponse(
                                text="Small model answer",
                                model="qwen2.5:0.5b",
                                input_tokens=5,
                                output_tokens=2,
                                cost_usd=None,
                                latency_seconds=0.01,
                            )
                        ),
                        "large": SimpleNamespace(
                            generate_response=lambda prompt: ModelResponse(
                                text="Large model answer",
                                model="llama3.2:1b",
                                input_tokens=5,
                                output_tokens=2,
                                cost_usd=None,
                                latency_seconds=0.01,
                            )
                        ),
                    },
                    model_names={
                        "small": "qwen2.5:0.5b",
                        "large": "llama3.2:1b",
                    },
                    choose_model=lambda query: ("small", 0.8),
                ),
                BaseRouter(),
                RuleRouter(),
            ),
        ):
            ml_answer = self.client.post(
                "/api/answer",
                json={"query": "What is RAG?", "router": "ml_router"},
            )
            rule_answer = self.client.post(
                "/api/answer",
                json={"query": "What is RAG?", "router": "rule_router"},
            )

        self.assertFalse(ml_answer.json()["cache_hit"])
        self.assertFalse(rule_answer.json()["cache_hit"])
        self.assertEqual(
            ml_answer.json()["request_id"] == rule_answer.json()["request_id"],
            False,
        )

    def test_cache_stats_endpoint_reports_hits_and_misses(self):
        self.issue_answer()
        self.issue_answer()

        response = self.client.get("/api/cache/stats")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["hits"], 1)
        self.assertEqual(response.json()["misses"], 1)
        self.assertEqual(response.json()["hit_rate_percent"], 50.0)

    def test_answer_endpoint_returns_feedback_id_for_grounding_fallback(self):
        retrieval_pipeline = SimpleNamespace(search=lambda query, top_k: [])
        llm_router = SimpleNamespace(clients={}, model_names={})
        with patch(
            "api.app._get_generation_components",
            return_value=(
                retrieval_pipeline,
                llm_router,
                BaseRouter(),
                RuleRouter(),
            ),
        ):
            response = self.client.post(
                "/api/answer",
                json={"query": "What is RAG?", "router": "ml_router"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["answer"],
            "I'm sorry, but the provided documents do not contain any "
            "information related to your question.",
        )
        self.assertEqual(response.json()["model"], "none_fallback")

        feedback = self.client.post(
            "/api/feedback",
            json={"request_id": response.json()["request_id"], "rating": 3},
        )
        self.assertEqual(feedback.status_code, 201)


if __name__ == "__main__":
    unittest.main()
