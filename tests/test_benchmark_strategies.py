import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from evaluation import benchmark_updated
from llm.ollama_client import ModelResponse


class BenchmarkStrategyTests(unittest.TestCase):
    def test_configuration_order_rotates_fairly_and_repeatably(self):
        configurations = {
            "base_router": object(),
            "rule_router": object(),
            "ml_router": object(),
        }

        first_pass = [
            [
                name
                for name, _ in benchmark_updated._rotate_configurations(
                    configurations,
                    question_index,
                )
            ]
            for question_index in range(1, 4)
        ]
        repeated_pass = [
            [
                name
                for name, _ in benchmark_updated._rotate_configurations(
                    configurations,
                    question_index,
                )
            ]
            for question_index in range(1, 4)
        ]

        self.assertEqual(first_pass, repeated_pass)
        self.assertEqual(
            [order[0] for order in first_pass],
            ["base_router", "rule_router", "ml_router"],
        )

    def test_append_results_expands_existing_csv_for_cost_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.csv"
            with output_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["run_id", "query_id", "answer"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "run_id": "old-run",
                        "query_id": "old-query",
                        "answer": "historical answer",
                    }
                )

            with patch.object(
                benchmark_updated,
                "OUTPUT_CSV",
                str(output_path),
            ):
                benchmark_updated._append_results(
                    [
                        {
                            "run_id": "new-run",
                            "query_id": "new-query",
                            "answer": "new answer",
                        }
                    ]
                )

            with output_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                reader = csv.DictReader(file)
                rows = list(reader)

        self.assertIn("cost_usd", reader.fieldnames)
        self.assertEqual(
            [(row["run_id"], row["answer"]) for row in rows],
            [
                ("old-run", "historical answer"),
                ("new-run", "new answer"),
            ],
        )
        self.assertEqual(rows[0]["cost_usd"], "")

    def test_benchmark_runs_all_strategies_on_the_same_query(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "questions.csv"
            output_path = Path(temp_dir) / "results.csv"
            with input_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "query_id",
                        "query",
                        "query_type",
                        "expected_answer",
                        "expected_source",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "query_id": "Q1",
                        "query": "Write a complex React component",
                        "query_type": "factual",
                        "expected_answer": "A component.",
                        "expected_source": "test.pdf",
                    }
                )

            large_client = MagicMock()
            large_client.generate_response.return_value = ModelResponse(
                text="Large response",
                model="llama3.2:1b",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.0002,
                latency_seconds=0.02,
            )
            small_client = MagicMock()
            small_client.generate_response.return_value = ModelResponse(
                text="Small response",
                model="qwen2.5:0.5b",
                input_tokens=10,
                output_tokens=5,
                cost_usd=None,
                latency_seconds=0.01,
            )
            fake_router = SimpleNamespace(
                clients={"large": large_client, "small": small_client},
                model_names={
                    "large": "llama3.2:1b",
                    "small": "qwen2.5:0.5b",
                },
                choose_model=lambda query: ("small", 0.8),
            )

            with (
                patch.object(benchmark_updated, "INPUT_CSV", str(input_path)),
                patch.object(benchmark_updated, "OUTPUT_CSV", str(output_path)),
                patch.object(
                    benchmark_updated,
                    "RetrievalPipeline",
                    return_value=MagicMock(
                        search=MagicMock(
                            return_value=[
                                {
                                    "text": "context",
                                    "metadata": {"filename": "test.pdf"},
                                }
                            ]
                        )
                    ),
                ),
                patch.object(
                    benchmark_updated,
                    "LLMRouter",
                    return_value=fake_router,
                ),
                patch.object(
                    benchmark_updated.PromptBuilder,
                    "build_prompt",
                    return_value="prompt",
                ),
            ):
                benchmark_updated.run_benchmark()

            with output_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(
            {row["configuration"] for row in rows},
            {"base_router", "rule_router", "ml_router"},
        )
        models_by_strategy = {
            row["configuration"]: row["selected_model"] for row in rows
        }
        self.assertEqual(models_by_strategy["base_router"], "llama3.2:1b")
        self.assertEqual(models_by_strategy["rule_router"], "llama3.2:1b")
        self.assertEqual(models_by_strategy["ml_router"], "qwen2.5:0.5b")
        self.assertEqual(large_client.generate_response.call_count, 2)
        self.assertEqual(small_client.generate_response.call_count, 1)
        self.assertEqual(rows[0]["input_tokens"], "10")
        self.assertEqual(rows[0]["output_tokens"], "5")
        self.assertEqual(rows[0]["cost_usd"], "0.0002")
        self.assertEqual(
            rows[0]["expected_source_retrieved"],
            "1",
        )
        self.assertEqual(
            rows[0]["retrieved_sources"],
            "test.pdf",
        )
        self.assertEqual(rows[2]["cost_usd"], "")


if __name__ == "__main__":
    unittest.main()
