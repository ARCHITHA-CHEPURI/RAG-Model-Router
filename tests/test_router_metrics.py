import unittest
from contextlib import redirect_stdout
from io import StringIO

from evaluation.router_metrics import analyze_results, print_report


def benchmark_row(configuration, selected_model, query_id, query_type, total):
    return {
        "run_id": "run-1",
        "timestamp": "2026-10-26T10:00:00",
        "query_id": query_id,
        "configuration": configuration,
        "query_type": query_type,
        "selected_model": selected_model,
        "generation_seconds": str(total),
        "total_seconds": str(total),
        "cost_usd": "",
        "expected_source_retrieved": "1",
        "auto_correct": "1",
        "manual_correct": "",
    }


class RouterMetricsTests(unittest.TestCase):
    def test_reports_quality_latency_and_labeled_proxy_savings(self):
        rows = [
            benchmark_row(
                "llama_only", "llama3.2:1b", "Q1", "factual", 10
            ),
            benchmark_row(
                "llama_only", "none_fallback", "Q2", "out_of_scope", 0
            ),
            benchmark_row(
                "ml_router", "qwen2.5:0.5b", "Q1", "factual", 5
            ),
            benchmark_row(
                "ml_router", "none_fallback", "Q2", "out_of_scope", 0
            ),
        ]

        report = analyze_results(rows)

        self.assertEqual(
            report["configurations"]["ml_router"][
                "automated_factual_accuracy"
            ],
            {"correct": 1, "scored": 1, "accuracy_percent": 100.0},
        )
        self.assertEqual(
            report["configurations"]["ml_router"][
                "expected_source_hit_rate_at_3"
            ],
            {"correct": 1, "scored": 1, "accuracy_percent": 100.0},
        )
        self.assertEqual(
            report["comparisons_vs_baseline"]["ml_router"][
                "virtual_cost_savings_percent"
            ],
            50.0,
        )
        self.assertEqual(report["comparison_baseline"], "llama_only")
        self.assertIn("not currency", report["cost_note"])

    def test_rejects_unknown_models_instead_of_underreporting_cost(self):
        rows = [
            benchmark_row("ml_router", "unpriced-model", "Q1", "factual", 1)
        ]

        with self.assertRaisesRegex(ValueError, "No virtual cost weight"):
            analyze_results(rows)

    def test_accepts_pandas_binary_score_serialization(self):
        rows = [
            benchmark_row(
                "base_router",
                "llama3.2:1b",
                "Q1",
                "factual",
                1,
            )
        ]
        rows[0]["manual_correct"] = "1.0"
        rows[0]["expected_source_retrieved"] = "1.0"

        report = analyze_results(rows)

        self.assertEqual(
            report["configurations"]["base_router"][
                "manual_factual_accuracy"
            ]["accuracy_percent"],
            100.0,
        )
        self.assertEqual(
            report["configurations"]["base_router"][
                "expected_source_hit_rate_at_3"
            ]["accuracy_percent"],
            100.0,
        )

    def test_uses_base_router_as_the_comparison_baseline(self):
        rows = [
            benchmark_row(
                "base_router", "llama3.2:1b", "Q1", "factual", 10
            ),
            benchmark_row(
                "rule_router", "qwen2.5:0.5b", "Q1", "factual", 5
            ),
        ]

        report = analyze_results(rows)

        self.assertEqual(report["comparison_baseline"], "base_router")
        self.assertEqual(
            report["comparisons_vs_baseline"]["rule_router"][
                "virtual_cost_savings_percent"
            ],
            50.0,
        )

    def test_reports_provider_cost_savings_only_with_complete_cost_data(self):
        baseline_rows = [
            benchmark_row("base_router", "llama3.2:1b", "Q1", "factual", 10),
            benchmark_row("base_router", "llama3.2:1b", "Q2", "factual", 10),
        ]
        routed_rows = [
            benchmark_row("ml_router", "qwen2.5:0.5b", "Q1", "factual", 5),
            benchmark_row("ml_router", "qwen2.5:0.5b", "Q2", "factual", 5),
        ]
        for row, cost in zip(baseline_rows, ("0.01", "0.02")):
            row["cost_usd"] = cost
        for row, cost in zip(routed_rows, ("0.005", "0.01")):
            row["cost_usd"] = cost

        report = analyze_results(baseline_rows + routed_rows)

        self.assertEqual(
            report["configurations"]["ml_router"][
                "litellm_estimated_cost_usd"
            ],
            0.015,
        )
        self.assertEqual(
            report["comparisons_vs_baseline"]["ml_router"][
                "litellm_estimated_cost_savings_percent"
            ],
            50.0,
        )
        self.assertEqual(
            report["comparisons_vs_baseline"]["ml_router"][
                "cost_estimate_coverage_percent"
            ],
            100.0,
        )

        routed_rows[1]["cost_usd"] = ""
        incomplete = analyze_results(baseline_rows + routed_rows)
        self.assertIsNone(
            incomplete["comparisons_vs_baseline"]["ml_router"][
                "litellm_estimated_cost_savings_percent"
            ]
        )
        self.assertEqual(
            incomplete["comparisons_vs_baseline"]["ml_router"][
                "cost_estimate_coverage_percent"
            ],
            50.0,
        )

    def test_fallback_rows_do_not_inflate_cost_estimate_coverage(self):
        baseline_rows = [
            benchmark_row("base_router", "llama3.2:1b", "Q1", "factual", 10),
            benchmark_row("base_router", "none_fallback", "Q2", "out_of_scope", 0),
        ]
        routed_rows = [
            benchmark_row("ml_router", "qwen2.5:0.5b", "Q1", "factual", 5),
            benchmark_row("ml_router", "none_fallback", "Q2", "out_of_scope", 0),
        ]
        baseline_rows[0]["cost_usd"] = ""
        routed_rows[0]["cost_usd"] = ""
        baseline_rows[1]["cost_usd"] = "0"
        routed_rows[1]["cost_usd"] = "0"

        report = analyze_results(baseline_rows + routed_rows)
        comparison = report["comparisons_vs_baseline"]["ml_router"]

        self.assertEqual(comparison["cost_estimate_coverage_percent"], 0.0)
        self.assertIsNone(
            comparison["litellm_estimated_cost_savings_percent"]
        )

    def test_paired_manual_quality_requires_complete_reviews(self):
        baseline_rows = [
            benchmark_row("base_router", "llama3.2:1b", "Q1", "factual", 10),
            benchmark_row("base_router", "llama3.2:1b", "Q2", "factual", 10),
        ]
        routed_rows = [
            benchmark_row("ml_router", "qwen2.5:0.5b", "Q1", "factual", 5),
            benchmark_row("ml_router", "llama3.2:1b", "Q2", "factual", 8),
        ]
        for row, score in zip(baseline_rows, ("1", "1")):
            row["manual_correct"] = score
        for row, score in zip(routed_rows, ("1", "0")):
            row["manual_correct"] = score

        report = analyze_results(baseline_rows + routed_rows)
        quality = report["comparisons_vs_baseline"]["ml_router"][
            "paired_manual_factual_quality"
        ]

        self.assertEqual(quality["review_coverage_percent"], 100.0)
        self.assertEqual(quality["baseline_accuracy_percent"], 100.0)
        self.assertEqual(quality["routed_accuracy_percent"], 50.0)
        self.assertEqual(
            quality["accuracy_change_percentage_points"],
            -50.0,
        )

        routed_rows[1]["manual_correct"] = ""
        incomplete_report = analyze_results(baseline_rows + routed_rows)
        incomplete_quality = incomplete_report["comparisons_vs_baseline"][
            "ml_router"
        ]["paired_manual_factual_quality"]
        self.assertEqual(incomplete_quality["review_coverage_percent"], 50.0)
        self.assertIsNone(
            incomplete_quality["accuracy_change_percentage_points"]
        )

    def test_report_prints_cleanly_when_provider_cost_is_unavailable(self):
        report = analyze_results(
            [
                benchmark_row(
                    "base_router",
                    "llama3.2:1b",
                    "Q1",
                    "factual",
                    1,
                )
            ]
        )
        output = StringIO()

        with redirect_stdout(output):
            print_report(report)

        self.assertIn("n/a", output.getvalue())


if __name__ == "__main__":
    unittest.main()
