import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT / "evaluation" / "benchmark_scored_updated.csv"
)

# Relative model-size units, not provider prices or measured electricity cost.
MODEL_COST_UNITS = {
    "qwen2.5:0.5b": 0.5,
    "llama3.2:1b": 1.0,
    "none_fallback": 0.0,
}

REQUIRED_FIELDS = {
    "run_id",
    "timestamp",
    "query_id",
    "configuration",
    "query_type",
    "selected_model",
    "generation_seconds",
    "total_seconds",
}


def _score(row, field):
    value = row.get(field, "").strip()
    if not value:
        return None
    if value in {"0", "0.0"}:
        return 0
    if value in {"1", "1.0"}:
        return 1
    raise ValueError(f"Expected {field} to be 0, 1, or empty; got {value!r}.")


def _number(row, field):
    value = row.get(field, "").strip()
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"Expected a numeric {field}; got {value!r}."
        ) from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Expected a finite, non-negative {field}.")
    return number


def _optional_number(row, field):
    value = row.get(field, "")
    if value is None or not str(value).strip():
        return None
    return _number(row, field)


def load_results(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Benchmark CSV is missing required fields: {sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"No benchmark results found in {path}.")

    return rows


def _latest_run(rows):
    runs = defaultdict(list)
    for row in rows:
        runs[row["run_id"]].append(row)
    return max(
        runs,
        key=lambda run_id: max(
            (row["timestamp"], run_id) for row in runs[run_id]
        ),
    )


def _percentile_95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _accuracy(rows, query_type, field):
    scores = [
        _score(row, field)
        for row in rows
        if row["query_type"].strip().lower() == query_type
    ]
    scored = [score for score in scores if score is not None]
    return {
        "correct": sum(scored),
        "scored": len(scored),
        "accuracy_percent": (
            round(sum(scored) / len(scored) * 100, 1)
            if scored
            else None
        ),
    }


def _paired_manual_quality(baseline, routed, shared_ids):
    factual_ids = [
        query_id
        for query_id in shared_ids
        if baseline[query_id]["query_type"].strip().lower() == "factual"
        and routed[query_id]["query_type"].strip().lower() == "factual"
    ]
    reviewed_ids = [
        query_id
        for query_id in factual_ids
        if str(baseline[query_id].get("manual_correct") or "").strip()
        in {"0", "1"}
        and str(routed[query_id].get("manual_correct") or "").strip()
        in {"0", "1"}
    ]
    coverage = (
        round(len(reviewed_ids) / len(factual_ids) * 100, 1)
        if factual_ids
        else None
    )
    if not factual_ids or len(reviewed_ids) != len(factual_ids):
        return {
            "queries_compared": len(factual_ids),
            "reviewed_queries": len(reviewed_ids),
            "review_coverage_percent": coverage,
            "baseline_accuracy_percent": None,
            "routed_accuracy_percent": None,
            "accuracy_change_percentage_points": None,
        }

    baseline_accuracy = (
        sum(int(baseline[query_id]["manual_correct"]) for query_id in reviewed_ids)
        / len(reviewed_ids)
        * 100
    )
    routed_accuracy = (
        sum(int(routed[query_id]["manual_correct"]) for query_id in reviewed_ids)
        / len(reviewed_ids)
        * 100
    )
    return {
        "queries_compared": len(factual_ids),
        "reviewed_queries": len(reviewed_ids),
        "review_coverage_percent": coverage,
        "baseline_accuracy_percent": round(baseline_accuracy, 1),
        "routed_accuracy_percent": round(routed_accuracy, 1),
        "accuracy_change_percentage_points": round(
            routed_accuracy - baseline_accuracy,
            1,
        ),
    }


def analyze_results(rows, run_id=None):
    selected_run = run_id or _latest_run(rows)
    run_rows = [row for row in rows if row["run_id"] == selected_run]
    if not run_rows:
        raise ValueError(f"No benchmark rows found for run {selected_run!r}.")

    groups = defaultdict(list)
    query_configs = defaultdict(dict)
    for row in run_rows:
        configuration = row["configuration"].strip()
        query_id = row["query_id"].strip()
        if not configuration or not query_id:
            raise ValueError("Benchmark rows need configuration and query_id.")
        if query_id in query_configs[configuration]:
            raise ValueError(
                f"Duplicate query_id {query_id!r} in {configuration!r}."
            )
        if row["selected_model"] not in MODEL_COST_UNITS:
            raise ValueError(
                "No virtual cost weight is configured for model "
                f"{row['selected_model']!r}."
            )

        row["_generation_seconds"] = _number(row, "generation_seconds")
        row["_total_seconds"] = _number(row, "total_seconds")
        row["_model_cost_units"] = MODEL_COST_UNITS[row["selected_model"]]
        row["_cost_usd"] = _optional_number(row, "cost_usd")
        groups[configuration].append(row)
        query_configs[configuration][query_id] = row

    metrics = {}
    for configuration, config_rows in sorted(groups.items()):
        total_times = [row["_total_seconds"] for row in config_rows]
        generated = [
            row for row in config_rows
            if row["selected_model"] != "none_fallback"
        ]
        factual = _accuracy(config_rows, "factual", "auto_correct")
        out_of_scope = _accuracy(config_rows, "out_of_scope", "auto_correct")
        manual_factual = _accuracy(config_rows, "factual", "manual_correct")
        manual_out_of_scope = _accuracy(
            config_rows,
            "out_of_scope",
            "manual_correct",
        )
        source_hit_rate = _accuracy(
            config_rows,
            "factual",
            "expected_source_retrieved",
        )
        units = sum(row["_model_cost_units"] for row in config_rows)
        generated_rows = [
            row for row in config_rows
            if row["selected_model"] != "none_fallback"
        ]
        known_cost_rows = [
            row for row in generated_rows if row["_cost_usd"] is not None
        ]

        metrics[configuration] = {
            "queries": len(config_rows),
            "median_total_seconds": round(statistics.median(total_times), 3),
            "p95_total_seconds": round(_percentile_95(total_times), 3),
            "median_generation_seconds": round(
                statistics.median(
                    row["_generation_seconds"] for row in config_rows
                ),
                3,
            ),
            "automated_factual_accuracy": factual,
            "automated_out_of_scope_accuracy": out_of_scope,
            "manual_factual_accuracy": manual_factual,
            "manual_out_of_scope_accuracy": manual_out_of_scope,
            "expected_source_hit_rate_at_3": source_hit_rate,
            "small_model_share_percent": round(
                sum(
                    row["selected_model"] == "qwen2.5:0.5b"
                    for row in generated
                )
                / len(generated)
                * 100,
                1,
            ) if generated else 0.0,
            "virtual_cost_units": round(units, 3),
            "litellm_estimated_cost_usd": (
                round(
                    sum(row["_cost_usd"] for row in known_cost_rows),
                    8,
                )
                if known_cost_rows
                else None
            ),
            "cost_estimate_coverage_percent": round(
                len(known_cost_rows) / len(generated_rows) * 100,
                1,
            ) if generated_rows else 100.0,
        }

    comparisons = {}
    baseline_name = (
        "base_router"
        if "base_router" in query_configs
        else "llama_only"
    )
    baseline = query_configs.get(baseline_name)
    if baseline:
        for configuration, config_rows in query_configs.items():
            if configuration == baseline_name:
                continue

            shared_ids = baseline.keys() & config_rows.keys()
            if not shared_ids:
                continue

            comparable_generated_ids = [
                query_id
                for query_id in shared_ids
                if baseline[query_id]["selected_model"] != "none_fallback"
                and config_rows[query_id]["selected_model"] != "none_fallback"
            ]
            comparable_baseline_units = sum(
                baseline[query_id]["_model_cost_units"]
                for query_id in shared_ids
            )
            comparable_config_units = sum(
                config_rows[query_id]["_model_cost_units"]
                for query_id in shared_ids
            )
            known_cost_ids = [
                query_id
                for query_id in comparable_generated_ids
                if baseline[query_id]["_cost_usd"] is not None
                and config_rows[query_id]["_cost_usd"] is not None
            ]
            baseline_cost = sum(
                baseline[query_id]["_cost_usd"]
                for query_id in known_cost_ids
            )
            config_cost = sum(
                config_rows[query_id]["_cost_usd"]
                for query_id in known_cost_ids
            )
            baseline_latency = statistics.median(
                baseline[query_id]["_total_seconds"]
                for query_id in shared_ids
            )
            config_latency = statistics.median(
                config_rows[query_id]["_total_seconds"]
                for query_id in shared_ids
            )
            comparisons[configuration] = {
                "queries_compared": len(shared_ids),
                "median_latency_change_percent": round(
                    (config_latency - baseline_latency)
                    / baseline_latency
                    * 100,
                    1,
                ) if baseline_latency else None,
                "virtual_cost_savings_percent": round(
                    (comparable_baseline_units - comparable_config_units)
                    / comparable_baseline_units
                    * 100,
                    1,
                ) if comparable_baseline_units else None,
                "cost_estimate_queries_compared": len(known_cost_ids),
                "cost_estimate_coverage_percent": round(
                    len(known_cost_ids) / len(comparable_generated_ids) * 100,
                    1,
                ) if comparable_generated_ids else None,
                "litellm_estimated_cost_savings_percent": round(
                    (baseline_cost - config_cost) / baseline_cost * 100,
                    1,
                ) if (
                    comparable_generated_ids
                    and len(known_cost_ids) == len(comparable_generated_ids)
                    and baseline_cost > 0
                ) else None,
                "paired_manual_factual_quality": _paired_manual_quality(
                    baseline,
                    config_rows,
                    shared_ids,
                ),
            }

    return {
        "run_id": selected_run,
        "scoring_note": (
            "Automated accuracy is the benchmark's heuristic score, not "
            "human-verified answer quality."
        ),
        "cost_note": (
            "USD costs are LiteLLM estimates based on its model pricing, not "
            "invoice-verified charges; savings are omitted when estimate "
            "coverage is incomplete. Local Ollama inference has no provider "
            "bill in this metric, but still consumes electricity and "
            "hardware. Virtual cost units are a separate model-size proxy, "
            "not currency or measured energy."
        ),
        "configurations": metrics,
        "comparison_baseline": baseline_name,
        "comparisons_vs_baseline": comparisons,
    }


def print_report(report):
    print(f"Benchmark run: {report['run_id']}")
    print(
        f"{'Router':<16} {'N':>3} {'Median s':>10} {'P95 s':>8} "
        f"{'Factual auto':>14} {'Out-of-scope':>14} {'Small %':>9} "
        f"{'LiteLLM $':>12} {'Coverage':>10}"
    )
    for name, values in report["configurations"].items():
        cost = values["litellm_estimated_cost_usd"]
        cost_text = f"{cost:.6f}" if cost is not None else "n/a"
        factual = values["automated_factual_accuracy"]
        factual_text = (
            f"{factual['correct']}/{factual['scored']}"
            if factual["scored"]
            else "n/a"
        )
        out_of_scope = values["automated_out_of_scope_accuracy"]
        out_of_scope_text = (
            f"{out_of_scope['correct']}/{out_of_scope['scored']}"
            if out_of_scope["scored"]
            else "n/a"
        )
        print(
            f"{name:<16} {values['queries']:>3} "
            f"{values['median_total_seconds']:>10.3f} "
            f"{values['p95_total_seconds']:>8.3f} "
            f"{factual_text:>14} {out_of_scope_text:>14} "
            f"{values['small_model_share_percent']:>8.1f}% "
            f"{cost_text:>12} "
            f"{values['cost_estimate_coverage_percent']:>9.1f}%"
        )

    baseline = report["comparison_baseline"]
    for name, comparison in report["comparisons_vs_baseline"].items():
        quality = comparison["paired_manual_factual_quality"]
        print(
            f"{name} vs {baseline}: median latency change "
            f"{comparison['median_latency_change_percent']}%; "
            f"LiteLLM-estimated cost savings "
            f"{comparison['litellm_estimated_cost_savings_percent']}% "
            f"({comparison['cost_estimate_coverage_percent']}% coverage); "
            f"virtual cost proxy savings "
            f"{comparison['virtual_cost_savings_percent']}%; "
            "paired human-reviewed factual accuracy change "
            f"{quality['accuracy_change_percentage_points']} percentage "
            f"points ({quality['review_coverage_percent']}% review coverage)"
        )
    print(report["scoring_note"])
    print(report["cost_note"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize router latency, quality, provider-reported costs, "
            "and cost proxies."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Scored benchmark CSV (defaults to evaluation/benchmark_scored_updated.csv).",
    )
    parser.add_argument(
        "--run-id",
        help="Analyze this run ID; otherwise analyze the newest run.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for a machine-readable JSON report.",
    )
    args = parser.parse_args()

    report = analyze_results(load_results(args.input_csv), args.run_id)
    print_report(report)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report written to {args.json_out}")


if __name__ == "__main__":
    main()
