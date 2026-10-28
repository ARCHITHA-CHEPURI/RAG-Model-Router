
import csv
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV = os.path.join(
    ROOT, "evaluation", "benchmark_scored_updated.csv"
)

OUTPUT_CSV = os.path.join(
    ROOT, "evaluation", "final_benchmark_summary.csv"
)


def accuracy(rows):
    scored = [
        row for row in rows
        if row.get("manual_correct", "").strip() in {"0", "1"}
    ]

    if not scored:
        return 0, 0, 0.0

    correct = sum(
        row["manual_correct"].strip() == "1"
        for row in scored
    )

    return correct, len(scored), round(100 * correct / len(scored), 1)


def median_time(rows, column):
    values = [
        float(row[column])
        for row in rows
        if row.get(column, "").strip()
        and not row.get("error", "").strip()
        and row.get("selected_model") != "none_fallback"
    ]

    return round(statistics.median(values), 2) if values else ""


if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
    all_rows = list(csv.DictReader(f))

if not all_rows:
    raise ValueError("The scored benchmark CSV is empty.")

# Keep only the latest benchmark run, if the CSV contains multiple runs.
latest_run_id = max(row.get("run_id", "") for row in all_rows)
rows = [
    row for row in all_rows
    if row.get("run_id", "") == latest_run_id
]

configurations = ["qwen_only", "llama_only", "ml_router"]
summary = []

for config in configurations:
    config_rows = [
        row for row in rows
        if row.get("configuration") == config
    ]

    factual_rows = [
        row for row in config_rows
        if row.get("query_id", "").startswith("Q")
        and row["query_id"][1:].isdigit()
        and 1 <= int(row["query_id"][1:]) <= 8
    ]

    fallback_rows = [
        row for row in config_rows
        if row.get("query_id", "").startswith("Q")
        and row["query_id"][1:].isdigit()
        and 9 <= int(row["query_id"][1:]) <= 12
    ]

    overall_correct, overall_total, overall_accuracy = accuracy(config_rows)
    factual_correct, factual_total, factual_accuracy = accuracy(factual_rows)
    fallback_correct, fallback_total, fallback_accuracy = accuracy(fallback_rows)

    summary.append({
        "configuration": config,
        "questions_in_run": len(config_rows),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy_percent": overall_accuracy,
        "factual_correct": factual_correct,
        "factual_total": factual_total,
        "factual_accuracy_percent": factual_accuracy,
        "fallback_correct": fallback_correct,
        "fallback_total": fallback_total,
        "fallback_accuracy_percent": fallback_accuracy,
        "median_generation_seconds": median_time(
            config_rows, "generation_seconds"
        ),
        "median_total_seconds": median_time(
            config_rows, "total_seconds"
        ),
    })

with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("\nFINAL BENCHMARK SUMMARY")
print("-" * 78)

for item in summary:
    print(
        f"{item['configuration']:12} | "
        f"Overall: {item['overall_correct']}/{item['overall_total']} "
        f"({item['overall_accuracy_percent']}%) | "
        f"Factual: {item['factual_correct']}/{item['factual_total']} "
        f"({item['factual_accuracy_percent']}%) | "
        f"Fallback: {item['fallback_correct']}/{item['fallback_total']} "
        f"({item['fallback_accuracy_percent']}%) | "
        f"Median generation: {item['median_generation_seconds']} s"
    )

print(f"\nSummary saved to: {OUTPUT_CSV}")
print("Source benchmark files were not modified.")
