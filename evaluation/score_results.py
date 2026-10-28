
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Load benchmark results
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
results_path = BASE_DIR / "benchmark_results.csv"
output_path = BASE_DIR / "benchmark_scored.csv"

if not results_path.exists():
    raise FileNotFoundError(
        f"Benchmark results file not found: {results_path}"
    )

df = pd.read_csv(results_path)

required_columns = {
    "query",
    "answer",
    "selected_model",
    "query_type",
    "configuration",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

# Safely handle missing text values.
for column in ["query", "answer", "selected_model", "query_type"]:
    df[column] = df[column].fillna("").astype(str)

# --------------------------------------------------
# 2. Preliminary automatic grading
# --------------------------------------------------

def grade(row):
    q = row["query"].strip().lower()
    a = row["answer"].strip().lower()
    model = row["selected_model"].strip().lower()
    query_type = row["query_type"].strip().lower()

    # Out-of-scope questions must use the intended fallback.
    if query_type == "out_of_scope":
        return int(
            model == "none_fallback"
            and "provided documents do not contain" in a
        )

    # Do not automatically grade empty answers as correct.
    if not a:
        return 0

    # RAG components.
    if "fundamental components of a rag" in q:
        return int(
            "retrieval" in a and "generation" in a
        )

    # Hybrid retrieval.
    if "hybrid retrieval combine" in q:
        return int(
            "dense" in a
            and ("keyword" in a or "sparse" in a)
        )

    # Transformer architecture.
    if "attention is all you need" in q:
        return int("transformer" in a)

    # BLEU score.
    if "bleu score" in q:
        return int("28.4" in a)

    # Tata's five core values.
    if "five core values" in q:
        values = [
            "integrity",
            "unity",
            "responsibility",
            "pioneering",
            "excellence",
        ]
        return int(all(value in a for value in values))

    # Microsoft's founding year.
    if "microsoft founded" in q:
        return int("1975" in a)

    # IBM's total revenue and growth.
    if "ibm's total revenue" in q:
        has_revenue = (
            "$17.0 billion" in a
            or "$17 billion" in a
            or "17.0 billion" in a
            or "17 billion" in a
        )
        has_growth = (
            "8 percent" in a
            or "8%" in a
        )
        return int(has_revenue and has_growth)

    # Dietary Guidelines for Indians.
    if "dietary guidelines for indians" in q:
        return int(
            "national institute of nutrition" in a
            and "2011" in a
        )

    # Unknown question types are not automatically graded.
    return ""

# Recalculate the automatic score every time this script runs.
df["auto_correct"] = df.apply(grade, axis=1)

# --------------------------------------------------
# 3. Save scored results separately
# --------------------------------------------------

df.to_csv(output_path, index=False)

print(f"Scored results saved to: {output_path}")
print("\nPreliminary automatic accuracy by configuration:")

# --------------------------------------------------
# 4. Accuracy summary
# --------------------------------------------------

for config, group in df.groupby("configuration"):
    scored = group[group["auto_correct"].isin([0, 1])]

    if scored.empty:
        print(f"{config}: no automatically graded rows")
        continue

    correct = scored["auto_correct"].astype(int).sum()
    total = len(scored)
    accuracy = correct / total * 100

    print(
        f"{config}: {correct}/{total} "
        f"= {accuracy:.1f}% "
        f"({total} scored rows)"
    )

# --------------------------------------------------
# 5. Timing summaries
# --------------------------------------------------

def report_median_time(column, title, subset):
    if column not in subset.columns:
        print(f"\n{title}: column '{column}' not available.")
        return

    times = pd.to_numeric(
        subset[column], errors="coerce"
    ).dropna()

    if times.empty:
        print(f"\n{title}: no valid timing values.")
        return

    print(f"\n{title} (median seconds):")

    medians = (
        subset.assign(
            _time=pd.to_numeric(
                subset[column], errors="coerce"
            )
        )
        .groupby("configuration")["_time"]
        .median()
        .round(2)
    )

    print(medians.to_string())


report_median_time(
    "generation_seconds",
    "Median generation time",
    df,
)

report_median_time(
    "total_seconds",
    "Median total time",
    df,
)

# --------------------------------------------------
# 6. Factual-question timing summary
# --------------------------------------------------

factual = df[
    df["query_type"].str.strip().str.lower() == "factual"
].copy()

if not factual.empty:
    report_median_time(
        "total_seconds",
        "Factual questions - median total time",
        factual,
    )
else:
    print("\nNo rows marked with query_type='factual'.")

print(
    "\nNote: automatic scores are keyword-based preliminary "
    "checks, not verified human accuracy."
)
