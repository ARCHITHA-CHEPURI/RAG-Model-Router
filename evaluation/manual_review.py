import argparse
import csv
from pathlib import Path


REVIEW_FIELDS = [
    "run_id",
    "query_id",
    "configuration",
    "selected_model",
    "query_type",
    "query",
    "expected_answer",
    "expected_source",
    "retrieved_sources",
    "answer",
    "manual_correct",
    "review_notes",
]
REVIEW_KEY = ("run_id", "query_id", "configuration")


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        header = file.readline()
        delimiter = max(
            (",", "\t", ";"),
            key=header.count,
        )
        file.seek(0)
        reader = csv.DictReader(
            file,
            delimiter=delimiter,
            quotechar='"',
            doublequote=True,
        )
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return reader.fieldnames, list(reader)


def _review_key(row):
    key = tuple(row.get(field, "").strip() for field in REVIEW_KEY)
    if not all(key):
        raise ValueError(
            "Review rows require run_id, query_id, and configuration."
        )
    return key


def export_review_sheet(input_path, output_path, run_id=None):
    _, rows = _read_csv(input_path)
    if not rows:
        raise ValueError(f"No benchmark results found in {input_path}.")

    selected_run = run_id or max(
        rows,
        key=lambda row: (row.get("timestamp", ""), row.get("run_id", "")),
    ).get("run_id", "")
    selected_rows = [
        row for row in rows if row.get("run_id", "") == selected_run
    ]
    if not selected_rows:
        raise ValueError(f"No benchmark rows found for run {selected_run!r}.")

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(
            f"Review sheet already exists: {output}. Choose another path "
            "to avoid overwriting reviewer work."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in REVIEW_FIELDS
                }
            )
    return selected_run, len(selected_rows)


def apply_review_sheet(input_path, review_path, output_path):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Review output must not overwrite the input benchmark.")

    input_fields, result_rows = _read_csv(input_path)
    review_fields, review_rows = _read_csv(review_path)
    required_fields = set(REVIEW_KEY) | {"manual_correct"}
    missing_fields = required_fields - set(review_fields)
    if missing_fields:
        raise ValueError(
            f"Review CSV is missing fields: {sorted(missing_fields)}"
        )

    result_by_key = {}
    for row in result_rows:
        key = _review_key(row)
        if key in result_by_key:
            raise ValueError(f"Duplicate benchmark result row: {key}.")
        result_by_key[key] = row

    review_scores = {}
    for row in review_rows:
        key = _review_key(row)
        if key not in result_by_key:
            raise ValueError(f"Review row does not match a benchmark result: {key}.")
        if key in review_scores:
            raise ValueError(f"Duplicate review row: {key}.")
        score = row.get("manual_correct", "").strip()
        if score and score not in {"0", "1"}:
            raise ValueError(
                f"Expected manual_correct to be 0, 1, or empty for {key}."
            )
        review_scores[key] = score

    if "manual_correct" not in input_fields:
        input_fields.append("manual_correct")

    for key, score in review_scores.items():
        if score:
            result_by_key[key]["manual_correct"] = score

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=input_fields)
        writer.writeheader()
        writer.writerows(result_rows)

    reviewed = sum(
        row.get("manual_correct", "").strip() in {"0", "1"}
        for row in result_rows
    )
    return reviewed, len(result_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export and apply human review for RAG benchmark answers."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-id",
        help="Run to export; defaults to the most recently timestamped run.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply scores from an existing review sheet to the output CSV.",
    )
    args = parser.parse_args()

    if args.apply:
        if args.output is None:
            parser.error("--output is required with --apply.")
        reviewed, total = apply_review_sheet(
            args.input,
            args.review,
            args.output,
        )
        print(f"Reviewed answers: {reviewed}/{total}")
        print(f"Scored benchmark saved to: {args.output}")
    else:
        run_id, count = export_review_sheet(
            args.input,
            args.review,
            args.run_id,
        )
        print(f"Exported {count} answers from run {run_id} to: {args.review}")
        print("Review each answer, enter 1 or 0 in manual_correct, and save.")
        print("Then rerun this command with --apply and --output to score it.")


if __name__ == "__main__":
    main()
