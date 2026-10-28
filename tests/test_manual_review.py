import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.manual_review import (
    apply_review_sheet,
    export_review_sheet,
)


class ManualReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.input_path = self.directory / "results.csv"
        self.review_path = self.directory / "review.csv"
        self.output_path = self.directory / "scored.csv"
        self.rows = [
            {
                "run_id": "run-1",
                "timestamp": "2026-10-26T12:00:00",
                "query_id": "Q1",
                "configuration": "base_router",
                "selected_model": "llama3.2:1b",
                "query_type": "factual",
                "query": "Question?",
                "expected_answer": "Expected.",
                "expected_source": "manual.pdf",
                "retrieved_sources": "manual.pdf",
                "answer": "Answer.",
                "manual_correct": "",
            },
            {
                "run_id": "run-1",
                "timestamp": "2026-10-26T12:00:00",
                "query_id": "Q1",
                "configuration": "ml_router",
                "selected_model": "qwen2.5:0.5b",
                "query_type": "factual",
                "query": "Question?",
                "expected_answer": "Expected.",
                "expected_source": "manual.pdf",
                "retrieved_sources": "manual.pdf",
                "answer": "Answer.",
                "manual_correct": "",
            },
        ]
        self._write_csv(self.input_path, self.rows)
        self.addCleanup(self.temp_dir.cleanup)

    @staticmethod
    def _write_csv(path, rows):
        fields = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_exports_latest_run_with_empty_manual_scores(self):
        run_id, count = export_review_sheet(
            self.input_path,
            self.review_path,
        )

        with self.review_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reviews = list(csv.DictReader(file))

        self.assertEqual(run_id, "run-1")
        self.assertEqual(count, 2)
        self.assertEqual(len(reviews), 2)
        self.assertEqual(reviews[0]["manual_correct"], "")
        self.assertEqual(reviews[0]["answer"], "Answer.")

    def test_applies_manual_scores_without_modifying_source_file(self):
        _, count = export_review_sheet(self.input_path, self.review_path)
        with self.review_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reviews = list(csv.DictReader(file))
        reviews[0]["manual_correct"] = "1"
        reviews[1]["manual_correct"] = "0"
        self._write_csv(self.review_path, reviews)

        reviewed, total = apply_review_sheet(
            self.input_path,
            self.review_path,
            self.output_path,
        )

        with self.output_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            results = list(csv.DictReader(file))
        self.assertEqual(count, 2)
        self.assertEqual((reviewed, total), (2, 2))
        self.assertEqual(
            [row["manual_correct"] for row in results],
            ["1", "0"],
        )
        with self.input_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            original = list(csv.DictReader(file))
        self.assertEqual(original[0]["manual_correct"], "")

    def test_reads_excel_tab_delimited_review_export(self):
        export_review_sheet(self.input_path, self.review_path)
        with self.review_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reviews = list(csv.DictReader(file))
        for row in reviews:
            row["manual_correct"] = "1"
        fields = list(reviews[0])
        with self.review_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(reviews)

        reviewed, total = apply_review_sheet(
            self.input_path,
            self.review_path,
            self.output_path,
        )

        self.assertEqual((reviewed, total), (2, 2))
        with self.output_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(
            [row["manual_correct"] for row in rows],
            ["1", "1"],
        )

    def test_reads_quoted_commas_and_quotes_from_scored_csv(self):
        result_rows = [
            {
                **self.rows[0],
                "answer": 'The paper "Attention Is All You Need" proposes it.',
                "auto_correct": "1",
            }
        ]
        self._write_csv(self.input_path, result_rows)
        self._write_csv(
            self.review_path,
            [
                {
                    **result_rows[0],
                    "review_notes": "Includes quoted, punctuated text.",
                    "manual_correct": "1",
                }
            ],
        )

        reviewed, total = apply_review_sheet(
            self.input_path,
            self.review_path,
            self.output_path,
        )

        self.assertEqual((reviewed, total), (1, 1))
        with self.output_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            row = next(csv.DictReader(file))
        self.assertEqual(
            row["answer"],
            'The paper "Attention Is All You Need" proposes it.',
        )

    def test_rejects_non_binary_manual_score(self):
        export_review_sheet(self.input_path, self.review_path)
        with self.review_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reviews = list(csv.DictReader(file))
        reviews[0]["manual_correct"] = "yes"
        self._write_csv(self.review_path, reviews)

        with self.assertRaisesRegex(ValueError, "manual_correct"):
            apply_review_sheet(
                self.input_path,
                self.review_path,
                self.output_path,
            )

    def test_export_does_not_overwrite_existing_reviewer_work(self):
        self.review_path.write_text("reviewer work", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            export_review_sheet(self.input_path, self.review_path)

    def test_rejects_review_rows_for_unknown_benchmark_outputs(self):
        export_review_sheet(self.input_path, self.review_path)
        with self.review_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reviews = list(csv.DictReader(file))
        reviews[0]["query_id"] = "unknown"
        self._write_csv(self.review_path, reviews)

        with self.assertRaisesRegex(ValueError, "does not match"):
            apply_review_sheet(
                self.input_path,
                self.review_path,
                self.output_path,
            )


if __name__ == "__main__":
    unittest.main()
