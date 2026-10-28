
import csv
import time
import statistics
import argparse
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import time
# Keep your other existing imports here
from retrieval_engine.rag_pipeline import RetrievalPipeline
from llm.llm_router import LLMRouter
from llm.prompt_builder import PromptBuilder
from llm.router_strategies import BaseRouter, RuleRouter


ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "evaluation" / "eval_questions.csv"
OUTPUT_CSV = ROOT / "evaluation" / "benchmark_results_updated.csv"

FALLBACK = PromptBuilder.FALLBACK

FIELDS = [
    "run_id",
    "timestamp",
    "query_id",
    "query",
    "query_type",
    "configuration",
    "execution_order",
    "selected_model",
    "router_confidence",
    "retrieved_chunks",
    "retrieved_sources",
    "expected_source_retrieved",
    "retrieval_seconds",
    "prompt_seconds",
    "generation_seconds",
    "total_seconds",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "answer",
    "expected_answer",
    "expected_source",
    "manual_correct",
    "error",
]


def run_benchmark(input_csv=None, output_csv=None):
    input_path = Path(input_csv) if input_csv else Path(INPUT_CSV)
    output_path = Path(output_csv) if output_csv else Path(OUTPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        questions = [
            row for row in csv.DictReader(f)
            if row.get("query", "").strip()
        ]

    if not questions:
        raise ValueError(
            "No questions found. Add questions to "
            "evaluation/eval_questions.csv first."
        )

    print("Initializing retrieval and routing components...")
    init_start = time.perf_counter()

    retriever = RetrievalPipeline(
        use_cross_encoder=False,
        min_similarity=0.30,
    )
    router = LLMRouter()
    prompt_builder = PromptBuilder()

    init_seconds = time.perf_counter() - init_start

    print(f"Initialization: {init_seconds:.2f} seconds")
    print(f"Questions loaded: {len(questions)}")
    allowed_sources = sorted(
        {
            Path(item["expected_source"]).name
            for item in questions
            if item.get("expected_source", "").strip()
        }
    )
    if allowed_sources:
        print(f"Retrieval restricted to evaluation sources: {allowed_sources}")

    configurations = {
        "base_router": BaseRouter(),
        "rule_router": RuleRouter(),
        "ml_router": router,
    }
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []

    for question_index, item in enumerate(questions, start=1):
        query = item["query"].strip()
        query_id = item.get("query_id") or str(question_index)
        query_type = item.get("query_type", "")
        expected_answer = item.get("expected_answer", "")
        expected_source = item.get("expected_source", "")

        print(f"\n[{question_index}/{len(questions)}] {query}")

        retrieval_start = time.perf_counter()
        results = retriever.search(
            query,
            top_k=3,
            allowed_sources=allowed_sources,
        )
        retrieval_seconds = time.perf_counter() - retrieval_start

        prompt_start = time.perf_counter()
        prompt = prompt_builder.build_prompt(query, results)
        prompt_seconds = time.perf_counter() - prompt_start
        retrieved_sources = sorted(
            {
                Path(
                    str(
                        (result.get("metadata") or {}).get("filename")
                        or (result.get("metadata") or {}).get("document_name")
                        or result.get("source")
                        or ""
                    )
                ).name
                for result in results
                if (
                    (result.get("metadata") or {}).get("filename")
                    or (result.get("metadata") or {}).get("document_name")
                    or result.get("source")
                )
            }
        )
        expected_source = Path(expected_source).name if expected_source else ""
        expected_source_retrieved = (
            int(expected_source in retrieved_sources)
            if expected_source
            else ""
        )

        execution_order = _rotate_configurations(
            configurations,
            question_index,
        )
        for order_index, (configuration, strategy) in enumerate(
            execution_order,
            start=1,
        ):
            generation_seconds = 0.0
            confidence = ""
            error = ""
            input_tokens = ""
            output_tokens = ""
            cost_usd = 0.0 if not results or not prompt else ""

            if not results or not prompt:
                answer = FALLBACK
                selected_model = "none_fallback"

            else:
                try:
                    generation_start = time.perf_counter()

                    model_type, route_confidence = strategy.choose_model(query)
                    selected_model = router.model_names[model_type]
                    model_response = router.clients[
                        model_type
                    ].generate_response(prompt)
                    answer = model_response.text
                    input_tokens = (
                        model_response.input_tokens
                        if model_response.input_tokens is not None
                        else ""
                    )
                    output_tokens = (
                        model_response.output_tokens
                        if model_response.output_tokens is not None
                        else ""
                    )
                    cost_usd = (
                        model_response.cost_usd
                        if model_response.cost_usd is not None
                        else ""
                    )
                    if route_confidence is not None:
                        confidence = round(route_confidence, 3)

                    generation_seconds = (
                        time.perf_counter() - generation_start
                    )

                    if not answer:
                        error = "Empty answer"
                    elif answer.startswith("Error:"):
                        error = answer
                except Exception as exc:
                    answer = ""
                    selected_model = "error"
                    error = f"{type(exc).__name__}: {exc}"

            total_seconds = (
                retrieval_seconds + prompt_seconds + generation_seconds
            )

            row = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query_id": query_id,
                "query": query,
                "query_type": query_type,
                "configuration": configuration,
                "execution_order": order_index,
                "selected_model": selected_model,
                "router_confidence": confidence,
                "retrieved_chunks": len(results),
                "retrieved_sources": "|".join(retrieved_sources),
                "expected_source_retrieved": expected_source_retrieved,
                "retrieval_seconds": round(retrieval_seconds, 4),
                "prompt_seconds": round(prompt_seconds, 4),
                "generation_seconds": round(generation_seconds, 4),
                "total_seconds": round(total_seconds, 4),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "answer": answer,
                "expected_answer": expected_answer,
                "expected_source": expected_source,
                "manual_correct": "",
                "error": error,
            }

            rows.append(row)

            print(
                f"  {configuration}: "
                f"model={selected_model}, "
                f"generation={generation_seconds:.2f}s"
            )

    _append_results(rows, output_path)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Initialization: {init_seconds:.2f}s")
    print(f"Recorded runs: {len(rows)}")
    print(f"Results: {output_path}")

    for configuration in configurations:
        times = [
            row["generation_seconds"]
            for row in rows
            if row["configuration"] == configuration
            and not row["error"]
            and row["retrieved_chunks"] > 0
        ]

        if times:
            print(
                f"{configuration}: "
                f"median generation = "
                f"{statistics.median(times):.2f}s"
            )


def _append_results(rows, output_path=None):
    output_path = Path(output_path) if output_path else Path(OUTPUT_CSV)
    existing_rows = []
    output_fields = FIELDS
    rewrite_existing = False

    if output_path.is_file():
        with output_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)

        if existing_fields != FIELDS:
            output_fields = FIELDS + [
                field for field in existing_fields if field not in FIELDS
            ]
            rewrite_existing = True

    mode = "w" if rewrite_existing or not output_path.exists() else "a"
    with output_path.open(mode, encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        if mode == "w":
            writer.writeheader()
            writer.writerows(existing_rows)
        writer.writerows(rows)


def _rotate_configurations(configurations, question_index):
    entries = list(configurations.items())
    offset = (question_index - 1) % len(entries)
    return entries[offset:] + entries[:offset]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the local RAG router strategies."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help="Question CSV (defaults to evaluation/eval_questions.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=(
            "Append benchmark rows here (defaults to "
            "evaluation/benchmark_results_updated.csv)."
        ),
    )
    args = parser.parse_args()
    run_benchmark(args.input, args.output)


if __name__ == "__main__":
    main()
