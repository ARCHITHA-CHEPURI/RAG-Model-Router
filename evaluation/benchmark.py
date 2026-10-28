
import csv
import os
import time
import statistics
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


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(ROOT, "evaluation", "eval_questions.csv")
OUTPUT_CSV = os.path.join(ROOT, "evaluation", "benchmark_results_updated.csv")

FALLBACK = PromptBuilder.FALLBACK

FIELDS = [
    "run_id",
    "timestamp",
    "query_id",
    "query",
    "query_type",
    "configuration",
    "selected_model",
    "router_confidence",
    "retrieved_chunks",
    "retrieval_seconds",
    "prompt_seconds",
    "generation_seconds",
    "total_seconds",
    "answer",
    "expected_answer",
    "expected_source",
    "manual_correct",
    "error",
]


def run_benchmark():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_CSV}"
        )

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
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

    configurations = ["qwen_only", "llama_only", "ml_router"]
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
        results = retriever.search(query, top_k=3)
        retrieval_seconds = time.perf_counter() - retrieval_start

        prompt_start = time.perf_counter()
        prompt = prompt_builder.build_prompt(query, results)
        prompt_seconds = time.perf_counter() - prompt_start

        for configuration in configurations:
            generation_seconds = 0.0
            confidence = ""
            error = ""

            if not results or not prompt:
                answer = FALLBACK
                selected_model = "none_fallback"

            else:
                try:
                    generation_start = time.perf_counter()

                    if configuration == "qwen_only":
                        selected_model = "qwen2.5:0.5b"
                        answer = router.clients["small"].generate(prompt)

                    elif configuration == "llama_only":
                        selected_model = "llama3.2:1b"
                        answer = router.clients["large"].generate(prompt)

                    else:
                        result = router.generate(query, prompt=prompt)
                        answer = result.get("response", "")
                        selected_model = result.get("model", "unknown")
                        confidence = result.get("confidence", "")

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
                "selected_model": selected_model,
                "router_confidence": confidence,
                "retrieved_chunks": len(results),
                "retrieval_seconds": round(retrieval_seconds, 4),
                "prompt_seconds": round(prompt_seconds, 4),
                "generation_seconds": round(generation_seconds, 4),
                "total_seconds": round(total_seconds, 4),
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

    file_exists = os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Initialization: {init_seconds:.2f}s")
    print(f"Recorded runs: {len(rows)}")
    print(f"Results: {OUTPUT_CSV}")

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


if __name__ == "__main__":
    run_benchmark()
