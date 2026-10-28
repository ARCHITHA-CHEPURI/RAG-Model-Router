
import time

from retrieval_engine.rag_pipeline import RetrievalPipeline
from llm.llm_router import LLMRouter
from llm.prompt_builder import PromptBuilder


def print_separator(char="-", width=70):
    print(char * width)


def get_source(result):
    """Get the source filename from a retrieval result."""
    metadata = result.get("metadata") or {}

    return (
        result.get("source")
        or metadata.get("source")
        or metadata.get("filename")
        or metadata.get("document_name")
        or metadata.get("file_name")
        or "Unknown source"
    )


def main():
    print("=" * 70)
    print("AI MODEL ROUTER - HYBRID RAG")
    print("=" * 70)

    # 1. Initialize the retrieval pipeline and LLM router.
    print("\n[1] Initializing retrieval pipeline and LLM router...")

    initialization_start = time.perf_counter()

    try:
        retrieval_pipeline = RetrievalPipeline(
            use_cross_encoder=False,
            min_similarity=0.30,
        )
        llm_router = LLMRouter()

    except Exception as error:
        print(f"\nInitialization failed: {error}")
        return

    initialization_time = time.perf_counter() - initialization_start

    print(f"Initialization time: {initialization_time:.2f} seconds")

    # 2. Get the user's question.
    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a non-empty question.")
        return

    print("\n" + "=" * 70)
    print("PROCESSING QUERY")
    print("=" * 70)

    # 3. Retrieve document chunks using hybrid retrieval.
    print("\n[2] Retrieving relevant document chunks...")

    retrieval_start = time.perf_counter()

    try:
        results = retrieval_pipeline.search(
            query,
            top_k=3,
        )

    except Exception as error:
        print(f"Retrieval failed: {error}")
        return

    retrieval_time = time.perf_counter() - retrieval_start

    print(f"Retrieval time: {retrieval_time:.2f} seconds")
    print(f"Retrieved chunks: {len(results)}")

    print("\n")
    print_separator()
    print("RETRIEVAL RESULTS")
    print_separator()

    for position, result in enumerate(results, start=1):
        print(f"\nResult {position}")
        print(f"Source: {get_source(result)}")

        # Debugging: verify which fields are returned.
        print("DEBUG result keys:", list(result.keys()))

        dense_score = result.get("dense_score")
        dense_rank = result.get("dense_rank")

        print("DEBUG dense_score:", repr(dense_score))
        print("DEBUG dense_rank:", repr(dense_rank))

        if dense_score is not None:
            print(f"Dense similarity: {float(dense_score):.4f}")
        else:
            print("Dense similarity: Not available for this result")

        bm25_score = result.get("bm25_score")
        if bm25_score is not None:
            print(f"BM25 score: {float(bm25_score):.4f}")

        rrf_score = result.get("rrf_score")
        if rrf_score is not None:
            print(f"RRF score: {float(rrf_score):.6f}")

        if dense_rank is not None:
            print(f"Dense rank: {dense_rank}")

        sparse_rank = result.get("sparse_rank")
        if sparse_rank is not None:
            print(f"Sparse rank: {sparse_rank}")

        rerank_score = result.get("re_rank_score")
        if rerank_score is not None:
            print(f"Cross-Encoder rerank score: {float(rerank_score):.4f}")

        content = result.get("text", "")
        preview_length = 500

        if len(content) > preview_length:
            content = content[:preview_length] + "..."

        print(f"Content: {content}")

    # 4. Build the RAG prompt using your existing PromptBuilder class.
    print("\n[3] Building the RAG prompt...")

    prompt_start = time.perf_counter()

    try:
        prompt = PromptBuilder.build_prompt(query, results)

    except Exception as error:
        print(f"Prompt construction failed: {error}")
        return

    prompt_time = time.perf_counter() - prompt_start

    print(f"Prompt construction time: {prompt_time:.4f} seconds")

    if not prompt:
        print(f"\n{PromptBuilder.FALLBACK}")
        print(
            "The model was not called because the retrieved context "
            "does not support the question's required terms."
        )
        return

    print(f"Prompt length: {len(prompt):,} characters")
    print(f"Prompt word count: {len(prompt.split()):,} words")

    # 5. Route the query and generate the answer.
    print("\n[4] Selecting an LLM and generating the answer...")
    print("Please wait; generation may take some time.")

    try:
        generation_result = llm_router.generate(
            query,
            prompt=prompt,
        )

    except Exception as error:
        print(f"\nGeneration failed: {error}")
        return

    response = generation_result.get("response", "")
    generation_time = generation_result.get("latency_seconds", 0.0)

    # 6. Display the answer.
    print("\n")
    print("=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    if not isinstance(response, str) or not response.strip():
        print("The model returned an empty response.")
    elif response.startswith("Error:"):
        print(f"Generation error: {response}")
    else:
        print(response)

    # 7. Display the source documents.
    print("\n")
    print_separator()
    print("SOURCE DOCUMENTS")
    print_separator()

    sources = list(dict.fromkeys(get_source(result) for result in results))

    if sources:
        for source in sources:
            print(f"- {source}")
    else:
        print("- No source documents were retrieved.")

    # 8. Display model-routing details.
    print("\n")
    print_separator()
    print("LLM ROUTING DETAILS")
    print_separator()

    print(f"Selected model: {generation_result.get('model', 'Unknown')}")
    print(f"Model category: {generation_result.get('category', 'Unknown')}")
    print(
        "Routing confidence: "
        f"{generation_result.get('confidence', 'Unknown')}"
    )
    print(
        "Router-reported latency: "
        f"{generation_result.get('latency_seconds', 'Unknown')} seconds"
    )
    print(f"Measured generation time: {generation_time:.2f} seconds")

    # 9. Display performance measurements.
    total_time = (
        initialization_time
        + retrieval_time
        + prompt_time
        + generation_time
    )

    print("\n")
    print("=" * 70)
    print("PERFORMANCE BREAKDOWN")
    print("=" * 70)

    print(f"Initialization:          {initialization_time:>10.2f} seconds")
    print(f"Document retrieval:      {retrieval_time:>10.2f} seconds")
    print(f"Prompt construction:     {prompt_time:>10.4f} seconds")
    print(f"LLM generation:          {generation_time:>10.2f} seconds")

    print_separator()
    print(f"Total measured time:     {total_time:>10.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
