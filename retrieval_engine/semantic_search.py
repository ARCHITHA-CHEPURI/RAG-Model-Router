import os
import sys

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Also ensure local retrieval_engine directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from rag_pipeline import RetrievalPipeline


def run_interactive_search():
    print("=" * 80)
    print("Welcome to the Improved Hybrid Retrieval CLI!")
    print("=" * 80)
    
    use_ce_input = input("Enable neural Cross-Encoder re-ranking? (y/n) [default: n]: ").strip().lower()
    use_cross_encoder = use_ce_input == 'y' or use_ce_input == 'yes'

    print("\nInitializing retrieval pipeline...")
    pipeline = RetrievalPipeline(use_cross_encoder=use_cross_encoder)
    print("\nPipeline ready!")
    print("=" * 80)

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()

        if not query:
            continue
        if query.lower() == "exit":
            break

        results = pipeline.search(query, top_k=5)

        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        if not results:
            print("No matching results found.")
            print("=" * 80)
            continue

        for rank, res in enumerate(results, start=1):
            meta = res["metadata"]
            
            # Formulate the score printout
            score_details = f"RRF Score: {res['rrf_score']:.4f}"
            if "re_rank_score" in res:
                score_details = f"Re-rank Score: {res['re_rank_score']:.4f} | " + score_details
            
            # Print source ranking information
            dense_rank_str = str(res["dense_rank"]) if res["dense_rank"] is not None else "N/A"
            sparse_rank_str = str(res["sparse_rank"]) if res["sparse_rank"] is not None else "N/A"
            rank_details = f"Dense Rank: {dense_rank_str} | Sparse Rank: {sparse_rank_str}"

            print(f"\nResult {rank} [{score_details}]")
            print(f"Retrieval Ranks  : {rank_details}")
            print(f"Document         : {meta.get('document_name', meta.get('filename', 'Unknown'))}")
            print(f"Chunk / Total    : {meta.get('chunk_number', 0)} / {meta.get('total_chunks', 0)}")
            print("-" * 80)
            print(res["text"][:600])
            if len(res["text"]) > 600:
                print("... [Truncated]")
            print("-" * 80)

        print("=" * 80)


if __name__ == "__main__":
    run_interactive_search()