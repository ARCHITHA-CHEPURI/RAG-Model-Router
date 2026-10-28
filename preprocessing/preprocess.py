import os
import sys
import json

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cleaner import clean_text
from chunker import SemanticChunker
from metadata import generate_metadata

from config import (
    PROCESSED_DOCUMENTS,
    PROCESSED_CHUNKS,
)

# ==========================================================
# Main Preprocessing Pipeline
# ==========================================================

def main():

    print("=" * 60)
    print("Starting Preprocessing Pipeline...")
    print("=" * 60)

    # ------------------------------------------------------
    # Check input file
    # ------------------------------------------------------

    if not os.path.exists(PROCESSED_DOCUMENTS):
        print("\nERROR:")
        print("processed_documents.json not found!")
        print(PROCESSED_DOCUMENTS)
        return

    print(f"\nReading documents from:\n{PROCESSED_DOCUMENTS}\n")

    with open(PROCESSED_DOCUMENTS, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents.\n")

    processed_chunks = []

    chunk_id = 1

    # Create chunker once
    chunker = SemanticChunker(
        max_words=450,
        overlap_sentences=2
    )

    # ------------------------------------------------------
    # Process every document
    # ------------------------------------------------------

    for doc in documents:

        filename = os.path.basename(doc["filename"])

        print(f"Processing: {filename}")

        text = doc.get("text", "")

        if not text.strip():
            print("Skipped (Empty document)\n")
            continue

        # ------------------------------
        # Step 1 : Clean Text
        # ------------------------------

        cleaned_text = clean_text(text)

        # ------------------------------
        # Step 2 : Chunk Text
        # ------------------------------

        chunks = chunker.chunk_text(cleaned_text)

        total_chunks = len(chunks)

        # ------------------------------
        # Step 3 : Metadata
        # ------------------------------

        for i, chunk in enumerate(chunks, start=1):

            metadata = generate_metadata(
                doc=doc,
                chunk_number=i,
                total_chunks=total_chunks,
                text=chunk
            )

            metadata["id"] = chunk_id

            processed_chunks.append(
                {
                    "text": chunk,
                    "metadata": metadata
                }
            )

            chunk_id += 1

        print(f"Generated {total_chunks} chunks.\n")

    # ------------------------------------------------------
    # Create output directory if needed
    # ------------------------------------------------------

    os.makedirs(os.path.dirname(PROCESSED_CHUNKS), exist_ok=True)

    # ------------------------------------------------------
    # Save processed chunks
    # ------------------------------------------------------

    with open(PROCESSED_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(
            processed_chunks,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 60)
    print("Preprocessing Completed Successfully!")
    print(f"Generated {len(processed_chunks)} chunks.")
    print(f"Saved to:\n{PROCESSED_CHUNKS}")
    print("=" * 60)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()