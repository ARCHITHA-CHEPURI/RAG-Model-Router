import os
import sys
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    PROCESSED_CHUNKS,
    EMBEDDINGS,
    EMBEDDING_METADATA,
)

# ==========================================================
# Load Embedding Model
# ==========================================================

print("=" * 60)
print("Loading Sentence Transformer Model...")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully!\n")

# ==========================================================
# Load Processed Chunks
# ==========================================================

if not os.path.exists(PROCESSED_CHUNKS):
    raise FileNotFoundError(
        f"processed_chunks.json not found!\n{PROCESSED_CHUNKS}"
    )

with open(PROCESSED_CHUNKS, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks.\n")

# ==========================================================
# Generate Embeddings
# ==========================================================

texts = []
metadata_entries = []

for chunk in chunks:
    texts.append(chunk["text"])
    metadata_entries.append({
        "text": chunk["text"],
        "metadata": chunk["metadata"]
    })

print("Generating embeddings...\n")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

# ==========================================================
# Create Output Directories
# ==========================================================

os.makedirs(os.path.dirname(EMBEDDINGS), exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDING_METADATA), exist_ok=True)

# ==========================================================
# Save Embeddings
# ==========================================================

np.save(EMBEDDINGS, embeddings)

with open(EMBEDDING_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata_entries, f, indent=4, ensure_ascii=False)

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 60)
print("Embedding Generation Completed Successfully!")
print("=" * 60)

print(f"Generated Embeddings : {len(embeddings)}")
print(f"Embedding Dimension  : {embeddings.shape[1]}")

print("\nSaved Files")

print(f"Embeddings : {EMBEDDINGS}")
print(f"Metadata   : {EMBEDDING_METADATA}")

print("=" * 60)