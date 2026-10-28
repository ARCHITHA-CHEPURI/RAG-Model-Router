import os
import sys
import numpy as np

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from vector_store.faiss_store import FAISSVectorStore

embeddings_path = config.EMBEDDINGS
index_path = config.FAISS_INDEX

# Ensure target directories exist
os.makedirs(os.path.dirname(index_path), exist_ok=True)

embeddings = np.load(embeddings_path)

dimension = embeddings.shape[1]

store = FAISSVectorStore(dimension)

store.add_embeddings(embeddings)

store.save(index_path)

print("--------------------------------")
print("FAISS Index Built Successfully")
print(f"Dimension: {dimension}")
print(f"Vectors Indexed: {len(embeddings)}")
print(f"Saved at: {index_path}")
print("--------------------------------")