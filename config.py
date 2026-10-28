
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================
# Data folders
# ==========================

DATA_DIR = os.path.join(BASE_DIR, "DATA")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")

# ==========================
# Processed Files
# ==========================

PROCESSED_DOCUMENTS = os.path.join(
    PROCESSED_DATA_DIR, "processed_documents.json"
)

PROCESSED_CHUNKS = os.path.join(
    PROCESSED_DATA_DIR, "processed_chunks.json"
)

# Embeddings and their metadata are saved in the project root.
EMBEDDINGS = os.path.join(BASE_DIR, "embeddings.npy")

# Actual metadata file is in the project root
EMBEDDING_METADATA = os.path.join(
    BASE_DIR, "embedding_metadata.json"
)

# ==========================
# Vector Database
# ==========================

# Actual FAISS index is inside vector_store
FAISS_INDEX = os.path.join(
    BASE_DIR, "vector_store", "index.faiss"
)
