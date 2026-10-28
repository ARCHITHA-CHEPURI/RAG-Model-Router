from sentence_transformers import SentenceTransformer

# Load model only once
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text):
    """
    Generate embedding for a single text chunk.
    """
    return model.encode(text, convert_to_numpy=True)