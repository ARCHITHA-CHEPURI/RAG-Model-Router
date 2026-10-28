import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings):
        embeddings = np.asarray(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        query_embedding = np.asarray([query_embedding]).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        return scores[0], indices[0]

    def save(self, path):
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path):
        index = faiss.read_index(path)

        obj = cls(index.d)

        obj.index = index

        return obj