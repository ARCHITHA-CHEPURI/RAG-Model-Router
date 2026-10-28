import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from retrieval_engine.rag_pipeline import RetrievalPipeline


class RagSourceFilterTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
        pipeline.metadata = [
            {
                "text": "RAG uses retrieval and generation.",
                "metadata": {"filename": r"C:\docs\rag.pdf"},
            },
            {
                "text": "Transformer uses attention.",
                "metadata": {"document_name": "transformer.pdf"},
            },
            {
                "text": "Unrelated company information.",
                "metadata": {"filename": "company.pdf"},
            },
        ]
        pipeline.use_cross_encoder = False
        pipeline.min_similarity = 0.3
        pipeline.cross_encoder = None
        pipeline.embed_model = SimpleNamespace(encode=lambda query: [0.1, 0.2])
        pipeline.vector_store = SimpleNamespace(
            search=Mock(
                return_value=(
                    np.array([0.95, 0.9, 0.8]),
                    np.array([0, 1, 2]),
                )
            )
        )
        pipeline.bm25 = SimpleNamespace(
            get_scores=lambda query: np.array([2.0, 3.0, 1.0])
        )
        return pipeline

    def test_allowed_sources_exclude_other_corpus_documents(self):
        pipeline = self.make_pipeline()

        results = pipeline.search(
            "RAG retrieval",
            top_k=3,
            allowed_sources=["rag.pdf"],
        )

        self.assertTrue(results)
        self.assertTrue(
            all(
                result["metadata"].get("filename", "").endswith("rag.pdf")
                for result in results
            )
        )
        pipeline.vector_store.search.assert_called_once_with(
            [0.1, 0.2],
            top_k=3,
        )

    def test_empty_allowed_sources_return_no_results(self):
        pipeline = self.make_pipeline()

        results = pipeline.search("RAG retrieval", allowed_sources=[])

        self.assertEqual(results, [])
        pipeline.vector_store.search.assert_not_called()

    def test_unfiltered_search_preserves_existing_behavior(self):
        pipeline = self.make_pipeline()

        results = pipeline.search("RAG retrieval", top_k=2)

        self.assertEqual(len(results), 2)
        pipeline.vector_store.search.assert_called_once_with(
            [0.1, 0.2],
            top_k=3,
        )


if __name__ == "__main__":
    unittest.main()
