import unittest

from llm.prompt_builder import PromptBuilder


class PromptBuilderTests(unittest.TestCase):
    def test_returns_no_prompt_when_technical_terms_are_missing(self):
        chunks = [
            {
                "text": (
                    "Hybrid retrieval combines dense vector search with "
                    "traditional keyword-based methods."
                ),
                "metadata": {"document_name": "rag.pdf"},
            }
        ]

        prompt = PromptBuilder.build_prompt(
            "Explain how hybrid retrieval combines semantic search, "
            "BM25, and Reciprocal Rank Fusion in a RAG system.",
            chunks,
        )

        self.assertIsNone(prompt)

    def test_does_not_count_terms_hidden_by_context_truncation(self):
        chunks = [
            {
                "text": "RAG " + ("background " * 400) + "BM25",
                "metadata": {"document_name": "rag.pdf"},
            }
        ]

        prompt = PromptBuilder.build_prompt("Explain BM25 in RAG.", chunks)

        self.assertIsNone(prompt)

    def test_builds_prompt_when_required_terms_are_supported(self):
        chunks = [
            {
                "text": (
                    "RAG uses BM25 and Reciprocal Rank Fusion to combine "
                    "retrieval results."
                ),
                "metadata": {"document_name": "rag.pdf"},
            }
        ]

        prompt = PromptBuilder.build_prompt(
            "Explain BM25 and Reciprocal Rank Fusion in a RAG system.",
            chunks,
        )

        self.assertIsNotNone(prompt)
        self.assertIn(chunks[0]["text"], prompt)

    def test_returns_no_prompt_when_there_are_no_chunks(self):
        self.assertIsNone(PromptBuilder.build_prompt("What is RAG?", []))


if __name__ == "__main__":
    unittest.main()
