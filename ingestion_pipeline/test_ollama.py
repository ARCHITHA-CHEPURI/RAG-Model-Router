from llm.ollama_client import OllamaClient

llm = OllamaClient()

prompt = """
Context: RAG stands for Retrieval-Augmented Generation.
Question: What does RAG stand for?
Answer using the context only.
"""

print(llm.generate(prompt))