
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from llm.ollama_client import LiteLLMClient
from llm.router_strategies import BaseRouter


class LLMRouter(BaseRouter):
    def __init__(self, feedback_preference_provider=None):
        self.feedback_preference_provider = feedback_preference_provider
        # Hand-labelled examples for query-complexity routing.
        # "small" selects Qwen; "large" selects Llama.
        training_data = [
            # Simple questions: small model
            ("What is RAG?", "small"),
            ("Define artificial intelligence", "small"),
            ("What is Python?", "small"),
            ("Explain machine learning briefly", "small"),
            ("What does an API do?", "small"),
            ("Define a database", "small"),
            ("What is an embedding?", "small"),
            ("Explain FAISS simply", "small"),
            ("What is a vector database?", "small"),
            ("Define neural networks", "small"),
            ("What is retrieval augmented generation?", "small"),
            ("Explain SQL joins briefly", "small"),
            ("What is recursion?", "small"),
            ("Summarize this paragraph", "small"),
            ("List the main components of RAG", "small"),
            ("What is the purpose of an index?", "small"),
            ("Explain this concept in simple words", "small"),
            ("Give a short definition of cloud computing", "small"),
            ("What is a REST API?", "small"),
            ("Name common sorting algorithms", "small"),
            ("What does this Python function do?", "small"),
            ("Explain supervised learning briefly", "small"),
            ("What is keyword search?", "small"),
            ("Give a concise explanation of SQL", "small"),

            # More involved questions: large model
            ("Compare two machine learning algorithms in detail", "large"),
            ("Design a scalable RAG architecture", "large"),
            ("Debug this complex Python application", "large"),
            ("Explain the tradeoffs between database designs", "large"),
            ("Analyze the time complexity of this algorithm", "large"),
            ("Develop a multi-step solution to this problem", "large"),
            ("Compare vector search and keyword search with examples", "large"),
            ("Design a reliable distributed system", "large"),
            ("Explain how to optimize a production API", "large"),
            ("Analyze the security risks of this architecture", "large"),
            ("Create a detailed implementation plan for a chatbot", "large"),
            ("Evaluate multiple approaches and justify the best solution", "large"),
            ("Debug this recursive algorithm and explain the fix", "large"),
            ("Compare alternative database indexing strategies", "large"),
            ("Design a fault-tolerant microservices architecture", "large"),
            ("Analyze performance bottlenecks across this application", "large"),
            ("Develop a step-by-step plan to migrate a database", "large"),
            ("Compare model-routing strategies and their tradeoffs", "large"),
            ("Explain how to improve retrieval quality in a RAG pipeline", "large"),
            ("Investigate why this API has high latency", "large"),
            ("Design an evaluation plan for a machine learning system", "large"),
            ("Review this architecture for reliability and security", "large"),
            ("Solve this problem and explain each reasoning step", "large"),
            ("Compare multiple approaches using performance constraints", "large"),
        ]

        training_queries = [item[0] for item in training_data]
        training_labels = [item[1] for item in training_data]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
        )

        features = self.vectorizer.fit_transform(training_queries)

        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.classifier.fit(features, training_labels)

        small_model = os.getenv(
            "ROUTER_SMALL_MODEL",
            "ollama_chat/qwen2.5:0.5b",
        )
        large_model = os.getenv(
            "ROUTER_LARGE_MODEL",
            "ollama_chat/llama3.2:1b",
        )

        self.clients = {
            "small": self._create_client("SMALL", small_model),
            "large": self._create_client("LARGE", large_model),
        }

        self.model_names = {
            "small": self.clients["small"].display_name,
            "large": self.clients["large"].display_name,
        }

        self.confidence_threshold = 0.65

    @staticmethod
    def _create_client(route, model):
        api_base = os.getenv(f"ROUTER_{route}_API_BASE")
        api_key = os.getenv(f"ROUTER_{route}_API_KEY")
        if model.startswith(("ollama_chat/", "ollama/")):
            base_url = (
                api_base
                or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ).rstrip("/")
            api_base = base_url.removesuffix("/v1")
            return LiteLLMClient(
                model=model,
                api_base=api_base,
                api_key=api_key,
            )
        return LiteLLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
        )

    def choose_model(self, query):
        self._validate_query(query)

        features = self.vectorizer.transform([query])
        probabilities = self.classifier.predict_proba(features)[0]

        best_index = probabilities.argmax()
        prediction = self.classifier.classes_[best_index]
        confidence = float(probabilities[best_index])

        if confidence < self.confidence_threshold:
            feedback_preference = (
                self.feedback_preference_provider(
                    self.model_names["small"],
                    self.model_names["large"],
                )
                if self.feedback_preference_provider is not None
                else None
            )
            prediction = (
                feedback_preference
                if feedback_preference in {"small", "large"}
                else "large"
            )

        return prediction, confidence

    def generate(self, query, prompt=None):
        model_type, confidence = self.choose_model(query)
        client = self.clients[model_type]

        input_text = prompt if prompt else query

        start_time = time.perf_counter()
        model_response = client.generate_response(input_text)
        latency = time.perf_counter() - start_time

        return {
            "query": query,
            "model": self.model_names[model_type],
            "category": model_type,
            "confidence": round(confidence, 3),
            "latency_seconds": round(latency, 2),
            "response": model_response.text,
            "input_tokens": model_response.input_tokens,
            "output_tokens": model_response.output_tokens,
            "cost_usd": model_response.cost_usd,
        }
