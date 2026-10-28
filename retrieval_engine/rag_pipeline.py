
import json
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

import config
from vector_store.faiss_store import FAISSVectorStore
from vector_store.bm25 import BM25Okapi


class RetrievalPipeline:
    """
    Hybrid retrieval pipeline:
    - FAISS for semantic similarity
    - BM25 for keyword matching
    - Reciprocal Rank Fusion (RRF)
    - Optional Cross-Encoder reranking
    """

    def __init__(
        self,
        base_dir=None,
        use_cross_encoder=False,
        min_similarity=0.30,
    ):
        self.base_dir = base_dir
        self.use_cross_encoder = use_cross_encoder
        self.min_similarity = min_similarity

        # Load the embedding model.
        self.embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Load the saved FAISS index.
        self.vector_store = FAISSVectorStore.load(
            config.FAISS_INDEX
        )

        # Load stored document chunks and metadata.
        with open(
            config.EMBEDDING_METADATA,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        if not isinstance(self.metadata, list):
            raise ValueError(
                "Expected embedding metadata to be a list of chunks."
            )

        # Prepare the BM25 corpus.
        corpus = [
            item.get("text", "")
            for item in self.metadata
        ]

        self.bm25 = BM25Okapi(corpus)

        # Load Cross-Encoder only when requested.
        self.cross_encoder = None

        if self.use_cross_encoder:
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

    def search(
        self,
        query,
        top_k=5,
        rrf_k=60,
        candidate_pool_factor=4,
        allowed_sources=None,
    ):
        """
        Retrieve relevant chunks using hybrid search.

        Returns a list of dictionaries containing the text,
        metadata, dense similarity, BM25 score, RRF score,
        and retrieval ranks.

        Returns [] when the best FAISS similarity is below
        the configured threshold.
        """

        if not isinstance(query, str) or not query.strip():
            return []

        if not self.metadata:
            return []

        if top_k < 1:
            return []

        allowed_source_names = (
            {
                Path(str(source)).name.casefold()
                for source in allowed_sources
            }
            if allowed_sources is not None
            else None
        )
        if allowed_source_names is not None and not allowed_source_names:
            return []

        pool_size = len(self.metadata) if allowed_source_names else min(
            len(self.metadata),
            max(top_k, top_k * candidate_pool_factor),
        )

        def source_is_allowed(index):
            if allowed_source_names is None:
                return True
            chunk_metadata = self.metadata[index].get("metadata") or {}
            source = (
                chunk_metadata.get("filename")
                or chunk_metadata.get("document_name")
                or ""
            )
            return Path(str(source)).name.casefold() in allowed_source_names

        # --------------------------------------------------
        # 1. FAISS semantic search
        # --------------------------------------------------
        query_embedding = self.embed_model.encode(query)

        dense_scores, dense_indices = self.vector_store.search(
            query_embedding,
            top_k=pool_size,
        )

        dense_scores = np.asarray(dense_scores).reshape(-1)
        dense_indices = np.asarray(dense_indices).reshape(-1)

        dense_results = {}

        for rank, (score, idx) in enumerate(
            zip(dense_scores, dense_indices),
            start=1,
        ):
            idx = int(idx)

            if idx < 0 or idx >= len(self.metadata):
                continue
            if not source_is_allowed(idx):
                continue

            dense_results[idx] = {
                "score": float(score),
                "rank": rank,
            }

        # No valid FAISS results.
        if not dense_results:
            return []

        # --------------------------------------------------
        # 2. Relevance threshold
        # --------------------------------------------------
        # This assumes the FAISS index uses normalized vectors
        # and inner-product search, as in your project.
        best_dense_score = max(
            result["score"]
            for result in dense_results.values()
        )

        if best_dense_score < self.min_similarity:
            return []

        # --------------------------------------------------
        # 3. BM25 keyword search
        # --------------------------------------------------
        bm25_scores = np.asarray(
            self.bm25.get_scores(query)
        ).reshape(-1)

        sparse_indices = np.argsort(
            bm25_scores
        )[::-1][:pool_size]

        sparse_results = {}

        for rank, idx in enumerate(
            sparse_indices,
            start=1,
        ):
            idx = int(idx)

            if idx < 0 or idx >= len(self.metadata):
                continue
            if not source_is_allowed(idx):
                continue

            score = float(bm25_scores[idx])

            if score <= 0:
                continue

            sparse_results[idx] = {
                "score": score,
                "rank": rank,
            }

        # --------------------------------------------------
        # 4. Reciprocal Rank Fusion
        # --------------------------------------------------
        rrf_scores = {}

        for idx, result in dense_results.items():
            rrf_scores[idx] = (
                rrf_scores.get(idx, 0.0)
                + 1.0 / (rrf_k + result["rank"])
            )

        for idx, result in sparse_results.items():
            rrf_scores[idx] = (
                rrf_scores.get(idx, 0.0)
                + 1.0 / (rrf_k + result["rank"])
            )

        ranked_indices = sorted(
            rrf_scores.keys(),
            key=lambda idx: rrf_scores[idx],
            reverse=True,
        )

        # --------------------------------------------------
        # 5. Build retrieval results
        # --------------------------------------------------
        candidates = []

        for idx in ranked_indices:
            chunk = self.metadata[idx]

            candidates.append(
                {
                    "index": idx,
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                    "dense_score": (
                        dense_results[idx]["score"]
                        if idx in dense_results
                        else None
                    ),
                    "bm25_score": (
                        sparse_results[idx]["score"]
                        if idx in sparse_results
                        else 0.0
                    ),
                    "rrf_score": float(rrf_scores[idx]),
                    "dense_rank": (
                        dense_results[idx]["rank"]
                        if idx in dense_results
                        else None
                    ),
                    "sparse_rank": (
                        sparse_results[idx]["rank"]
                        if idx in sparse_results
                        else None
                    ),
                }
            )

        # --------------------------------------------------
        # 6. Optional Cross-Encoder reranking
        # --------------------------------------------------
        if self.cross_encoder is not None and candidates:
            pairs = [
                (query, candidate["text"])
                for candidate in candidates
            ]

            scores = self.cross_encoder.predict(pairs)

            for candidate, score in zip(candidates, scores):
                candidate["re_rank_score"] = float(score)

            candidates.sort(
                key=lambda item: item["re_rank_score"],
                reverse=True,
            )

        return candidates[:top_k]
