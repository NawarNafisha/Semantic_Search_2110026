"""Core semantic search logic using sentence-transformers + FAISS."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import normalize_vectors, shorten_text


class SemanticSearchEngine:
    """In-memory semantic search engine for StackOverflow-like Q&A data."""

    def __init__(self, data_path: Path) -> None:
        # Path to JSON dataset file.
        self.data_path = data_path
        # In-memory records loaded from JSON.
        self.records: list[dict] = []
        # FAISS index (initialized on load).
        self.index: faiss.IndexFlatIP | None = None

        # Query embedding model (no API key needed).
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load(self) -> None:
        """Load records and build FAISS cosine-similarity index."""
        # Ensure dataset file exists.
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found at {self.data_path}. Run: python -m app.load_data"
            )

        # Read JSON data.
        with self.data_path.open("r", encoding="utf-8") as f:
            self.records = json.load(f)

        # Validate data is present.
        if not isinstance(self.records, list) or not self.records:
            raise ValueError("Dataset JSON is empty. Run: python -m app.load_data")

        # Build embedding matrix from precomputed dataset embeddings.
        embeddings = np.asarray([row["embeddings"] for row in self.records], dtype=np.float32)

        # Normalize vectors to support cosine similarity via inner product.
        embeddings = normalize_vectors(embeddings)

        # Create FAISS index with inner product metric.
        vector_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(vector_dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search top-k semantically similar records for a user query."""
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

        # Convert user query text into embedding.
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)

        # Normalize query embedding for cosine similarity.
        query_embedding = normalize_vectors(query_embedding)

        # Run vector search.
        k = min(top_k, len(self.records))
        scores, indices = self.index.search(query_embedding, k)

        # Format API response records.
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.records[int(idx)]
            results.append(
                {
                    "question": row["question"],
                    "answer": shorten_text(row["answer"]),
                    "score": float(score),
                }
            )

        return results
