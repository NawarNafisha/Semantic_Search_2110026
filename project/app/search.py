"""Core semantic search logic using sentence-transformers + FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import adapt_vector_dim, normalize_vectors, shorten_text


class SemanticSearchEngine:
    """In-memory semantic search engine for StackOverflow Q&A."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.records: list[dict[str, Any]] = []
        self.index: faiss.IndexFlatIP | None = None
        self.embedding_dim: int | None = None

        # Lightweight model requiring no API key.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load(self) -> None:
        """Load JSON dataset and build FAISS cosine-similarity index."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found at {self.data_path}. "
                "Run: python -m app.load_data"
            )

        with self.data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            raise ValueError(
                "Dataset is empty or invalid JSON list. "
                "Run: python -m app.load_data"
            )

        cleaned_records: list[dict[str, Any]] = []
        cleaned_vectors: list[list[float]] = []

        for row in data:
            # Basic validation to prevent runtime crashes.
            if not isinstance(row, dict):
                continue
            if "question" not in row or "answer" not in row or "embeddings" not in row:
                continue

            vector = np.asarray(row["embeddings"], dtype=np.float32).flatten()
            if vector.size == 0:
                continue

            cleaned_records.append(row)
            cleaned_vectors.append(vector.tolist())

        if not cleaned_records:
            raise ValueError("No valid rows found in dataset.json")

        embeddings = np.asarray(cleaned_vectors, dtype=np.float32)
        embeddings = normalize_vectors(embeddings)

        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.records = cleaned_records

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k semantically similar Q&A pairs."""
        if self.index is None or self.embedding_dim is None:
            raise RuntimeError("Search index not initialized.")

        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        query_embedding = adapt_vector_dim(query_embedding, self.embedding_dim)
        query_embedding = normalize_vectors(query_embedding)

        k = min(top_k, len(self.records))
        scores, indices = self.index.search(query_embedding, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            record = self.records[int(idx)]
            results.append(
                {
                    "question": str(record["question"]),
                    "answer": shorten_text(str(record["answer"])),
                    "score": float(score),
                }
            )

        return results
