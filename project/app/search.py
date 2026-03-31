"""Core semantic search logic using sentence-transformers + FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import normalize_vectors, shorten_text


class SemanticSearchEngine:
    """In-memory semantic search engine for StackOverflow Q&A."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.records: List[dict] = []
        self.index: faiss.IndexFlatIP | None = None

        # Use same model for query encoding (as requested).
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load(self) -> None:
        """Load JSON dataset and build FAISS cosine-similarity index."""
        with self.data_path.open("r", encoding="utf-8") as f:
            self.records = json.load(f)

        if not self.records:
            raise ValueError("Dataset is empty. Run app/load_data.py first.")

        # Stack embeddings into a matrix.
        embeddings = np.array([r["embeddings"] for r in self.records], dtype=np.float32)
        embeddings = normalize_vectors(embeddings)

        # IndexFlatIP + normalized vectors => cosine similarity.
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k semantically similar Q&A pairs."""
        if self.index is None:
            raise RuntimeError("Index not initialized. Call load() first.")

        # Convert input text into embedding.
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        query_embedding = normalize_vectors(query_embedding)

        # Retrieve nearest vectors.
        scores, indices = self.index.search(query_embedding, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            record = self.records[int(idx)]
            results.append(
                {
                    "question": record["question"],
                    "answer": shorten_text(record["answer"]),
                    "score": float(score),
                }
            )

        return results
