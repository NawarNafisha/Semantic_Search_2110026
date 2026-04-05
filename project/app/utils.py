"""Utility helper functions used by the semantic search app."""

from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors so inner product equals cosine similarity."""
    # Convert to float32 because FAISS expects float32 vectors.
    vectors = np.asarray(vectors, dtype=np.float32)

    # Ensure input is always 2D: (n_vectors, dim).
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Compute norms and avoid division-by-zero.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return vectors / norms


def shorten_text(text: str, max_chars: int = 300) -> str:
    """Return a short preview of answer text for the frontend cards."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
