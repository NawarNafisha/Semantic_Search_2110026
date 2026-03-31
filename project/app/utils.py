"""Utility helpers for semantic search app."""

from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity search.

    FAISS can perform cosine similarity by normalizing vectors and using
    inner-product search.
    """
    # Convert to float32 because FAISS expects float32 arrays.
    vectors = np.asarray(vectors, dtype=np.float32)

    # Compute norms and avoid division by zero.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return vectors / norms


def shorten_text(text: str, max_chars: int = 300) -> str:
    """Return a short preview of long answer text for the UI."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
