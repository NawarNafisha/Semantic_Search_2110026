"""Utility helpers for semantic search app."""

from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity search."""
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def adapt_vector_dim(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Adapt a vector to the target dimension by truncating or zero-padding.

    Why this exists:
    Some public datasets do not document which embedding model produced their
    vectors. If query and dataset dimensions differ, FAISS search fails.
    This helper keeps the server running instead of crashing.
    """
    vector = np.asarray(vector, dtype=np.float32)

    if vector.ndim == 2:
        vector = vector[0]

    current_dim = vector.shape[0]
    if current_dim == target_dim:
        return vector.reshape(1, -1)

    if current_dim > target_dim:
        return vector[:target_dim].reshape(1, -1)

    pad_width = target_dim - current_dim
    padded = np.pad(vector, (0, pad_width), mode="constant", constant_values=0.0)
    return padded.reshape(1, -1)


def shorten_text(text: str, max_chars: int = 300) -> str:
    """Return a short preview of long answer text for the UI."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
