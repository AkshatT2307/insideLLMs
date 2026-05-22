"""
geometry.py — Cosine similarity, simplex deviation, concept vector norms.
Pure math, no I/O.
"""
import numpy as np


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. vectors: (K, d) → (K, K)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    unit = vectors / norms
    return unit @ unit.T


def simplex_deviation(cos_mat: np.ndarray) -> float:
    """Frobenius distance from ideal simplex. Ideal off-diag = -1/(K-1)."""
    K = cos_mat.shape[0]
    ideal = np.full_like(cos_mat, -1.0 / (K - 1))
    np.fill_diagonal(ideal, 1.0)
    return float(np.linalg.norm(cos_mat - ideal, "fro"))


def concept_norms(vectors: np.ndarray) -> np.ndarray:
    """L2 norms of each vector. vectors: (K, d) → (K,)."""
    return np.linalg.norm(vectors, axis=1)
