"""
separability.py — Fisher Discriminant Ratio computation.
Pure math, no I/O.

Uses the trace identity:
  tr(S_W) = sum_k [ sum_{x in k} ||x||^2  -  n_k * ||mu_k||^2 ]
  tr(S_B) = sum_k n_k * ||d_k||^2     where d_k = mu_k - mu_global
  FDR = tr(S_B) / tr(S_W)
"""
import numpy as np


def fdr_from_stats(concept_vectors: np.ndarray,
                   sample_counts: np.ndarray,
                   sq_norm_sums: np.ndarray,
                   mean_sq_norms: np.ndarray) -> float:
    """
    Compute FDR from pre-aggregated statistics.

    Parameters
    ----------
    concept_vectors : (K, d) — domain concept vectors (mu_k - mu_global)
    sample_counts   : (K,)   — samples per domain
    sq_norm_sums    : (K,)   — sum of ||x||^2 per domain
    mean_sq_norms   : (K,)   — ||mu_k||^2 per domain

    Returns
    -------
    float : FDR = tr(S_B) / tr(S_W)
    """
    # Between-class: sum_k n_k * ||d_k||^2
    concept_sq = np.sum(concept_vectors ** 2, axis=1)  # (K,)
    tr_sb = np.sum(sample_counts * concept_sq)

    # Within-class: sum_k [ sq_sum_k - n_k * ||mu_k||^2 ]
    tr_sw = np.sum(sq_norm_sums - sample_counts * mean_sq_norms)

    if tr_sw < 1e-12:
        return 0.0
    return float(tr_sb / tr_sw)


def normalized_fdr(norm_means: np.ndarray,
                   sample_counts: np.ndarray) -> float:
    """
    FDR on unit-normalized activations.

    Since ||x/||x||||^2 = 1 for all x:
      tr(S_W) = sum_k n_k * (1 - ||mu_hat_k||^2)
      tr(S_B) = sum_k n_k * ||mu_hat_k - mu_hat_global||^2

    Parameters
    ----------
    norm_means     : (K, d) — mean of x/||x|| per domain
    sample_counts  : (K,)   — samples per domain

    Returns
    -------
    float : normalized FDR
    """
    total = sample_counts.sum()
    global_mean = (sample_counts[:, None] * norm_means).sum(axis=0) / total

    concept_vecs = norm_means - global_mean[None, :]
    concept_sq = np.sum(concept_vecs ** 2, axis=1)
    tr_sb = np.sum(sample_counts * concept_sq)

    mean_sq = np.sum(norm_means ** 2, axis=1)  # ||mu_hat_k||^2
    tr_sw = np.sum(sample_counts * (1.0 - mean_sq))

    if tr_sw < 1e-12:
        return 0.0
    return float(tr_sb / tr_sw)
