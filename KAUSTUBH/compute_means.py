"""
compute_means.py — Compute and save domain means, global means, and concept vectors.

Domain concept vector: d_k^ℓ = μ_k^ℓ − μ^ℓ
where μ_k^ℓ is the mean activation for domain k at layer ℓ,
and μ^ℓ is the global mean across all domains.

Usage:
    python compute_means.py
"""
import os
import time
import numpy as np
from config import DOMAINS, COMPONENTS, MODE, NUM_LAYERS, HIDDEN_DIM, RESULTS_DIR
from loader import h5_path, get_num_samples
from logger import setup_logger
import h5py


def compute_domain_mean(domain: str, component: str, mode: str,
                        batch_size: int = 1000) -> tuple:
    """
    Compute the mean activation for one domain, one component, streaming.

    Returns
    -------
    mean : np.ndarray, shape (L, d)
    count : int
    """
    path = h5_path(domain)
    with h5py.File(path, "r") as f:
        dataset = f[f"{component}/{mode}"]
        N = dataset.shape[0]

        running_sum = np.zeros((NUM_LAYERS, HIDDEN_DIM), dtype=np.float64)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = dataset[start:end].astype(np.float64)  # (B, L, d)
            running_sum += batch.sum(axis=0)

        mean = running_sum / N
    return mean, N


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "domain_means.npz")
    log = setup_logger("compute_means")

    log("Computing domain means and concept vectors")
    log(f"  Components: {COMPONENTS}")
    log(f"  Mode: {MODE}")
    log(f"  Domains: {DOMAINS}")
    log()

    t0 = time.time()

    # ── Compute per-domain means ──
    # domain_means[component][domain] = (L, d)
    domain_means = {comp: {} for comp in COMPONENTS}
    sample_counts = {}

    for domain in DOMAINS:
        n = get_num_samples(domain)
        sample_counts[domain] = n
        log(f"  {domain}: {n} samples")

        for comp in COMPONENTS:
            mean, _ = compute_domain_mean(domain, comp, MODE)
            domain_means[comp][domain] = mean
            log(f"    {comp}: mean computed")

    # ── Compute global means (weighted by sample count) ──
    total_samples = sum(sample_counts.values())
    global_means = {}

    for comp in COMPONENTS:
        weighted_sum = np.zeros((NUM_LAYERS, HIDDEN_DIM), dtype=np.float64)
        for domain in DOMAINS:
            weighted_sum += sample_counts[domain] * domain_means[comp][domain]
        global_means[comp] = weighted_sum / total_samples

    # ── Compute domain concept vectors: d_k = μ_k − μ ──
    concept_vectors = {comp: {} for comp in COMPONENTS}
    for comp in COMPONENTS:
        for domain in DOMAINS:
            concept_vectors[comp][domain] = (
                domain_means[comp][domain] - global_means[comp]
            )

    # ── Save everything ──
    save_dict = {}

    # Sample counts
    save_dict["domains"] = np.array(DOMAINS)
    save_dict["sample_counts"] = np.array([sample_counts[d] for d in DOMAINS])

    for comp in COMPONENTS:
        # Global mean
        save_dict[f"global_mean_{comp}"] = global_means[comp].astype(np.float32)

        for domain in DOMAINS:
            # Domain mean
            save_dict[f"domain_mean_{comp}_{domain}"] = (
                domain_means[comp][domain].astype(np.float32)
            )
            # Concept vector
            save_dict[f"concept_vector_{comp}_{domain}"] = (
                concept_vectors[comp][domain].astype(np.float32)
            )

    np.savez(out_path, **save_dict)
    elapsed = time.time() - t0

    log(f"\n  Saved to {out_path}")
    log(f"  Time: {elapsed:.1f}s")

    # ── Quick sanity check: concept vector norms ──
    log("\n  Concept vector norms at layer 14 (mid-depth):")
    for comp in COMPONENTS:
        norms = [np.linalg.norm(concept_vectors[comp][d][14]) for d in DOMAINS]
        log(f"    {comp}: {['%.2f' % n for n in norms]}")


if __name__ == "__main__":
    main()
