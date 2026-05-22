"""
run_separability.py — Compute raw + normalized FDR across all layers.

Single streaming pass per domain/component to collect:
  - sum(||x||^2)  per layer   (for raw within-class scatter)
  - sum(x/||x||)  per layer   (for normalized means)

Usage:
    python run_separability.py
"""
import os
import json
import time
import numpy as np
import h5py
from config import DOMAINS, COMPONENTS, NUM_LAYERS, HIDDEN_DIM, MODE, RESULTS_DIR
from loader import h5_path, get_num_samples
from separability import fdr_from_stats, normalized_fdr
from logger import setup_logger


BATCH_SIZE = 500


def stream_stats(domain: str, comp: str, mode: str):
    """
    Single pass: compute sq_norm_sums and norm_mean_sums for all layers.

    Returns
    -------
    sq_sums   : (L,)     — sum of ||x||^2 per layer
    norm_sums : (L, d)   — sum of x/||x|| per layer
    n         : int
    """
    path = h5_path(domain)
    with h5py.File(path, "r") as f:
        ds = f[f"{comp}/{mode}"]
        N = ds.shape[0]

        sq_sums = np.zeros(NUM_LAYERS, dtype=np.float64)
        norm_sums = np.zeros((NUM_LAYERS, HIDDEN_DIM), dtype=np.float64)

        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            batch = ds[start:end].astype(np.float32)  # (B, L, d)

            # sum ||x||^2 per layer
            sq_sums += (batch.astype(np.float64) ** 2).sum(axis=(0, 2))

            # sum x/||x|| per layer
            norms = np.linalg.norm(batch, axis=2, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            norm_sums += (batch / norms).astype(np.float64).sum(axis=0)

    return sq_sums, norm_sums, N


def main():
    log = setup_logger("run_separability")
    out_dir = os.path.join(RESULTS_DIR, "separability")
    os.makedirs(out_dir, exist_ok=True)
    means_data = np.load(os.path.join(RESULTS_DIR, "domain_means.npz"))

    sample_counts = np.array([get_num_samples(d) for d in DOMAINS])
    log(f"Separability (FDR) — raw + normalized")
    log(f"  Domains: {DOMAINS}")
    log(f"  Sample counts: {list(sample_counts)}")
    log()

    t0 = time.time()

    # ── Stream to get stats ──
    # stats[comp][domain] = (sq_sums, norm_sums)
    stats = {comp: {} for comp in COMPONENTS}

    for comp in COMPONENTS:
        for domain in DOMAINS:
            sq, ns, n = stream_stats(domain, comp, MODE)
            stats[comp][domain] = (sq, ns)
            log(f"  Streamed {comp}/{domain} ({n} samples)")

    # ── Compute FDR per layer ──
    results = {}
    for comp in COMPONENTS:
        raw_fdr = np.zeros(NUM_LAYERS)
        norm_fdr = np.zeros(NUM_LAYERS)

        for layer in range(NUM_LAYERS):
            # Concept vectors from saved means
            cvecs = np.stack([
                means_data[f"concept_vector_{comp}_{d}"][layer] for d in DOMAINS
            ])  # (K, d)

            # Domain means for ||mu_k||^2
            dmeans = np.stack([
                means_data[f"domain_mean_{comp}_{d}"][layer] for d in DOMAINS
            ])
            mean_sq = np.sum(dmeans ** 2, axis=1)  # (K,)

            # sq_norm_sums per domain
            sq_arr = np.array([stats[comp][d][0][layer] for d in DOMAINS])

            raw_fdr[layer] = fdr_from_stats(cvecs, sample_counts, sq_arr, mean_sq)

            # Normalized means per domain
            norm_means = np.stack([
                stats[comp][d][1][layer] / get_num_samples(d)
                for d in DOMAINS
            ])
            norm_fdr[layer] = normalized_fdr(norm_means, sample_counts)

        results[comp] = {"raw": raw_fdr, "norm": norm_fdr}

        raw_peak = int(np.argmax(raw_fdr))
        norm_peak = int(np.argmax(norm_fdr))
        log(f"\n  {comp}:")
        log(f"    Raw  FDR: peak L{raw_peak} ({raw_fdr[raw_peak]:.6f}), "
            f"CV={np.std(raw_fdr)/np.mean(raw_fdr):.3f}")
        log(f"    Norm FDR: peak L{norm_peak} ({norm_fdr[norm_peak]:.6f}), "
            f"CV={np.std(norm_fdr)/np.mean(norm_fdr):.3f}")

    elapsed = time.time() - t0

    # ── Save ──
    save_dict = {"domains": np.array(DOMAINS), "sample_counts": sample_counts}
    for comp in COMPONENTS:
        save_dict[f"raw_fdr_{comp}"] = results[comp]["raw"]
        save_dict[f"norm_fdr_{comp}"] = results[comp]["norm"]
    np.savez(os.path.join(out_dir, "fdr_results.npz"), **save_dict)

    # JSON summary
    summary = {"elapsed_seconds": round(elapsed, 1), "components": {}}
    for comp in COMPONENTS:
        for kind in ["raw", "norm"]:
            arr = results[comp][kind]
            peak = int(np.argmax(arr))
            key = f"{comp}_{kind}"
            summary["components"][key] = {
                "values": [round(float(v), 6) for v in arr],
                "peak_layer": peak,
                "peak_value": round(float(arr[peak]), 6),
                "mean": round(float(np.mean(arr)), 6),
                "cv": round(float(np.std(arr) / np.mean(arr)), 4),
            }
    with open(os.path.join(out_dir, "fdr_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n  Time: {elapsed:.0f}s")
    log(f"  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
