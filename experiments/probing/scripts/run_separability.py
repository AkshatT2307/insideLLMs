"""
run_separability.py — Compute raw + normalized FDR across all layers.

Single streaming pass per domain/component to collect:
  - sum(||x||^2)  per layer   (for raw within-class scatter)
  - sum(x/||x||)  per layer   (for normalized means)

Usage:
    python run_separability.py
    python run_separability.py --config ../config.yaml
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import h5py
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.logger import setup_logger
from utils.model_loader import get_model_slug

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from loader import ActivationLoader
from separability import fdr_from_stats, normalized_fdr


BATCH_SIZE = 500


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Compute FDR separability")
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    return p.parse_args()


def stream_stats(loader, domain, comp, mode, num_layers, hidden_dim):
    """
    Single pass: compute sq_norm_sums and norm_mean_sums for all layers.
    """
    path = loader.h5_path(domain)
    with h5py.File(path, "r") as f:
        ds = f[f"{comp}/{mode}"]
        N = ds.shape[0]

        sq_sums = np.zeros(num_layers, dtype=np.float64)
        norm_sums = np.zeros((num_layers, hidden_dim), dtype=np.float64)

        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            batch = ds[start:end].astype(np.float32)

            sq_sums += (batch.astype(np.float64) ** 2).sum(axis=(0, 2))

            norms = np.linalg.norm(batch, axis=2, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            norm_sums += (batch / norms).astype(np.float64).sum(axis=0)

    return sq_sums, norm_sums, N


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_slug = get_model_slug(cfg["model"]["name"])
    results_base = os.path.join(os.path.dirname(args.config), cfg.get("results_dir", "./results"))
    results_dir = os.path.join(results_base, model_slug)

    domains = cfg["data"]["domains"]
    components = cfg["extraction"]["components"]
    mode = cfg["extraction"]["pooling"]

    loader = ActivationLoader(results_dir, pooling=mode)

    # Auto-detect dimensions
    with h5py.File(loader.h5_path(domains[0]), "r") as f:
        ds = f[f"{components[0]}/{mode}"]
        num_layers = ds.shape[1]
        hidden_dim = ds.shape[2]

    log = setup_logger("run_separability", log_dir=os.path.join(results_dir, "logs"))
    out_dir = os.path.join(results_dir, "separability")
    os.makedirs(out_dir, exist_ok=True)
    means_data = np.load(os.path.join(results_dir, "domain_means.npz"))

    sample_counts = np.array([loader.get_num_samples(d) for d in domains])
    log(f"Separability (FDR) — raw + normalized")
    log(f"  Model: {cfg['model']['name']} ({model_slug})")
    log(f"  Domains: {domains}")
    log(f"  Sample counts: {list(sample_counts)}")
    log()

    t0 = time.time()

    # ── Stream to get stats ──
    stats = {comp: {} for comp in components}

    for comp in components:
        for domain in domains:
            sq, ns, n = stream_stats(loader, domain, comp, mode, num_layers, hidden_dim)
            stats[comp][domain] = (sq, ns)
            log(f"  Streamed {comp}/{domain} ({n} samples)")

    # ── Compute FDR per layer ──
    results = {}
    for comp in components:
        raw_fdr = np.zeros(num_layers)
        norm_fdr = np.zeros(num_layers)

        for layer in range(num_layers):
            cvecs = np.stack([
                means_data[f"concept_vector_{comp}_{d}"][layer] for d in domains
            ])

            dmeans = np.stack([
                means_data[f"domain_mean_{comp}_{d}"][layer] for d in domains
            ])
            mean_sq = np.sum(dmeans ** 2, axis=1)

            sq_arr = np.array([stats[comp][d][0][layer] for d in domains])

            raw_fdr[layer] = fdr_from_stats(cvecs, sample_counts, sq_arr, mean_sq)

            norm_means = np.stack([
                stats[comp][d][1][layer] / loader.get_num_samples(d)
                for d in domains
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
    save_dict = {"domains": np.array(domains), "sample_counts": sample_counts}
    for comp in components:
        save_dict[f"raw_fdr_{comp}"] = results[comp]["raw"]
        save_dict[f"norm_fdr_{comp}"] = results[comp]["norm"]
    np.savez(os.path.join(out_dir, "fdr_results.npz"), **save_dict)

    summary = {"model": cfg["model"]["name"], "elapsed_seconds": round(elapsed, 1), "components": {}}
    for comp in components:
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
