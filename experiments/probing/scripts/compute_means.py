"""
compute_means.py — Compute and save domain means, global means, and concept vectors.

Domain concept vector: d_k^ℓ = μ_k^ℓ − μ^ℓ
where μ_k^ℓ is the mean activation for domain k at layer ℓ,
and μ^ℓ is the global mean across all domains.

Usage:
    python compute_means.py                         # uses config.yaml
    python compute_means.py --config ../config.yaml # explicit config
"""
import argparse
import os
import sys
import time
import numpy as np
import h5py
import yaml

# ── Project imports ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.logger import setup_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from loader import ActivationLoader


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Compute domain means and concept vectors")
    p.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    return p.parse_args()


def compute_domain_mean(loader, domain, component, mode, num_layers, hidden_dim,
                        batch_size=1000):
    """
    Compute the mean activation for one domain, one component, streaming.

    Returns
    -------
    mean : np.ndarray, shape (L, d)
    count : int
    """
    path = loader.h5_path(domain)
    with h5py.File(path, "r") as f:
        dataset = f[f"{component}/{mode}"]
        N = dataset.shape[0]

        running_sum = np.zeros((num_layers, hidden_dim), dtype=np.float64)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = dataset[start:end].astype(np.float64)  # (B, L, d)
            running_sum += batch.sum(axis=0)

        mean = running_sum / N
    return mean, N


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve paths
    from utils.model_loader import get_model_slug
    model_slug = get_model_slug(cfg["model"]["name"])
    results_base = os.path.join(os.path.dirname(args.config), cfg.get("results_dir", "./results"))
    results_dir = os.path.join(results_base, model_slug)
    os.makedirs(results_dir, exist_ok=True)

    domains = cfg["data"]["domains"]
    components = cfg["extraction"]["components"]
    mode = cfg["extraction"]["pooling"]

    loader = ActivationLoader(results_dir, pooling=mode)

    # Get model dimensions from first activation file
    first_domain = domains[0]
    with h5py.File(loader.h5_path(first_domain), "r") as f:
        ds = f[f"{components[0]}/{mode}"]
        num_layers = ds.shape[1]
        hidden_dim = ds.shape[2]

    log_dir = os.path.join(results_dir, "logs")
    log = setup_logger("compute_means", log_dir=log_dir)

    out_path = os.path.join(results_dir, "domain_means.npz")

    log("Computing domain means and concept vectors")
    log(f"  Model: {cfg['model']['name']} ({model_slug})")
    log(f"  Components: {components}")
    log(f"  Mode: {mode}")
    log(f"  Domains: {domains}")
    log(f"  Layers: {num_layers}, Hidden: {hidden_dim}")
    log()

    t0 = time.time()

    # ── Compute per-domain means ──
    domain_means = {comp: {} for comp in components}
    sample_counts = {}

    for domain in domains:
        n = loader.get_num_samples(domain)
        sample_counts[domain] = n
        log(f"  {domain}: {n} samples")

        for comp in components:
            mean, _ = compute_domain_mean(loader, domain, comp, mode,
                                          num_layers, hidden_dim)
            domain_means[comp][domain] = mean
            log(f"    {comp}: mean computed")

    # ── Compute global means (weighted by sample count) ──
    total_samples = sum(sample_counts.values())
    global_means = {}

    for comp in components:
        weighted_sum = np.zeros((num_layers, hidden_dim), dtype=np.float64)
        for domain in domains:
            weighted_sum += sample_counts[domain] * domain_means[comp][domain]
        global_means[comp] = weighted_sum / total_samples

    # ── Compute domain concept vectors: d_k = μ_k − μ ──
    concept_vectors = {comp: {} for comp in components}
    for comp in components:
        for domain in domains:
            concept_vectors[comp][domain] = (
                domain_means[comp][domain] - global_means[comp]
            )

    # ── Save everything ──
    save_dict = {}

    # Sample counts
    save_dict["domains"] = np.array(domains)
    save_dict["sample_counts"] = np.array([sample_counts[d] for d in domains])

    for comp in components:
        # Global mean
        save_dict[f"global_mean_{comp}"] = global_means[comp].astype(np.float32)

        for domain in domains:
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
    mid_layer = num_layers // 2
    log(f"\n  Concept vector norms at layer {mid_layer} (mid-depth):")
    for comp in components:
        norms = [np.linalg.norm(concept_vectors[comp][d][mid_layer]) for d in domains]
        log(f"    {comp}: {['%.2f' % n for n in norms]}")


if __name__ == "__main__":
    main()
