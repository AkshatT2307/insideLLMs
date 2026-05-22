"""
run_geometry.py — Compute geometric analysis from saved domain means.
Works entirely from results/{model_slug}/domain_means.npz.

Usage:
    python run_geometry.py
    python run_geometry.py --config ../config.yaml
"""
import argparse
import os
import sys
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.logger import setup_logger
from utils.model_loader import get_model_slug

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from geometry import cosine_matrix, simplex_deviation, concept_norms


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Geometry analysis from domain means")
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_slug = get_model_slug(cfg["model"]["name"])
    results_base = os.path.join(os.path.dirname(args.config), cfg.get("results_dir", "./results"))
    results_dir = os.path.join(results_base, model_slug)

    domains = cfg["data"]["domains"]
    components = cfg["extraction"]["components"]

    log = setup_logger("run_geometry", log_dir=os.path.join(results_dir, "logs"))
    data = np.load(os.path.join(results_dir, "domain_means.npz"))
    out_dir = os.path.join(results_dir, "geometry")
    os.makedirs(out_dir, exist_ok=True)

    # Auto-detect num_layers from data
    first_key = f"concept_vector_{components[0]}_{domains[0]}"
    num_layers = data[first_key].shape[0]

    K = len(domains)
    log(f"Geometry analysis: {K} domains, {num_layers} layers")
    log(f"  Model: {cfg['model']['name']} ({model_slug})")

    # Storage
    all_cos = {}
    all_simplex = {}
    all_norms = {}

    for comp in components:
        cos_mats = np.zeros((num_layers, K, K))
        simp_devs = np.zeros(num_layers)
        norms_arr = np.zeros((num_layers, K))

        for layer in range(num_layers):
            vecs = np.stack([
                data[f"concept_vector_{comp}_{d}"][layer] for d in domains
            ])

            cos_mats[layer] = cosine_matrix(vecs)
            simp_devs[layer] = simplex_deviation(cos_mats[layer])
            norms_arr[layer] = concept_norms(vecs)

        all_cos[comp] = cos_mats
        all_simplex[comp] = simp_devs
        all_norms[comp] = norms_arr

        log(f"  {comp}: simplex_dev range [{simp_devs.min():.3f}, {simp_devs.max():.3f}], "
            f"best layer L{np.argmin(simp_devs)}")

    # Save
    save_dict = {"domains": np.array(domains)}
    for comp in components:
        save_dict[f"cosine_{comp}"] = all_cos[comp]
        save_dict[f"simplex_{comp}"] = all_simplex[comp]
        save_dict[f"norms_{comp}"] = all_norms[comp]

    np.savez(os.path.join(out_dir, "geometry_results.npz"), **save_dict)
    log(f"\n  Saved to {out_dir}/geometry_results.npz")


if __name__ == "__main__":
    main()
