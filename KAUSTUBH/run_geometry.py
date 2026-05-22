"""
run_geometry.py — Compute geometric analysis from saved domain means.
Works entirely from results/domain_means.npz. No streaming needed.

Usage:
    python run_geometry.py
"""
import os
import numpy as np
from config import DOMAINS, COMPONENTS, NUM_LAYERS, RESULTS_DIR
from geometry import cosine_matrix, simplex_deviation, concept_norms
from logger import setup_logger


def main():
    log = setup_logger("run_geometry")
    data = np.load(os.path.join(RESULTS_DIR, "domain_means.npz"))
    out_dir = os.path.join(RESULTS_DIR, "geometry")
    os.makedirs(out_dir, exist_ok=True)

    K = len(DOMAINS)
    log(f"Geometry analysis: {K} domains, {NUM_LAYERS} layers")

    # Storage
    all_cos = {}       # [comp] → (L, K, K)
    all_simplex = {}   # [comp] → (L,)
    all_norms = {}     # [comp] → (L, K)

    for comp in COMPONENTS:
        cos_mats = np.zeros((NUM_LAYERS, K, K))
        simp_devs = np.zeros(NUM_LAYERS)
        norms_arr = np.zeros((NUM_LAYERS, K))

        for layer in range(NUM_LAYERS):
            vecs = np.stack([
                data[f"concept_vector_{comp}_{d}"][layer] for d in DOMAINS
            ])  # (K, d)

            cos_mats[layer] = cosine_matrix(vecs)
            simp_devs[layer] = simplex_deviation(cos_mats[layer])
            norms_arr[layer] = concept_norms(vecs)

        all_cos[comp] = cos_mats
        all_simplex[comp] = simp_devs
        all_norms[comp] = norms_arr

        log(f"  {comp}: simplex_dev range [{simp_devs.min():.3f}, {simp_devs.max():.3f}], "
            f"best layer L{np.argmin(simp_devs)}")

    # Save
    save_dict = {"domains": np.array(DOMAINS)}
    for comp in COMPONENTS:
        save_dict[f"cosine_{comp}"] = all_cos[comp]
        save_dict[f"simplex_{comp}"] = all_simplex[comp]
        save_dict[f"norms_{comp}"] = all_norms[comp]

    np.savez(os.path.join(out_dir, "geometry_results.npz"), **save_dict)
    log(f"\n  Saved to {out_dir}/geometry_results.npz")


if __name__ == "__main__":
    main()
