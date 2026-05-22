#!/usr/bin/env python3
"""
regularized_metrics.py

Implements:
1. Accurate FDR  — Regularized multiclass FDR: tr(S_W_reg^{-1} S_B) with Ledoit-Wolf shrinkage
2. OvA FDR       — Per-class One-vs-All FDR using pooled regularized S_W

Both share the expensive S_W matrix computation (streamed from activation files).
"""

import os
import numpy as np
import h5py
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# OAS Shrinkage (no sklearn dependency needed)
# ─────────────────────────────────────────────────────────────────────────────

def oas_shrinkage(S_W, n_eff, d):
    """
    Oracle Approximating Shrinkage for within-class scatter matrix.

    Parameters
    ----------
    S_W   : (d, d) raw within-class scatter matrix
    n_eff : effective sample count (n_total for OAS formula)
    d     : dimensionality

    Returns
    -------
    S_W_reg : (d, d) regularized scatter (positive definite)
    alpha   : shrinkage coefficient in [0, 1]
    """
    # Covariance = S_W / n_eff
    tr_sigma = np.trace(S_W) / n_eff
    # ||Sigma||_F^2 = ||S_W||_F^2 / n_eff^2
    frob_sq_sigma = np.sum(S_W ** 2) / (n_eff ** 2)

    numerator = ((n_eff - 2) / n_eff) * frob_sq_sigma + tr_sigma ** 2
    denominator = (n_eff + 2) * (frob_sq_sigma - tr_sigma ** 2 / d)

    if denominator <= 0 or np.isnan(denominator):
        alpha = 1.0  # full shrinkage to identity
    else:
        alpha = float(np.clip(numerator / denominator, 0.0, 1.0))

    # S_W_reg = (1 - alpha) * S_W + alpha * (tr(S_W)/d) * I
    tr_sw = np.trace(S_W)
    S_W_reg = (1.0 - alpha) * S_W + alpha * (tr_sw / d) * np.eye(d)

    return S_W_reg, alpha


# ─────────────────────────────────────────────────────────────────────────────
# Stream within-class scatter MATRICES (full d×d) for all layers at once
# ─────────────────────────────────────────────────────────────────────────────

def build_scatter_matrices(
    domain_means, sample_counts, activations_dir,
    component, mode, batch_size=500,
):
    """
    Stream through activation files and build the full within-class scatter
    matrix S_W^ℓ ∈ R^{d×d} for every layer simultaneously.

    Returns
    -------
    S_W_layers : list of np.ndarray, each (d, d) in float64
    """
    domains = list(domain_means.keys())
    mu0 = domain_means[domains[0]][component][mode]
    num_layers, hidden_dim = mu0.shape

    # Allocate all scatter matrices (L × d × d) — ~2.8 GB for L=28, d=3584
    S_W_layers = [np.zeros((hidden_dim, hidden_dim), dtype=np.float64)
                  for _ in range(num_layers)]

    for d in domains:
        mu_k = domain_means[d][component][mode].astype(np.float64)  # (L, D)
        act_path = os.path.join(activations_dir, f"{d}_activations.h5")

        with h5py.File(act_path, "r") as f:
            dataset = f[f"{component}/{mode}"]  # (N, L, D)
            N = dataset.shape[0]

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = dataset[start:end].astype(np.float64)  # (B, L, D)

                for ell in range(num_layers):
                    x_l = batch[:, ell, :]           # (B, D)
                    centered = x_l - mu_k[ell]       # (B, D)
                    S_W_layers[ell] += centered.T @ centered  # (D, D)

                if start % (batch_size * 5) == 0:
                    print(f"      {d}: {end}/{N}", end="\r", flush=True)

        print(f"      {d}: {N}/{N} ✓")

    return S_W_layers


# ─────────────────────────────────────────────────────────────────────────────
# Compute both Accurate FDR and OvA FDR (shared S_W computation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_regularized_fdr_and_ova(
    domain_vectors, domain_means, sample_counts, domains,
    activations_dir, component, mode, batch_size=500,
):
    """
    Compute Accurate FDR and OvA FDR for all layers of one (component, mode).

    Uses a single streaming pass to build S_W, then for each layer:
    1. Regularize S_W via OAS
    2. Cholesky factorize
    3. Low-rank solve for Accurate FDR = tr(V^T S_W_reg^{-1} V)
    4. Per-class OvA FDR from the same solve

    Returns
    -------
    accurate_fdr : (L,) array
    ova_scores   : dict {domain: (L,) array}  — per-class OvA FDR
    ova_hmean    : (L,) array — harmonic mean across classes
    ova_amean    : (L,) array — arithmetic mean across classes
    alphas       : (L,) array — shrinkage coefficients
    """
    K = len(domains)
    mu0 = domain_means[domains[0]][component][mode]
    num_layers, hidden_dim = mu0.shape
    n_total = sum(sample_counts[d] for d in domains)

    print(f"    Building S_W matrices ({component}/{mode})...")
    S_W_layers = build_scatter_matrices(
        domain_means, sample_counts, activations_dir,
        component, mode, batch_size,
    )

    # Prepare domain vectors for all layers: V[:,k] = sqrt(n_k) * d_k
    # d_k has shape (L, D)
    accurate_fdr = np.zeros(num_layers, dtype=np.float64)
    ova_scores = {d: np.zeros(num_layers, dtype=np.float64) for d in domains}
    ova_hmean = np.zeros(num_layers, dtype=np.float64)
    ova_amean = np.zeros(num_layers, dtype=np.float64)
    alphas = np.zeros(num_layers, dtype=np.float64)

    print(f"    Regularizing and solving per layer...")
    for ell in range(num_layers):
        S_W = S_W_layers[ell]

        # OAS regularization
        S_W_reg, alpha = oas_shrinkage(S_W, n_total, hidden_dim)
        alphas[ell] = alpha

        # Cholesky factorization (done once per layer)
        try:
            cho = cho_factor(S_W_reg)
        except np.linalg.LinAlgError:
            # Fallback: add stronger diagonal regularization
            eps = 1e-6 * np.trace(S_W_reg) / hidden_dim
            S_W_reg += eps * np.eye(hidden_dim)
            cho = cho_factor(S_W_reg)

        # Build V matrix: (D, K) where V[:,k] = sqrt(n_k) * d_k[ell]
        V = np.zeros((hidden_dim, K), dtype=np.float64)
        nk_arr = np.zeros(K, dtype=np.float64)
        for ki, d in enumerate(domains):
            nk = sample_counts[d]
            nk_arr[ki] = nk
            dk = domain_vectors[d][component][mode][ell].astype(np.float64)
            V[:, ki] = np.sqrt(nk) * dk

        # Solve S_W_reg Z = V → Z = S_W_reg^{-1} V, shape (D, K)
        Z = cho_solve(cho, V)

        # Accurate FDR = tr(V^T Z) = sum_k V[:,k]^T Z[:,k]
        accurate_fdr[ell] = np.sum(V * Z)

        # Per-class OvA FDR
        # FDR_k^OvA = (n / (n - n_k))^2 * d_k^T S_W_reg^{-1} d_k
        # d_k^T S_W_reg^{-1} d_k = (1/n_k) * V[:,k]^T Z[:,k]
        per_class = np.zeros(K)
        for ki, d in enumerate(domains):
            nk = nk_arr[ki]
            dk_Sinv_dk = np.dot(V[:, ki], Z[:, ki]) / nk
            scale = (n_total / (n_total - nk)) ** 2
            ova_val = scale * dk_Sinv_dk
            ova_scores[d][ell] = ova_val
            per_class[ki] = ova_val

        ova_amean[ell] = np.mean(per_class)
        # Harmonic mean (guard against zeros)
        safe = per_class + 1e-12
        ova_hmean[ell] = K / np.sum(1.0 / safe)

        if ell % 5 == 0 or ell == num_layers - 1:
            print(f"      Layer {ell}: accurate_fdr={accurate_fdr[ell]:.4f}, "
                  f"ova_hmean={ova_hmean[ell]:.4f}, α={alpha:.4f}")

    # Free scatter matrices
    del S_W_layers

    return accurate_fdr, ova_scores, ova_hmean, ova_amean, alphas


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"attn": "#FF6B6B", "mlp": "#4ECDC4", "layer": "#6C5CE7"}
LABELS = {"attn": "Attention Only", "mlp": "MLP Only", "layer": "Full Residual Stream"}


def plot_metric_curves(results, mode, metric_name, ylabel, output_path, num_layers):
    """Plot 3-curve (attn/mlp/layer) line chart for a single mode."""
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(num_layers)

    for comp in ["attn", "mlp", "layer"]:
        ax.plot(layers, results[comp], color=COLORS[comp], linewidth=2.2,
                label=LABELS[comp], marker="o", markersize=4, alpha=0.9)

    mode_title = "Mean-Pool" if mode == "mean" else "Last-Token"
    ax.set_title(f"{metric_name} — {mode_title} Strategy",
                 fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=12, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=11)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_ova_heatmap(ova_scores, domains, mode, component, output_path, num_layers):
    """Plot K×L heatmap of per-class OvA FDR scores."""
    K = len(domains)
    matrix = np.zeros((K, num_layers))
    for ki, d in enumerate(domains):
        matrix[ki] = ova_scores[d]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(K))
    ax.set_yticklabels(domains, fontsize=11)
    ax.set_xlabel("Layer", fontsize=13)
    comp_label = {"attn": "Attention", "mlp": "MLP", "layer": "Full Residual"}[component]
    mode_title = "Mean-Pool" if mode == "mean" else "Last-Token"
    ax.set_title(f"Per-Class OvA FDR — {comp_label} ({mode_title})",
                 fontsize=14, fontweight="bold", pad=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("OvA FDR", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_regularized(activations_dir, vectors_file, output_dir, batch_size=500):
    """
    Compute Accurate FDR and OvA FDR for all components and modes.
    Single streaming pass per (component, mode) — shared S_W computation.
    """
    from calculate_separability import load_domain_data

    os.makedirs(output_dir, exist_ok=True)

    domain_vectors, domain_means, sample_counts, domains = load_domain_data(
        vectors_file, activations_dir
    )
    num_layers = domain_vectors[domains[0]]["attn"]["mean"].shape[0]
    K = len(domains)

    print(f"Domains: {domains}")
    print(f"Sample counts: {sample_counts}")
    print(f"Layers: {num_layers}, Hidden dim: {domain_vectors[domains[0]]['attn']['mean'].shape[1]}")
    print(f"Classes: {K}\n")

    components = ["attn", "mlp", "layer"]
    modes = ["mean", "last"]

    # Storage for all results
    acc_results = {}   # {mode: {comp: array}}
    ova_results = {}   # {mode: {comp: {domain: array}}}
    ova_hm = {}        # {mode: {comp: array}}
    ova_am = {}        # {mode: {comp: array}}

    for mode in modes:
        print(f"{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")
        acc_results[mode] = {}
        ova_results[mode] = {}
        ova_hm[mode] = {}
        ova_am[mode] = {}

        for comp in components:
            print(f"\n  Component: {comp}")
            afdr, ova_s, hmean, amean, alphas = compute_regularized_fdr_and_ova(
                domain_vectors, domain_means, sample_counts, domains,
                activations_dir, comp, mode, batch_size,
            )
            acc_results[mode][comp] = afdr
            ova_results[mode][comp] = ova_s
            ova_hm[mode][comp] = hmean
            ova_am[mode][comp] = amean

            print(f"    Accurate FDR range: [{afdr.min():.6f}, {afdr.max():.6f}]")
            print(f"    OvA Hmean range:    [{hmean.min():.6f}, {hmean.max():.6f}]")

    # ── Save Accurate FDR ────────────────────────────────────────────────
    save_acc = {}
    for mode in modes:
        for comp in components:
            save_acc[f"{mode}_{comp}_fdr"] = acc_results[mode][comp]
    save_acc["domains"] = np.array(domains)
    npz_path = os.path.join(output_dir, "accurate_fdr_scores.npz")
    np.savez(npz_path, **save_acc)
    print(f"\n  Accurate FDR saved: {npz_path}")

    # ── Save OvA FDR ─────────────────────────────────────────────────────
    save_ova = {}
    for mode in modes:
        for comp in components:
            save_ova[f"{mode}_{comp}_hmean"] = ova_hm[mode][comp]
            save_ova[f"{mode}_{comp}_amean"] = ova_am[mode][comp]
            for d in domains:
                save_ova[f"{mode}_{comp}_{d}"] = ova_results[mode][comp][d]
    save_ova["domains"] = np.array(domains)
    npz_path = os.path.join(output_dir, "ova_fdr_scores.npz")
    np.savez(npz_path, **save_ova)
    print(f"  OvA FDR saved: {npz_path}")

    # ── Plots: Accurate FDR (3-curve, one per mode) ──────────────────────
    for mode in modes:
        plot_metric_curves(
            acc_results[mode], mode,
            "Accurate FDR (Regularized)", r"FDR = tr($\hat{S}_W^{-1} S_B$)",
            os.path.join(output_dir, f"accurate_fdr_{mode}.png"),
            num_layers,
        )

    # ── Plots: OvA FDR harmonic mean (3-curve, one per mode) ─────────────
    for mode in modes:
        plot_metric_curves(
            ova_hm[mode], mode,
            "OvA FDR (Harmonic Mean)", "OvA FDR (harmonic mean)",
            os.path.join(output_dir, f"ova_fdr_{mode}.png"),
            num_layers,
        )

    # ── Plots: OvA heatmaps ──────────────────────────────────────────────
    for mode in modes:
        for comp in components:
            plot_ova_heatmap(
                ova_results[mode][comp], domains, mode, comp,
                os.path.join(output_dir, f"ova_heatmap_{comp}_{mode}.png"),
                num_layers,
            )

    print("\nAll regularized metrics done!")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Compute regularized FDR and OvA FDR metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--activations-dir", type=str, default="./activations")
    p.add_argument("--vectors-file", type=str, default="./results/domain_vectors.h5")
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args()

    compute_all_regularized(
        activations_dir=args.activations_dir,
        vectors_file=args.vectors_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
