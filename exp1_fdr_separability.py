#!/usr/bin/env python3
"""
exp1_fdr_separability.py — Experiment 6.2: Fisher Discriminant Ratio Separability

Computes the multiclass Fisher Discriminant Ratio (FDR) at every layer for:
  1. Full residual stream  x^ℓ
  2. Attention contributions  Δa^ℓ
  3. MLP contributions  Δm^ℓ

FDR^ℓ = tr(S_B^ℓ) / tr(S_W^ℓ)

where:
    tr(S_B) = Σ_k  n_k  ||d_k^ℓ||²       (between-class scatter — from domain concept vectors)
    tr(S_W) = Σ_k Σ_{x∈D_k} ||x - μ_k||²  (within-class scatter — streamed from activations)

Quantifies localisation via the coefficient of variation (CV = std/mean) of each
FDR curve across layers.  High CV → signal compressed into a few layers (localised).
Low CV → signal distributed uniformly.

Outputs:
  - Combined FDR plot (3 curves on one figure) for each pooling mode
  - Scatter decomposition plots (tr(S_B) and tr(S_W) separately)
  - Numerical results saved as .npz
  - JSON summary with CV values and peak layer info

Usage:
    python exp1_fdr_separability.py \\
        --activations-dir ./activations \\
        --vectors-file ./results/domain_vectors.h5 \\
        --output-dir ./results/exp1_fdr \\
        --batch-size 500
"""

import argparse
import json
import os
import time
from typing import Dict, Tuple

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (reuse pattern from calculate_separability.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_domain_data(vectors_file: str, activations_dir: str):
    """
    Load domain vectors, domain means, and sample counts from the
    pre-computed domain_vectors.h5 and raw activation files.

    Returns
    -------
    domain_vectors : {domain: {comp: {mode: ndarray(L, D)}}}
    domain_means   : {domain: {comp: {mode: ndarray(L, D)}}}
    sample_counts  : {domain: int}
    domains        : list of str
    """
    components = ["attn", "mlp", "layer"]
    modes = ["mean", "last"]

    with h5py.File(vectors_file, "r") as f:
        domains = list(f.attrs["domains"])
        sample_counts = {d: int(f.attrs[f"num_samples_{d}"]) for d in domains}

        # Load domain vectors  (d_k = μ_k - μ_global)
        domain_vectors = {}
        for d in domains:
            domain_vectors[d] = {}
            for comp in components:
                domain_vectors[d][comp] = {}
                for mode in modes:
                    key = f"domain_vectors/{d}/{comp}/{mode}"
                    domain_vectors[d][comp][mode] = f[key][:].astype(np.float32)

        # Load global means
        global_means = {}
        for comp in components:
            global_means[comp] = {}
            for mode in modes:
                key = f"global_mean/{comp}/{mode}"
                global_means[comp][mode] = f[key][:].astype(np.float32)

    # Reconstruct domain means: μ_k = d_k + μ_global
    domain_means = {}
    for d in domains:
        domain_means[d] = {}
        for comp in components:
            domain_means[d][comp] = {}
            for mode in modes:
                domain_means[d][comp][mode] = (
                    domain_vectors[d][comp][mode] + global_means[comp][mode]
                )

    return domain_vectors, domain_means, sample_counts, domains


# ─────────────────────────────────────────────────────────────────────────────
# Between-Class Scatter  tr(S_B)
# ─────────────────────────────────────────────────────────────────────────────

def compute_between_class_scatter(
    domain_vectors: Dict,
    sample_counts: Dict,
    component: str,
    mode: str,
) -> np.ndarray:
    """
    Compute tr(S_B) per layer.

    tr(S_B)^ℓ = Σ_k n_k ||d_k^ℓ||²

    This is literally the sum of squared norms of the domain concept vectors,
    weighted by sample counts.

    Returns
    -------
    tr_sb : np.ndarray, shape (num_layers,)
    """
    domains = list(domain_vectors.keys())
    dv0 = domain_vectors[domains[0]][component][mode]
    num_layers = dv0.shape[0]

    tr_sb = np.zeros(num_layers, dtype=np.float64)
    for d in domains:
        dv = domain_vectors[d][component][mode].astype(np.float64)  # (L, D)
        nk = sample_counts[d]
        tr_sb += nk * np.sum(dv ** 2, axis=1)  # per-layer squared norm

    return tr_sb


# ─────────────────────────────────────────────────────────────────────────────
# Within-Class Scatter  tr(S_W)  — streamed from activation files
# ─────────────────────────────────────────────────────────────────────────────

def compute_within_class_scatter(
    domain_means: Dict,
    activations_dir: str,
    component: str,
    mode: str,
    batch_size: int = 500,
) -> np.ndarray:
    """
    Compute tr(S_W) per layer by streaming through activation files.

    tr(S_W)^ℓ = Σ_k Σ_{x ∈ D_k} ||x^ℓ(x) - μ_k^ℓ||²

    Returns
    -------
    tr_sw : np.ndarray, shape (num_layers,)
    """
    domains = list(domain_means.keys())
    mu0 = domain_means[domains[0]][component][mode]
    num_layers = mu0.shape[0]

    tr_sw = np.zeros(num_layers, dtype=np.float64)

    for d in domains:
        mu_k = domain_means[d][component][mode].astype(np.float64)  # (L, D)
        act_path = os.path.join(activations_dir, f"{d}_activations.h5")

        with h5py.File(act_path, "r") as f:
            dataset = f[f"{component}/{mode}"]  # (N, L, D)
            N = dataset.shape[0]

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = dataset[start:end].astype(np.float64)  # (B, L, D)
                diff = batch - mu_k[np.newaxis, :, :]  # (B, L, D)
                # ||diff||² per sample per layer, then sum over batch & dim
                tr_sw += np.sum(diff ** 2, axis=(0, 2))  # (L,)

                if start % (batch_size * 10) == 0:
                    print(f"      {d}: {end}/{N}", end="\r", flush=True)

        print(f"      {d}: {N}/{N} ✓")

    return tr_sw


# ─────────────────────────────────────────────────────────────────────────────
# FDR Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_fdr(
    domain_vectors: Dict,
    domain_means: Dict,
    sample_counts: Dict,
    activations_dir: str,
    component: str,
    mode: str,
    batch_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FDR = tr(S_B) / tr(S_W) per layer for a given component and mode.

    Returns
    -------
    fdr   : np.ndarray, shape (num_layers,)
    tr_sb : np.ndarray, shape (num_layers,)
    tr_sw : np.ndarray, shape (num_layers,)
    """
    tr_sb = compute_between_class_scatter(
        domain_vectors, sample_counts, component, mode
    )
    tr_sw = compute_within_class_scatter(
        domain_means, activations_dir, component, mode, batch_size
    )

    # Avoid division by zero
    fdr = np.where(tr_sw > 0, tr_sb / tr_sw, 0.0)
    return fdr, tr_sb, tr_sw


# ─────────────────────────────────────────────────────────────────────────────
# Localisation: Coefficient of Variation
# ─────────────────────────────────────────────────────────────────────────────

def compute_cv(values: np.ndarray) -> float:
    """
    Coefficient of Variation = std / mean.

    High CV → signal is compressed into a few layers (localised).
    Low CV  → signal is distributed uniformly across layers.
    """
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    return float(np.std(values) / mean)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting: Combined FDR Curves
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "attn":  "#FF6B6B",   # coral red
    "mlp":   "#4ECDC4",   # teal
    "layer": "#6C5CE7",   # purple
}
LABELS = {
    "attn":  "Attention Δa",
    "mlp":   "MLP Δm",
    "layer": "Full Residual Stream",
}


def plot_fdr_combined(
    results: Dict,
    mode: str,
    cv_values: Dict,
    output_path: str,
    num_layers: int,
):
    """
    Plot all three FDR curves (attn, mlp, layer) on a single figure.

    Annotates with CV values and peak layers in the legend.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    layers = np.arange(num_layers)

    for comp in ["layer", "attn", "mlp"]:
        fdr = results[comp]["fdr"]
        peak_layer = int(np.argmax(fdr))
        cv = cv_values[comp]
        label = f"{LABELS[comp]}  (CV={cv:.3f}, peak=L{peak_layer})"
        ax.plot(
            layers, fdr,
            color=COLORS[comp],
            linewidth=2.2,
            label=label,
            marker="o",
            markersize=4,
            alpha=0.9,
        )
        # Mark peak with a vertical annotation
        ax.annotate(
            f"L{peak_layer}",
            xy=(peak_layer, fdr[peak_layer]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=COLORS[comp],
            arrowprops=dict(arrowstyle="-", color=COLORS[comp], alpha=0.5),
        )

    # Styling
    mode_title = "Mean-Pool" if mode == "mean" else "Last-Token"
    ax.set_title(
        f"Fisher Discriminant Ratio — {mode_title}\n"
        f"Separability of Domain Representations Across Layers",
        fontsize=15, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("FDR  =  tr(S_B) / tr(S_W)", fontsize=13)
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=11, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=11)

    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_scatter_decomposition(
    results: Dict,
    mode: str,
    output_path: str,
    num_layers: int,
):
    """
    Plot tr(S_B) and tr(S_W) separately for each component to show
    how between-class and within-class scatter evolve across layers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    layers = np.arange(num_layers)

    for ax, (metric_key, title, ylabel) in zip(axes, [
        ("tr_sb", "Between-Class Scatter  tr(S_B)", "tr(S_B)"),
        ("tr_sw", "Within-Class Scatter  tr(S_W)", "tr(S_W)"),
    ]):
        for comp in ["layer", "attn", "mlp"]:
            vals = results[comp][metric_key]
            ax.semilogy(
                layers, vals,
                color=COLORS[comp],
                linewidth=2.0,
                label=LABELS[comp],
                marker="o",
                markersize=3,
                alpha=0.9,
            )
        mode_title = "Mean-Pool" if mode == "mean" else "Last-Token"
        ax.set_title(f"{title} — {mode_title}",
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-0.5, num_layers - 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=10)
        ax.set_facecolor("#FAFAFA")

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_cv_bar_chart(
    cv_all: Dict,
    output_path: str,
):
    """
    Bar chart comparing CV values across components and modes.

    This is the localisation summary: high CV = localised, low CV = distributed.
    """
    modes = list(cv_all.keys())
    components = ["attn", "mlp", "layer"]
    n_modes = len(modes)
    n_comps = len(components)

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.22
    x = np.arange(n_comps)

    for i, mode in enumerate(modes):
        mode_label = "Mean-Pool" if mode == "mean" else "Last-Token"
        mode_color = "#2196F3" if mode == "mean" else "#FF9800"
        vals = [cv_all[mode][comp] for comp in components]
        bars = ax.bar(
            x + i * bar_width, vals,
            width=bar_width,
            label=mode_label,
            color=mode_color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
        )
        # Annotate bar values
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
            )

    ax.set_xticks(x + bar_width * (n_modes - 1) / 2)
    ax.set_xticklabels([LABELS[c] for c in components], fontsize=12)
    ax.set_ylabel("Coefficient of Variation (CV = std/mean)", fontsize=12)
    ax.set_title(
        "FDR Localisation — Coefficient of Variation\n"
        "Higher CV → more localised signal",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.tick_params(labelsize=11)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Computation
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    activations_dir: str,
    vectors_file: str,
    output_dir: str,
    batch_size: int = 500,
):
    """
    Full Experiment 6.2 pipeline:
      1. Load domain vectors and activation metadata
      2. Compute FDR for all 3 components × 2 modes
      3. Compute CV for localisation
      4. Generate plots and save results
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Experiment 6.2: Fisher Discriminant Ratio — Separability")
    print(f"{'='*60}")
    print(f"  Vectors file   : {vectors_file}")
    print(f"  Activations dir: {activations_dir}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Batch size     : {batch_size}")
    print()

    # ── Load data ────────────────────────────────────────────────────────
    t0 = time.time()
    domain_vectors, domain_means, sample_counts, domains = load_domain_data(
        vectors_file, activations_dir
    )
    print(f"  Domains: {domains}")
    print(f"  Sample counts: {sample_counts}")

    num_layers = domain_vectors[domains[0]]["attn"]["last"].shape[0]
    hidden_dim = domain_vectors[domains[0]]["attn"]["last"].shape[1]
    print(f"  Num layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print()

    components = ["attn", "mlp", "layer"]
    modes = ["last"]

    # ── Compute FDR for all components × modes ───────────────────────────
    all_results = {}   # {mode: {comp: {'fdr', 'tr_sb', 'tr_sw'}}}
    cv_all = {}        # {mode: {comp: float}}

    for mode in modes:
        print(f"  {'─'*50}")
        print(f"  Mode: {mode}")
        print(f"  {'─'*50}")
        all_results[mode] = {}
        cv_all[mode] = {}

        for comp in components:
            print(f"    Component: {comp}")
            fdr, tr_sb, tr_sw = compute_fdr(
                domain_vectors=domain_vectors,
                domain_means=domain_means,
                sample_counts=sample_counts,
                activations_dir=activations_dir,
                component=comp,
                mode=mode,
                batch_size=batch_size,
            )

            cv = compute_cv(fdr)

            all_results[mode][comp] = {
                "fdr": fdr,
                "tr_sb": tr_sb,
                "tr_sw": tr_sw,
            }
            cv_all[mode][comp] = cv

            peak = int(np.argmax(fdr))
            print(f"      FDR range : [{fdr.min():.6f}, {fdr.max():.6f}]")
            print(f"      Peak layer: L{peak}  (FDR={fdr[peak]:.6f})")
            print(f"      CV (std/mean): {cv:.4f}")
            print()

    elapsed = time.time() - t0
    print(f"  Total computation time: {elapsed:.1f}s\n")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"  {'='*60}")
    print(f"  LOCALISATION SUMMARY (Coefficient of Variation)")
    print(f"  {'='*60}")
    print(f"  {'Component':<25s} {'Last-Token CV':>14s}")
    print(f"  {'-'*40}")
    for comp in components:
        cv_last = cv_all["last"][comp]
        print(f"  {LABELS[comp]:<25s} {cv_last:>14.4f}")
    print()

    # Localisation claim check
    attn_cv = cv_all["last"]["attn"]
    mlp_cv = cv_all["last"]["mlp"]
    ratio = attn_cv / mlp_cv if mlp_cv > 0 else float("inf")
    verdict = "✓ LOCALISED" if attn_cv > mlp_cv else "✗ NOT LOCALISED"
    print(f"  [Last-Token] Attention CV / MLP CV = {ratio:.2f}  → {verdict}")
    print()

    # ── Save numerical results ───────────────────────────────────────────
    save_dict = {}
    for mode in modes:
        for comp in components:
            for key in ["fdr", "tr_sb", "tr_sw"]:
                save_dict[f"{mode}_{comp}_{key}"] = all_results[mode][comp][key]
            save_dict[f"{mode}_{comp}_cv"] = np.array([cv_all[mode][comp]])
    save_dict["domains"] = np.array(domains)
    save_dict["sample_counts"] = np.array([sample_counts[d] for d in domains])

    npz_path = os.path.join(output_dir, "exp1_fdr_scores.npz")
    np.savez(npz_path, **save_dict)
    print(f"  Scores saved: {npz_path}")

    # ── Save JSON summary ────────────────────────────────────────────────
    summary = {
        "experiment": "6.2 — Fisher Discriminant Ratio Separability",
        "domains": domains,
        "sample_counts": sample_counts,
        "num_layers": int(num_layers),
        "hidden_dim": int(hidden_dim),
        "computation_time_seconds": round(elapsed, 1),
    }

    for mode in modes:
        mode_key = mode
        summary[mode_key] = {}
        for comp in components:
            fdr = all_results[mode][comp]["fdr"]
            peak = int(np.argmax(fdr))
            summary[mode_key][comp] = {
                "cv": round(cv_all[mode][comp], 6),
                "fdr_mean": round(float(np.mean(fdr)), 6),
                "fdr_std": round(float(np.std(fdr)), 6),
                "fdr_min": round(float(np.min(fdr)), 6),
                "fdr_max": round(float(np.max(fdr)), 6),
                "peak_layer": peak,
                "peak_fdr": round(float(fdr[peak]), 6),
                "top5_layers": [int(x) for x in np.argsort(fdr)[-5:][::-1]],
            }

    # Add localisation comparison
    summary["localisation_analysis"] = {}
    for mode in modes:
        attn_cv = cv_all[mode]["attn"]
        mlp_cv = cv_all[mode]["mlp"]
        layer_cv = cv_all[mode]["layer"]
        summary["localisation_analysis"][mode] = {
            "attn_cv": round(attn_cv, 6),
            "mlp_cv": round(mlp_cv, 6),
            "layer_cv": round(layer_cv, 6),
            "attn_over_mlp_ratio": round(attn_cv / mlp_cv, 4) if mlp_cv > 0 else None,
            "attn_more_localised": bool(attn_cv > mlp_cv),
        }

    json_path = os.path.join(output_dir, "exp1_fdr_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {json_path}")

    # ── Generate plots ───────────────────────────────────────────────────
    print(f"\n  Generating plots...")

    for mode in modes:
        # 1. Combined FDR curves (the main §6.2 figure)
        plot_fdr_combined(
            all_results[mode], mode, cv_all[mode],
            os.path.join(output_dir, f"fdr_combined_{mode}.png"),
            num_layers,
        )

        # 2. Scatter decomposition (tr(S_B) and tr(S_W) separately)
        plot_scatter_decomposition(
            all_results[mode], mode,
            os.path.join(output_dir, f"scatter_decomposition_{mode}.png"),
            num_layers,
        )

    # 3. CV bar chart (localisation comparison across modes)
    plot_cv_bar_chart(
        cv_all,
        os.path.join(output_dir, "localisation_cv_comparison.png"),
    )

    print(f"\n{'='*60}")
    print(f"Experiment 6.2 complete!")
    print(f"  Output: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")

    return all_results, cv_all


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Experiment 6.2: Fisher Discriminant Ratio separability analysis. "
            "Computes FDR across layers for residual stream, attention contributions, "
            "and MLP contributions, with localisation quantified via CV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--activations-dir", type=str, default="./activations",
        help="Directory containing {domain}_activations.h5 files.",
    )
    p.add_argument(
        "--vectors-file", type=str, default="./results/domain_vectors.h5",
        help="Path to pre-computed domain_vectors.h5.",
    )
    p.add_argument(
        "--output-dir", type=str, default="./results/exp1_fdr",
        help="Directory for output plots and data files.",
    )
    p.add_argument(
        "--batch-size", type=int, default=500,
        help="Batch size for streaming through activation files.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    run_experiment(
        activations_dir=args.activations_dir,
        vectors_file=args.vectors_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
