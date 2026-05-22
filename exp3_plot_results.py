#!/usr/bin/env python3
"""
exp3_plot_results.py — Plotting for Experiment 3: Causal Activation Patching

Generates publication-quality figures from the outputs of Parts 1 & 2:

  1. Domain concept vector norms across layers (attn, mlp, layer)
  2. α calibration magnitudes per layer
  3. Propagation curves: shift score s^ℓ vs. layer for each patch point
  4. Shift-score heatmap (patch_layer × downstream_layer)
  5. Propagation decay / amplification analysis
  6. Combined summary figure

Usage
-----
    python exp3_plot_results.py \
        --vectors-dir ./results/exp3_causal \
        --patching-dir ./results/exp3_causal/patching \
        --output-dir ./results/exp3_causal/plots
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.color": "#c9d1d9",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 10,
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0e1117",
})

# Colour palette
COLOURS = {
    "cs": "#58a6ff",
    "q-bio": "#f778ba",
    "layer": "#79c0ff",
    "attn": "#d2a8ff",
    "mlp": "#7ee787",
    "shift": "#ffa657",
    "accent1": "#ff7b72",
    "accent2": "#d2a8ff",
    "accent3": "#79c0ff",
}

PATCH_CMAP = plt.cm.plasma


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Experiment 3 results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vectors-dir", type=str, default="./results/exp3_causal")
    p.add_argument("--patching-dir", type=str, default="./results/exp3_causal/patching")
    p.add_argument("--output-dir", type=str, default="./results/exp3_causal/plots")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Domain concept vector norms
# ─────────────────────────────────────────────────────────────────────────────
def plot_domain_vector_norms(vectors_dir: str, output_dir: str):
    """Plot ‖d^ℓ_k‖ for each domain × component across layers."""
    meta_path = os.path.join(vectors_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print("  [SKIP] metadata.json not found — skipping vector norm plot")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    domains = meta["domains"]
    norms_info = meta["domain_vector_norms"]
    components = ["layer", "attn", "mlp"]
    comp_labels = {"layer": "Residual Stream", "attn": "Attention (Δa)", "mlp": "MLP (Δm)"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle("Domain Concept Vector Norms ‖d$^{\\ell}_k$‖ Across Layers",
                 fontsize=16, fontweight="bold", y=1.02)

    for ax, comp in zip(axes, components):
        for domain in domains:
            norms = norms_info[domain][comp]["per_layer_norms"]
            layers = list(range(len(norms)))
            colour = COLOURS.get(domain, "#8b949e")
            ax.plot(layers, norms, "-o", color=colour, label=domain,
                    markersize=4, linewidth=2, alpha=0.9)
            ax.fill_between(layers, 0, norms, alpha=0.1, color=colour)

        ax.set_title(comp_labels[comp], fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("‖d$^{\\ell}_k$‖")
        ax.legend(framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(norms) - 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "01_domain_vector_norms.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Alpha calibration magnitudes
# ─────────────────────────────────────────────────────────────────────────────
def plot_alpha_magnitudes(vectors_dir: str, output_dir: str):
    """Plot α (projection magnitude) per layer for each domain."""
    meta_path = os.path.join(vectors_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return

    with open(meta_path) as f:
        meta = json.load(f)

    domains = meta["domains"]
    norms_info = meta["domain_vector_norms"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Calibration Magnitude α Per Layer\n"
                 "(typical projection of attention onto own concept direction)",
                 fontsize=14, fontweight="bold")

    for domain in domains:
        if "alpha" not in norms_info[domain]:
            continue
        alpha_vals = norms_info[domain]["alpha"]["per_layer"]
        layers = list(range(len(alpha_vals)))
        colour = COLOURS.get(domain, "#8b949e")
        ax.plot(layers, alpha_vals, "-s", color=colour, label=f"α ({domain})",
                markersize=5, linewidth=2, alpha=0.9)
        ax.fill_between(layers, 0, alpha_vals, alpha=0.08, color=colour)

    ax.set_xlabel("Layer")
    ax.set_ylabel("α (mean |projection|)")
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, layers[-1])

    plt.tight_layout()
    path = os.path.join(output_dir, "02_alpha_calibration.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Propagation curves (shift score per layer, one curve per patch point)
# ─────────────────────────────────────────────────────────────────────────────
def plot_propagation_curves(patching_dir: str, output_dir: str):
    """Plot s^ℓ vs layer for each patch point — the core propagation result."""
    results_path = os.path.join(patching_dir, "patching_results.json")
    if not os.path.exists(results_path):
        print("  [SKIP] patching_results.json not found")
        return

    with open(results_path) as f:
        res = json.load(f)

    per_patch = res["per_patch_layer"]
    patch_layers = sorted([int(k) for k in per_patch.keys()])
    num_layers = len(per_patch[str(patch_layers[0])]["shift_scores"])

    source = res["source_domain"]
    target = res["target_domain"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Propagation Analysis: Domain Shift Score s$^{{\\ell}}$\n"
                 f"Patching {source.upper()} → {target.upper()} attention signal",
                 fontsize=14, fontweight="bold")

    cmap_vals = np.linspace(0.15, 0.85, len(patch_layers))

    for i, pl in enumerate(patch_layers):
        scores = per_patch[str(pl)]["shift_scores"]
        layers = list(range(num_layers))
        colour = PATCH_CMAP(cmap_vals[i])

        # Only plot layers after the patch point
        post_layers = list(range(pl, num_layers))
        post_scores = [scores[l] for l in post_layers]

        ax.plot(post_layers, post_scores, "-o", color=colour,
                label=f"Patch @ L{pl}", markersize=4, linewidth=2, alpha=0.85)

        # Mark the patch point
        ax.axvline(x=pl, color=colour, linestyle="--", alpha=0.3, linewidth=1)

    ax.axhline(y=0, color="#484f58", linestyle="-", alpha=0.5, linewidth=1)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Domain Shift Score s$^{\\ell}$")
    ax.legend(loc="upper left", framealpha=0.7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_layers - 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "03_propagation_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Shift-score heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_shift_heatmap(patching_dir: str, output_dir: str):
    """Heatmap: patch_layer (y) × downstream_layer (x) → shift score."""
    matrix_path = os.path.join(patching_dir, "shift_scores_matrix.pt")
    results_path = os.path.join(patching_dir, "patching_results.json")
    if not os.path.exists(matrix_path):
        print("  [SKIP] shift_scores_matrix.pt not found")
        return

    matrix = torch.load(matrix_path, weights_only=True).numpy()

    with open(results_path) as f:
        res = json.load(f)

    patch_layers = res["shift_matrix_rows"]
    source = res["source_domain"]
    target = res["target_domain"]
    num_layers = matrix.shape[1]

    # Mask out layers before/at the patch point (they're not meaningful)
    masked = np.copy(matrix)
    for i, pl in enumerate(patch_layers):
        masked[i, :pl + 1] = np.nan

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Domain Shift Heatmap: {source.upper()} → {target.upper()}\n"
                 f"(s$^{{\\ell}}$ at downstream layer ℓ after patching at layer ℓ*)",
                 fontsize=14, fontweight="bold")

    # Use a diverging colormap centred at 0
    vmax = np.nanmax(np.abs(masked))
    im = ax.imshow(masked, aspect="auto", cmap="RdYlBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[0, num_layers, len(patch_layers) - 0.5, -0.5])

    ax.set_yticks(range(len(patch_layers)))
    ax.set_yticklabels([f"L{pl}" for pl in patch_layers])
    ax.set_xlabel("Downstream Layer ℓ")
    ax.set_ylabel("Patch Layer ℓ*")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Shift Score s$^{\\ell}$", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")

    plt.tight_layout()
    path = os.path.join(output_dir, "04_shift_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Propagation decay analysis
# ─────────────────────────────────────────────────────────────────────────────
def plot_propagation_decay(patching_dir: str, output_dir: str):
    """Plot how the shift score decays/amplifies as a function of distance from patch point."""
    results_path = os.path.join(patching_dir, "patching_results.json")
    if not os.path.exists(results_path):
        return

    with open(results_path) as f:
        res = json.load(f)

    per_patch = res["per_patch_layer"]
    patch_layers = sorted([int(k) for k in per_patch.keys()])
    num_layers = len(per_patch[str(patch_layers[0])]["shift_scores"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Propagation Dynamics Analysis", fontsize=14, fontweight="bold", y=1.02)

    cmap_vals = np.linspace(0.15, 0.85, len(patch_layers))

    # Left: shift vs distance from patch point
    for i, pl in enumerate(patch_layers):
        scores = per_patch[str(pl)]["shift_scores"]
        distances = list(range(1, num_layers - pl))
        shift_vals = [scores[pl + d] for d in distances]
        colour = PATCH_CMAP(cmap_vals[i])
        ax1.plot(distances, shift_vals, "-o", color=colour,
                 label=f"L{pl}", markersize=3, linewidth=1.8, alpha=0.85)

    ax1.axhline(y=0, color="#484f58", linestyle="-", alpha=0.5)
    ax1.set_xlabel("Distance from Patch Point (ℓ − ℓ*)")
    ax1.set_ylabel("Shift Score s$^{\\ell}$")
    ax1.set_title("Shift vs. Distance from Patch", fontweight="bold")
    ax1.legend(framealpha=0.7, ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: cumulative shift (integral of propagation)
    for i, pl in enumerate(patch_layers):
        scores = per_patch[str(pl)]["shift_scores"]
        distances = list(range(1, num_layers - pl))
        shift_vals = [scores[pl + d] for d in distances]
        cumulative = np.cumsum(shift_vals)
        colour = PATCH_CMAP(cmap_vals[i])
        ax2.plot(distances, cumulative, "-s", color=colour,
                 label=f"L{pl}", markersize=3, linewidth=1.8, alpha=0.85)

    ax2.axhline(y=0, color="#484f58", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Distance from Patch Point (ℓ − ℓ*)")
    ax2.set_ylabel("Cumulative Shift Score")
    ax2.set_title("Cumulative Propagation", fontweight="bold")
    ax2.legend(framealpha=0.7, ncol=2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "05_propagation_decay.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary bar chart — mean post-patch shift per patch layer
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_bars(patching_dir: str, output_dir: str):
    """Bar chart of mean post-patch shift score for each patch layer."""
    results_path = os.path.join(patching_dir, "patching_results.json")
    if not os.path.exists(results_path):
        return

    with open(results_path) as f:
        res = json.load(f)

    per_patch = res["per_patch_layer"]
    patch_layers = sorted([int(k) for k in per_patch.keys()])
    source = res["source_domain"]
    target = res["target_domain"]

    mean_shifts = [per_patch[str(pl)]["post_patch_mean_shift"] for pl in patch_layers]
    max_shifts = [per_patch[str(pl)]["post_patch_max_shift"] for pl in patch_layers]
    max_layers = [per_patch[str(pl)]["post_patch_max_layer"] for pl in patch_layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Patch Effectiveness Summary: {source.upper()} → {target.upper()}",
                 fontsize=14, fontweight="bold", y=1.02)

    x = np.arange(len(patch_layers))
    width = 0.6

    # Mean shift
    bars1 = ax1.bar(x, mean_shifts, width, color=COLOURS["shift"], alpha=0.85,
                    edgecolor="#30363d", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{pl}" for pl in patch_layers])
    ax1.set_xlabel("Patch Layer ℓ*")
    ax1.set_ylabel("Mean Post-Patch Shift Score")
    ax1.set_title("Mean Downstream Shift", fontweight="bold")
    ax1.axhline(y=0, color="#484f58", linestyle="-", alpha=0.5)
    ax1.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars1, mean_shifts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                 color="#c9d1d9")

    # Max shift with layer annotation
    bars2 = ax2.bar(x, max_shifts, width, color=COLOURS["accent2"], alpha=0.85,
                    edgecolor="#30363d", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{pl}" for pl in patch_layers])
    ax2.set_xlabel("Patch Layer ℓ*")
    ax2.set_ylabel("Max Post-Patch Shift Score")
    ax2.set_title("Peak Downstream Shift", fontweight="bold")
    ax2.axhline(y=0, color="#484f58", linestyle="-", alpha=0.5)
    ax2.grid(True, axis="y", alpha=0.3)

    for bar, val, ml in zip(bars2, max_shifts, max_layers):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.4f}\n@L{ml}", ha="center", va="bottom", fontsize=8,
                 color="#c9d1d9")

    plt.tight_layout()
    path = os.path.join(output_dir, "06_summary_bars.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Combined overview figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_combined_overview(vectors_dir: str, patching_dir: str, output_dir: str):
    """4-panel overview figure for paper / presentation."""
    meta_path = os.path.join(vectors_dir, "metadata.json")
    results_path = os.path.join(patching_dir, "patching_results.json")
    matrix_path = os.path.join(patching_dir, "shift_scores_matrix.pt")

    if not all(os.path.exists(p) for p in [meta_path, results_path, matrix_path]):
        print("  [SKIP] Not all data available for combined overview")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    with open(results_path) as f:
        res = json.load(f)

    matrix = torch.load(matrix_path, weights_only=True).numpy()
    patch_layers = res["shift_matrix_rows"]
    source = res["source_domain"]
    target = res["target_domain"]
    per_patch = res["per_patch_layer"]
    num_layers = matrix.shape[1]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Experiment 3: Causal Activation Patching — {source.upper()} → {target.upper()}",
                 fontsize=18, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Concept vector norms (attn)
    ax_a = fig.add_subplot(gs[0, 0])
    for domain in meta["domains"]:
        norms = meta["domain_vector_norms"][domain]["attn"]["per_layer_norms"]
        colour = COLOURS.get(domain, "#8b949e")
        ax_a.plot(range(len(norms)), norms, "-o", color=colour, label=domain,
                  markersize=3, linewidth=2)
        ax_a.fill_between(range(len(norms)), 0, norms, alpha=0.1, color=colour)
    ax_a.set_title("(A) Attention Concept Vector Norms", fontweight="bold")
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("‖d$^{attn,\\ell}_k$‖")
    ax_a.legend(framealpha=0.7)
    ax_a.grid(True, alpha=0.3)

    # Panel B: Propagation curves
    ax_b = fig.add_subplot(gs[0, 1])
    cmap_vals = np.linspace(0.15, 0.85, len(patch_layers))
    for i, pl in enumerate(patch_layers):
        scores = per_patch[str(pl)]["shift_scores"]
        post_layers = list(range(pl, num_layers))
        post_scores = [scores[l] for l in post_layers]
        colour = PATCH_CMAP(cmap_vals[i])
        ax_b.plot(post_layers, post_scores, "-o", color=colour,
                  label=f"ℓ*={pl}", markersize=3, linewidth=1.8, alpha=0.85)
    ax_b.axhline(y=0, color="#484f58", alpha=0.5)
    ax_b.set_title("(B) Propagation Curves", fontweight="bold")
    ax_b.set_xlabel("Layer ℓ")
    ax_b.set_ylabel("s$^{\\ell}$")
    ax_b.legend(framealpha=0.7, ncol=2, fontsize=9)
    ax_b.grid(True, alpha=0.3)

    # Panel C: Heatmap
    ax_c = fig.add_subplot(gs[1, 0])
    masked = np.copy(matrix)
    for i, pl in enumerate(patch_layers):
        masked[i, :pl + 1] = np.nan
    vmax = np.nanmax(np.abs(masked))
    im = ax_c.imshow(masked, aspect="auto", cmap="RdYlBu_r",
                     vmin=-vmax, vmax=vmax,
                     extent=[0, num_layers, len(patch_layers) - 0.5, -0.5])
    ax_c.set_yticks(range(len(patch_layers)))
    ax_c.set_yticklabels([f"L{pl}" for pl in patch_layers])
    ax_c.set_xlabel("Downstream Layer ℓ")
    ax_c.set_ylabel("Patch Layer ℓ*")
    ax_c.set_title("(C) Shift Score Heatmap", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.8)
    cbar.set_label("s$^{\\ell}$", color="#c9d1d9")

    # Panel D: Summary bars
    ax_d = fig.add_subplot(gs[1, 1])
    mean_shifts = [per_patch[str(pl)]["post_patch_mean_shift"] for pl in patch_layers]
    x = np.arange(len(patch_layers))
    ax_d.bar(x, mean_shifts, 0.6, color=COLOURS["shift"], alpha=0.85,
             edgecolor="#30363d")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"L{pl}" for pl in patch_layers])
    ax_d.set_xlabel("Patch Layer ℓ*")
    ax_d.set_ylabel("Mean Post-Patch Shift")
    ax_d.set_title("(D) Patch Effectiveness", fontweight="bold")
    ax_d.axhline(y=0, color="#484f58", alpha=0.5)
    ax_d.grid(True, axis="y", alpha=0.3)
    for xi, val in zip(x, mean_shifts):
        ax_d.text(xi, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                  color="#c9d1d9")

    path = os.path.join(output_dir, "07_combined_overview.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Experiment 3 — Generating Plots")
    print(f"  Vectors dir  : {args.vectors_dir}")
    print(f"  Patching dir : {args.patching_dir}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"{'='*60}\n")

    # Part 1 plots (domain vectors)
    print("── Part 1 Plots ──")
    plot_domain_vector_norms(args.vectors_dir, args.output_dir)
    plot_alpha_magnitudes(args.vectors_dir, args.output_dir)

    # Part 2 plots (patching results)
    print("\n── Part 2 Plots ──")
    plot_propagation_curves(args.patching_dir, args.output_dir)
    plot_shift_heatmap(args.patching_dir, args.output_dir)
    plot_propagation_decay(args.patching_dir, args.output_dir)
    plot_summary_bars(args.patching_dir, args.output_dir)

    # Combined
    print("\n── Combined ──")
    plot_combined_overview(args.vectors_dir, args.patching_dir, args.output_dir)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
