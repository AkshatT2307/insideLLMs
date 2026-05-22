#!/usr/bin/env python3
"""
generate_all_plots.py — Generate all plots for the fine-tuning experiments.

Generates three categories of plots:

1. Fine-tuning: Train & val perplexity across epochs (all domains).
2. Weight Change Norms: δ_attn vs δ_mlp across layers.
   - 3 epochs × 6 domains × 2 types (grouped / detailed) = 36 plots.
3. Domain Specificity: 6×6 heatmaps of cosine similarity.
   - 3 epochs × (7 modules + 2 grouped) = 27 heatmaps.

Usage:
    python generate_all_plots.py
"""

import os
import sys
import json
import math
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # FineTuning/
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

DOMAINS = ["cs", "eess", "math", "physics", "q-bio", "stat"]
EPOCHS = [1, 2, 3]
NUM_LAYERS = 28

ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
ALL_MODULES = ATTN_MODULES + MLP_MODULES

# ─────────────────────────────────────────────────────────────────────────────
# Style configuration
# ─────────────────────────────────────────────────────────────────────────────

# Publication-quality settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palettes
DOMAIN_COLORS = {
    "cs":      "#2196F3",  # blue
    "eess":    "#FF9800",  # orange
    "math":    "#4CAF50",  # green
    "physics": "#9C27B0",  # purple
    "q-bio":   "#F44336",  # red
    "stat":    "#00BCD4",  # teal
}

DOMAIN_LABELS = {
    "cs": "CS", "eess": "EESS", "math": "Math",
    "physics": "Physics", "q-bio": "Q-Bio", "stat": "Stat",
}

# Module colors for detailed plots
MODULE_COLORS = {
    "q_proj":    "#1565C0",  # dark blue
    "k_proj":    "#42A5F5",  # light blue
    "v_proj":    "#0D47A1",  # navy
    "o_proj":    "#64B5F6",  # sky blue
    "gate_proj": "#E65100",  # dark orange
    "up_proj":   "#FF9800",  # orange
    "down_proj": "#FFB74D",  # light orange
}

MODULE_LINESTYLES = {
    "q_proj": "-", "k_proj": "--", "v_proj": "-.", "o_proj": ":",
    "gate_proj": "-", "up_proj": "--", "down_proj": "-.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_training_logs():
    """Load all domain training logs."""
    logs = {}
    for domain in DOMAINS:
        path = os.path.join(LOGS_DIR, f"{domain}_training_log.json")
        if os.path.exists(path):
            with open(path) as f:
                logs[domain] = json.load(f)
    return logs


def load_baseline():
    """Load baseline validation results."""
    path = os.path.join(LOGS_DIR, "baseline_val_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_weight_norms(epoch):
    """Load weight change norms result for a given epoch."""
    path = os.path.join(RESULTS_DIR, f"weight_change_norms_epoch{epoch}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_domain_specificity(epoch):
    """Load domain specificity result for a given epoch."""
    path = os.path.join(RESULTS_DIR, f"domain_specificity_epoch{epoch}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 1: Fine-tuning — Train & Val Perplexity
# ═════════════════════════════════════════════════════════════════════════════

def plot_finetuning(training_logs, baseline):
    """Plot train & val perplexity across epochs for all domains."""
    out_dir = ensure_dir(os.path.join(PLOTS_DIR, "1_finetuning"))

    # ── (a) All domains in one figure ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for domain in DOMAINS:
        if domain not in training_logs:
            continue
        log = training_logs[domain]
        epoch_logs = log.get("epoch_logs", [])
        if not epoch_logs:
            continue

        epochs = [e["epoch"] for e in epoch_logs]
        train_ppl = [e["train_perplexity"] for e in epoch_logs]
        val_ppl = [e["val_perplexity"] for e in epoch_logs]
        color = DOMAIN_COLORS[domain]
        label = DOMAIN_LABELS[domain]

        # Add baseline as epoch 0
        if domain in baseline:
            bl_ppl = baseline[domain]["perplexity"]
            epochs_with_bl = [0] + epochs
            train_ppl_with_bl = [bl_ppl] + train_ppl
            val_ppl_with_bl = [bl_ppl] + val_ppl
        else:
            epochs_with_bl = epochs
            train_ppl_with_bl = train_ppl
            val_ppl_with_bl = val_ppl

        axes[0].plot(epochs_with_bl, train_ppl_with_bl, "o-", color=color,
                     label=label, linewidth=2, markersize=6)
        axes[1].plot(epochs_with_bl, val_ppl_with_bl, "s--", color=color,
                     label=label, linewidth=2, markersize=6)

    for ax, title in zip(axes, ["Train Perplexity", "Validation Perplexity"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.set_title(title)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["Base", "1", "2", "3"])
        ax.legend(loc="best", framealpha=0.9)

    fig.suptitle("LoRA Fine-Tuning: Perplexity Across Epochs", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "train_val_perplexity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── (b) Per-domain individual plots ───────────────────────────────────
    for domain in DOMAINS:
        if domain not in training_logs:
            continue
        log = training_logs[domain]
        epoch_logs = log.get("epoch_logs", [])
        if not epoch_logs:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        epochs = [e["epoch"] for e in epoch_logs]
        train_ppl = [e["train_perplexity"] for e in epoch_logs]
        val_ppl = [e["val_perplexity"] for e in epoch_logs]
        color = DOMAIN_COLORS[domain]

        if domain in baseline:
            bl_ppl = baseline[domain]["perplexity"]
            epochs = [0] + epochs
            train_ppl = [bl_ppl] + train_ppl
            val_ppl = [bl_ppl] + val_ppl

        ax.plot(epochs, train_ppl, "o-", color=color, label="Train", linewidth=2, markersize=7)
        ax.plot(epochs, val_ppl, "s--", color=color, label="Val", linewidth=2, markersize=7, alpha=0.7)
        ax.axhline(y=bl_ppl, color="gray", linestyle=":", alpha=0.5, label=f"Baseline ({bl_ppl:.2f})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.set_title(f"{DOMAIN_LABELS[domain]} — Train vs Val Perplexity")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["Base", "1", "2", "3"])
        ax.legend(loc="best")
        fig.tight_layout()

        path = os.path.join(out_dir, f"{domain}_perplexity.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── (c) Step-level training loss curves ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for domain in DOMAINS:
        if domain not in training_logs:
            continue
        log = training_logs[domain]
        step_logs = log.get("step_logs", [])
        if not step_logs:
            continue
        steps = [s["step"] for s in step_logs]
        losses = [s["loss"] for s in step_logs]
        color = DOMAIN_COLORS[domain]
        ax.plot(steps, losses, color=color, label=DOMAIN_LABELS[domain],
                linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Training Loss (windowed)")
    ax.set_title("Step-Level Training Loss Across All Domains")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(out_dir, "step_loss_all_domains.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 2: Weight Change Norms
# ═════════════════════════════════════════════════════════════════════════════

def plot_weight_change_norms():
    """
    Plot relative weight change norms across layers.
    Two types per (domain, epoch):
      - Grouped: mean attn vs mean mlp (2 lines)
      - Detailed: all 7 modules individually
    """
    for epoch in EPOCHS:
        data = load_weight_norms(epoch)
        if data is None:
            print(f"  [SKIP] weight_change_norms_epoch{epoch}.json not found")
            continue

        epoch_dir = ensure_dir(os.path.join(PLOTS_DIR, "2_weight_change_norms", f"epoch_{epoch}"))

        for domain in DOMAINS:
            if domain not in data.get("results", {}):
                continue
            result = data["results"][domain]
            grouped = result["grouped"]
            per_module = result["per_module"]

            layers = list(range(NUM_LAYERS))

            # ── Type 1: Grouped (mean attn vs mean mlp) ──────────────────
            fig, ax = plt.subplots(figsize=(10, 4.5))

            attn_norms = [grouped[str(l)]["attn"]["combined_relative_norm"] for l in layers]
            mlp_norms = [grouped[str(l)]["mlp"]["combined_relative_norm"] for l in layers]

            ax.plot(layers, attn_norms, "o-", color="#1976D2", label="Attention (combined)",
                    linewidth=2, markersize=4)
            ax.plot(layers, mlp_norms, "s-", color="#E65100", label="MLP (combined)",
                    linewidth=2, markersize=4)

            ax.set_xlabel("Layer")
            ax.set_ylabel("δ = ||ΔW||_F / ||W_base||_F")
            ax.set_title(f"{DOMAIN_LABELS[domain]} — Weight Change Norms (Epoch {epoch})")
            ax.legend(loc="best")
            ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            fig.tight_layout()

            path = os.path.join(epoch_dir, f"{domain}_grouped.png")
            fig.savefig(path)
            plt.close(fig)
            print(f"  Saved: {path}")

            # ── Type 2: Detailed (all 7 modules) ─────────────────────────
            fig, ax = plt.subplots(figsize=(12, 5))

            for module in ALL_MODULES:
                norms = []
                for l in layers:
                    layer_data = per_module.get(str(l), {})
                    if module in layer_data:
                        norms.append(layer_data[module]["relative_norm"])
                    else:
                        norms.append(0)

                color = MODULE_COLORS[module]
                ls = MODULE_LINESTYLES[module]
                group = "attn" if module in ATTN_MODULES else "mlp"
                lw = 1.8 if module in ATTN_MODULES else 2.2

                ax.plot(layers, norms, color=color, linestyle=ls,
                        label=f"{module} ({group})", linewidth=lw, alpha=0.85)

            ax.set_xlabel("Layer")
            ax.set_ylabel("δ = ||ΔW||_F / ||W_base||_F")
            ax.set_title(f"{DOMAIN_LABELS[domain]} — Per-Module Weight Change Norms (Epoch {epoch})")
            ax.legend(loc="best", ncol=2, fontsize=8, framealpha=0.9)
            ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            fig.tight_layout()

            path = os.path.join(epoch_dir, f"{domain}_detailed.png")
            fig.savefig(path)
            plt.close(fig)
            print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 3: Domain Specificity Heatmaps
# ═════════════════════════════════════════════════════════════════════════════

def average_similarity_matrix_across_layers(per_layer_data, key_name, domains):
    """
    Average the 6×6 similarity matrix across all layers.
    
    per_layer_data: dict of layer_idx (str) → { key_name: { 'matrix': [...], 'domains': [...] } }
    Returns: NxN numpy array
    """
    n = len(domains)
    matrices = []
    for layer_idx in range(NUM_LAYERS):
        layer_str = str(layer_idx)
        if layer_str in per_layer_data and key_name in per_layer_data[layer_str]:
            m = per_layer_data[layer_str][key_name]["matrix"]
            if m and len(m) == n:
                arr = np.array(m, dtype=float)
                # Replace None with NaN
                arr = np.where(arr == None, np.nan, arr)
                matrices.append(arr)
    if not matrices:
        return None
    return np.nanmean(matrices, axis=0)


def get_layerwise_avg_similarity(per_layer_data, key_name, domains, off_diagonal_only=False):
    """Calculate the average similarity for each layer."""
    n = len(domains)
    avgs = []
    layers = list(range(NUM_LAYERS))
    for layer_idx in layers:
        layer_str = str(layer_idx)
        if layer_str in per_layer_data and key_name in per_layer_data[layer_str]:
            m = per_layer_data[layer_str][key_name]["matrix"]
            if m and len(m) == n:
                arr = np.array(m, dtype=float)
                arr = np.where(arr == None, np.nan, arr)
                if off_diagonal_only:
                    val = np.nanmean(arr[~np.eye(n, dtype=bool)])
                else:
                    val = np.nanmean(arr)
                avgs.append(val)
            else:
                avgs.append(np.nan)
        else:
            avgs.append(np.nan)
    return layers, avgs


def plot_heatmap(matrix, domains, title, path, vmin=None, vmax=None):
    """Plot a single 6×6 heatmap."""
    n = len(domains)
    labels = [DOMAIN_LABELS.get(d, d) for d in domains]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    if vmin is None:
        vmin = np.nanmin(matrix[~np.eye(n, dtype=bool)])
    if vmax is None:
        vmax = np.nanmax(matrix)

    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal")

    # Annotations
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > (vmin + vmax) / 2 * 1.3 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def find_most_distinguishing_layer(per_layer_grouped, domains):
    """
    Find the layer with the largest gap between attention and MLP
    off-diagonal similarity (attn_offdiag − mlp_offdiag).

    Returns (best_layer_idx, gap_value, attn_offdiag, mlp_offdiag).
    """
    n = len(domains)
    mask = ~np.eye(n, dtype=bool)
    best_layer, best_gap = 0, -np.inf
    best_attn, best_mlp = 0.0, 0.0

    for layer_idx in range(NUM_LAYERS):
        layer_str = str(layer_idx)
        entry = per_layer_grouped.get(layer_str, {})
        if "attn" not in entry or "mlp" not in entry:
            continue
        attn_m = np.array(entry["attn"]["matrix"], dtype=float)
        mlp_m = np.array(entry["mlp"]["matrix"], dtype=float)
        attn_off = np.nanmean(attn_m[mask])
        mlp_off = np.nanmean(mlp_m[mask])
        gap = attn_off - mlp_off
        if gap > best_gap:
            best_gap = gap
            best_layer = layer_idx
            best_attn = attn_off
            best_mlp = mlp_off

    return best_layer, best_gap, best_attn, best_mlp


def plot_modulewise_at_layer(per_layer_module, per_layer_grouped, domains, layer_idx, gap, epoch, out_dir):
    """
    Plot a 3×3 grid of 6×6 heatmaps at a single layer:
      Row 0: q_proj, k_proj, v_proj
      Row 1: o_proj, Attn (pooled), MLP (pooled)
      Row 2: gate_proj, up_proj, down_proj
    """
    n = len(domains)
    labels = [DOMAIN_LABELS.get(d, d) for d in domains]
    layer_str = str(layer_idx)

    # Collect all 9 matrices
    subplot_specs = [
        # (row, col, key_source, key_name, display_title)
        (0, 0, "module", "q_proj",    "q_proj (Attn)"),
        (0, 1, "module", "k_proj",    "k_proj (Attn)"),
        (0, 2, "module", "v_proj",    "v_proj (Attn)"),
        (1, 0, "module", "o_proj",    "o_proj (Attn)"),
        (1, 1, "grouped", "attn",     "Attention (pooled)"),
        (1, 2, "grouped", "mlp",      "MLP (pooled)"),
        (2, 0, "module", "gate_proj", "gate_proj (MLP)"),
        (2, 1, "module", "up_proj",   "up_proj (MLP)"),
        (2, 2, "module", "down_proj", "down_proj (MLP)"),
    ]

    matrices = {}
    global_min, global_max = np.inf, -np.inf
    for row, col, src, key, title in subplot_specs:
        source = per_layer_module if src == "module" else per_layer_grouped
        entry = source.get(layer_str, {})
        if key in entry:
            arr = np.array(entry[key]["matrix"], dtype=float)
            arr = np.where(arr is None, np.nan, arr)
            matrices[(row, col)] = arr
            off_vals = arr[~np.eye(n, dtype=bool)]
            gmin = np.nanmin(off_vals)
            gmax = np.nanmax(arr)
            if gmin < global_min:
                global_min = gmin
            if gmax > global_max:
                global_max = gmax

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    for row, col, src, key, title in subplot_specs:
        ax = axes[row, col]
        if (row, col) not in matrices:
            ax.set_visible(False)
            continue

        arr = matrices[(row, col)]
        im = ax.imshow(arr, cmap="RdYlBu_r", vmin=global_min, vmax=global_max, aspect="equal")

        # Annotations
        for i in range(n):
            for j in range(n):
                val = arr[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > (global_min + global_max) / 2 * 1.3 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=text_color)

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Module-Wise Similarity at Most Distinguishing Layer {layer_idx} "
        f"(Attn−MLP gap = {gap:.4f}, Epoch {epoch})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    # Shared colorbar
    cbar_ax = fig.add_axes([1.02, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")

    path = os.path.join(out_dir, "modulewise_most_distinguishing_layer.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_domain_mixing_scores(per_layer_data, key_name, domains):
    """
    Compute three domain mixing scalar metrics per layer:
      1. MSE  = mean(S_offdiag²)     — penalises large cross-domain similarity quadratically
      2. MAE  = mean(|S_offdiag|)    — linear penalty
      3. Mean = mean(S_offdiag)       — simple average (same as existing off-diag mean)

    All range [0, 1].  0 = perfectly domain-specialised,  1 = fully mixed.

    Returns dict with keys 'mse', 'mae', 'mean', each a list of length NUM_LAYERS.
    """
    n = len(domains)
    mask = ~np.eye(n, dtype=bool)
    mse_scores, mae_scores, mean_scores = [], [], []

    for layer_idx in range(NUM_LAYERS):
        layer_str = str(layer_idx)
        entry = per_layer_data.get(layer_str, {})
        if key_name in entry:
            arr = np.array(entry[key_name]["matrix"], dtype=float)
            arr = np.where(arr is None, np.nan, arr)
            off = arr[mask]
            mse_scores.append(float(np.nanmean(off ** 2)))
            mae_scores.append(float(np.nanmean(np.abs(off))))
            mean_scores.append(float(np.nanmean(off)))
        else:
            mse_scores.append(np.nan)
            mae_scores.append(np.nan)
            mean_scores.append(np.nan)

    return {"mse": mse_scores, "mae": mae_scores, "mean": mean_scores}


def plot_domain_mixing_layerwise(per_layer_grouped, domains, epoch, out_dir):
    """
    Line plot of domain mixing scores (MSE, MAE, Mean) across layers,
    comparing Attention vs MLP.
    Produces one plot per metric variant.
    """
    attn_scores = compute_domain_mixing_scores(per_layer_grouped, "attn", domains)
    mlp_scores = compute_domain_mixing_scores(per_layer_grouped, "mlp", domains)
    layers = list(range(NUM_LAYERS))

    metric_info = {
        "mse":  ("MSE", "mean(S²_offdiag)"),
        "mae":  ("MAE", "mean(|S_offdiag|)"),
        "mean": ("Mean", "mean(S_offdiag)"),
    }

    for metric_key, (metric_label, formula) in metric_info.items():
        fig, ax = plt.subplots(figsize=(10, 4.5))

        ax.plot(layers, attn_scores[metric_key], "o-", color="#1976D2",
                label=f"Attention", linewidth=2, markersize=4)
        ax.plot(layers, mlp_scores[metric_key], "s-", color="#E65100",
                label=f"MLP", linewidth=2, markersize=4)

        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Domain Mixing Score ({formula})")
        ax.set_title(f"Domain Mixing — {metric_label} across Layers (Epoch {epoch})")
        ax.legend(loc="best")
        ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        fig.tight_layout()

        path = os.path.join(out_dir, f"domain_mixing_layerwise_{metric_key}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")

        # Print scalar summary
        attn_grand = np.nanmean(attn_scores[metric_key])
        mlp_grand = np.nanmean(mlp_scores[metric_key])
        print(f"    {metric_label} grand mean — Attn: {attn_grand:.6f}  MLP: {mlp_grand:.6f}")


def plot_domain_mixing_summary(per_layer_grouped, per_layer_module, domains, epoch, out_dir):
    """
    Bar chart of grand-mean domain mixing scores per module + pooled groups.
    One subplot per metric variant (MSE, MAE, Mean).
    """
    # Compute per-module scores
    module_scores = {}  # module → {mse: grand_mean, mae: grand_mean, mean: grand_mean}
    for module in ALL_MODULES:
        scores = compute_domain_mixing_scores(per_layer_module, module, domains)
        module_scores[module] = {
            k: float(np.nanmean(v)) for k, v in scores.items()
        }

    # Compute grouped scores
    for group in ["attn", "mlp"]:
        scores = compute_domain_mixing_scores(per_layer_grouped, group, domains)
        module_scores[f"{group}_pooled"] = {
            k: float(np.nanmean(v)) for k, v in scores.items()
        }

    # Bar chart labels and colors
    bar_keys = ALL_MODULES + ["attn_pooled", "mlp_pooled"]
    bar_labels = [m for m in ALL_MODULES] + ["Attn (pooled)", "MLP (pooled)"]
    bar_colors = []
    for k in bar_keys:
        if k in ATTN_MODULES or k == "attn_pooled":
            bar_colors.append("#1976D2")
        else:
            bar_colors.append("#E65100")

    metric_info = {
        "mse":  ("MSE", "mean(S²_offdiag)"),
        "mae":  ("MAE", "mean(|S_offdiag|)"),
        "mean": ("Mean", "mean(S_offdiag)"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    for ax, (metric_key, (metric_label, formula)) in zip(axes, metric_info.items()):
        values = [module_scores[k][metric_key] for k in bar_keys]
        x = np.arange(len(bar_keys))
        bars = ax.bar(x, values, color=bar_colors, edgecolor="white", linewidth=0.5, alpha=0.85)

        # Value annotations
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(f"Mixing Score ({formula})")
        ax.set_title(f"{metric_label} — Domain Mixing per Module", fontsize=11, fontweight="bold")

        # Add a separator line before pooled bars
        ax.axvline(x=len(ALL_MODULES) - 0.5, color="gray", linestyle="--", alpha=0.4)

    fig.suptitle(f"Domain Mixing Summary — Grand Mean Across Layers (Epoch {epoch})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "domain_mixing_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_domain_specificity():
    """
    Plot 6×6 cross-domain cosine similarity heatmaps.
    For each epoch:
      - 7 per-module heatmaps (q/k/v/o_proj, gate/up/down_proj)
      - 2 grouped heatmaps (avg attn, avg mlp)
      - 1 module-wise subplot at most distinguishing layer
      - 3 domain mixing layerwise plots (MSE, MAE, Mean)
      - 1 domain mixing summary bar chart
    """
    for epoch in EPOCHS:
        data = load_domain_specificity(epoch)
        if data is None:
            print(f"  [SKIP] domain_specificity_epoch{epoch}.json not found")
            continue

        epoch_dir = ensure_dir(os.path.join(PLOTS_DIR, "3_domain_specificity", f"epoch_{epoch}"))
        domains = data["config"]["domains"]
        per_layer_grouped = data["per_layer_grouped"]
        per_layer_module = data["per_layer_per_module"]

        # ── Per-layer average similarity (attn vs mlp) ───────────────────
        fig, ax = plt.subplots(figsize=(10, 4.5))
        
        layers, attn_avgs = get_layerwise_avg_similarity(per_layer_grouped, "attn", domains, off_diagonal_only=False)
        _, mlp_avgs = get_layerwise_avg_similarity(per_layer_grouped, "mlp", domains, off_diagonal_only=False)
        
        ax.plot(layers, attn_avgs, "o-", color="#1976D2", label="Attention (Whole Heatmap Avg)", linewidth=2, markersize=4)
        ax.plot(layers, mlp_avgs, "s-", color="#E65100", label="MLP (Whole Heatmap Avg)", linewidth=2, markersize=4)
        
        ax.set_xlabel("Layer")
        ax.set_ylabel("Average Similarity")
        ax.set_title(f"Average Whole Heatmap Similarity across Layers (Epoch {epoch})")
        ax.legend(loc="best")
        ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        fig.tight_layout()
        
        path = os.path.join(epoch_dir, "layerwise_avg_similarity.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")

        # ── Module-wise heatmap at most distinguishing layer ──────────────
        best_layer, gap, best_attn, best_mlp = find_most_distinguishing_layer(
            per_layer_grouped, domains
        )
        print(f"  Most distinguishing layer (Epoch {epoch}): Layer {best_layer} "
              f"(Attn={best_attn:.4f}, MLP={best_mlp:.4f}, gap={gap:.4f})")
        plot_modulewise_at_layer(
            per_layer_module, per_layer_grouped, domains,
            best_layer, gap, epoch, epoch_dir,
        )

        # ── Domain mixing scores (layerwise + summary) ────────────────────
        print(f"  Domain mixing scores (Epoch {epoch}):")
        plot_domain_mixing_layerwise(per_layer_grouped, domains, epoch, epoch_dir)
        plot_domain_mixing_summary(
            per_layer_grouped, per_layer_module, domains, epoch, epoch_dir,
        )

        # ── Grouped heatmaps (avg attn, avg mlp) ─────────────────────────
        for group_name in ["attn", "mlp"]:
            avg_matrix = average_similarity_matrix_across_layers(
                per_layer_grouped, group_name, domains
            )
            if avg_matrix is None:
                continue

            title = f"{'Attention' if group_name == 'attn' else 'MLP'} — Cross-Domain Similarity (Epoch {epoch})"
            path = os.path.join(epoch_dir, f"{group_name}_mean.png")
            plot_heatmap(avg_matrix, domains, title, path)

        # ── Per-module heatmaps ───────────────────────────────────────────
        for module in ALL_MODULES:
            avg_matrix = average_similarity_matrix_across_layers(
                per_layer_module, module, domains
            )
            if avg_matrix is None:
                continue

            group = "Attn" if module in ATTN_MODULES else "MLP"
            title = f"{module} ({group}) — Cross-Domain Similarity (Epoch {epoch})"
            path = os.path.join(epoch_dir, f"{module}.png")
            plot_heatmap(avg_matrix, domains, title, path)

    # ── Comparison: Attn vs MLP across epochs ─────────────────────────────
    comparison_dir = ensure_dir(os.path.join(PLOTS_DIR, "3_domain_specificity", "comparison"))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for col, epoch in enumerate(EPOCHS):
        data = load_domain_specificity(epoch)
        if data is None:
            continue
        domains = data["config"]["domains"]
        per_layer_grouped = data["per_layer_grouped"]

        for row, group_name in enumerate(["attn", "mlp"]):
            ax = axes[row, col]
            avg_matrix = average_similarity_matrix_across_layers(
                per_layer_grouped, group_name, domains
            )
            if avg_matrix is None:
                continue

            n = len(domains)
            labels = [DOMAIN_LABELS.get(d, d) for d in domains]
            im = ax.imshow(avg_matrix, cmap="RdYlBu_r", vmin=0, vmax=0.5, aspect="equal")

            for i in range(n):
                for j in range(n):
                    val = avg_matrix[i, j]
                    if not np.isnan(val):
                        text_color = "white" if val > 0.35 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, fontweight="bold", color=text_color)

            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=8)

            group_label = "Attention" if group_name == "attn" else "MLP"
            ax.set_title(f"{group_label} — Epoch {epoch}", fontsize=11, fontweight="bold")

    fig.suptitle("Cross-Domain Similarity: Attention vs MLP Across Epochs",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")
    path = os.path.join(comparison_dir, "attn_vs_mlp_all_epochs.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Generating All Experiment Plots")
    print("=" * 65)

    # ── 1. Fine-tuning plots ──────────────────────────────────────────────
    print("\n[1/3] Fine-tuning: Train & Val Perplexity")
    print("─" * 50)
    training_logs = load_training_logs()
    baseline = load_baseline()
    if training_logs:
        plot_finetuning(training_logs, baseline)
    else:
        print("  [SKIP] No training logs found")

    # ── 2. Weight Change Norms ────────────────────────────────────────────
    print("\n[2/3] Weight Change Norms: δ across layers")
    print("─" * 50)
    plot_weight_change_norms()

    # ── 3. Domain Specificity ─────────────────────────────────────────────
    print("\n[3/3] Domain Specificity: 6×6 Heatmaps")
    print("─" * 50)
    plot_domain_specificity()

    # ── Summary ───────────────────────────────────────────────────────────
    total = 0
    for root, dirs, files in os.walk(PLOTS_DIR):
        total += len([f for f in files if f.endswith(".png")])

    print(f"\n{'=' * 65}")
    print(f"  Done! Generated {total} plots.")
    print(f"  Output: {PLOTS_DIR}/")
    print(f"{'=' * 65}")

    # Print folder tree
    print(f"\n  Folder structure:")
    for root, dirs, files in os.walk(PLOTS_DIR):
        level = root.replace(PLOTS_DIR, "").count(os.sep)
        indent = "    " + "  " * level
        basename = os.path.basename(root)
        print(f"{indent}{basename}/")
        sub_indent = "    " + "  " * (level + 1)
        for f in sorted(files):
            print(f"{sub_indent}{f}")


if __name__ == "__main__":
    main()
