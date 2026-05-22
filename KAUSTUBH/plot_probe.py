"""
plot_probe.py — Visualize linear probing results.

Reads results/probe/probe_accuracy.npz and probe_summary.json,
generates publication-quality plots.

Usage:
    python plot_probe.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import COMPONENTS, NUM_LAYERS, RESULTS_DIR
from logger import setup_logger


# ── Style ──
COLORS = {
    "attn":  "#E74C3C",   # red
    "mlp":   "#2ECC71",   # green
    "layer": "#3498DB",   # blue
}
LABELS = {
    "attn":  "Attention (Δa)",
    "mlp":   "MLP (Δm)",
    "layer": "Residual Stream (x)",
}


def plot_accuracy_curves(summary: dict, out_dir: str, log):
    """Main probe accuracy plot — 3 curves across layers."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    layers = np.arange(NUM_LAYERS)

    for comp in ["layer", "attn", "mlp"]:
        accs = summary["components"][comp]["accuracy_per_layer"]
        peak = summary["components"][comp]["peak_layer"]
        peak_acc = summary["components"][comp]["peak_accuracy"]

        ax.plot(layers, accs, color=COLORS[comp], linewidth=2.2,
                label=f"{LABELS[comp]}  (peak L{peak}: {peak_acc:.3f})",
                marker="o", markersize=4, alpha=0.9)

        # Mark peak
        ax.plot(peak, peak_acc, "D", color=COLORS[comp], markersize=8,
                zorder=5, markeredgecolor="white", markeredgewidth=1.5)

    # Chance level
    ax.axhline(y=1/6, color="gray", linestyle=":", alpha=0.5, label="Chance (1/6)")

    ax.set_title("Linear Probe Accuracy — Domain Classification\n"
                 "Can a linear classifier decode domain from each component?",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Test Accuracy (6-class)", fontsize=12)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.set_ylim(0.6, 0.95)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=10, framealpha=0.9, loc="lower right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    path = os.path.join(out_dir, "probe_accuracy.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    path = os.path.join(out_dir, "probe_accuracy.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    log(f"  Saved: {path}")


def plot_attn_minus_mlp(summary: dict, out_dir: str, log):
    """Plot the difference: attn accuracy minus mlp accuracy per layer."""
    fig, ax = plt.subplots(figsize=(12, 4))
    layers = np.arange(NUM_LAYERS)

    attn_accs = np.array(summary["components"]["attn"]["accuracy_per_layer"])
    mlp_accs = np.array(summary["components"]["mlp"]["accuracy_per_layer"])
    diff = attn_accs - mlp_accs

    colors = ["#E74C3C" if d > 0 else "#2ECC71" for d in diff]
    ax.bar(layers, diff, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)

    ax.set_title("Attention − MLP Probe Accuracy (per layer)\n"
                 "Red = attention more informative, Green = MLP more informative",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Δ Accuracy (attn − mlp)", fontsize=12)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    path = os.path.join(out_dir, "probe_attn_minus_mlp.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")

    path = os.path.join(out_dir, "probe_attn_minus_mlp.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    log(f"  Saved: {path}")


def plot_layer_profile_shapes(summary: dict, out_dir: str, log):
    """
    Normalize each curve to [0,1] to compare SHAPE, ignoring absolute level.
    Shows where each component's domain info peaks relative to its own range.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(NUM_LAYERS)

    for comp in ["attn", "mlp", "layer"]:
        accs = np.array(summary["components"][comp]["accuracy_per_layer"])
        # Min-max normalize
        normed = (accs - accs.min()) / (accs.max() - accs.min() + 1e-9)
        ax.plot(layers, normed, color=COLORS[comp], linewidth=2.2,
                label=LABELS[comp], marker="o", markersize=3, alpha=0.85)

    ax.set_title("Probe Accuracy — Normalized Shape Comparison\n"
                 "Each curve min-max normalized to [0,1] to compare layer profiles",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Normalized Accuracy", fontsize=12)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    path = os.path.join(out_dir, "probe_shape_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    path = os.path.join(out_dir, "probe_shape_comparison.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    log(f"  Saved: {path}")


def main():
    log = setup_logger("plot_probe")

    probe_dir = os.path.join(RESULTS_DIR, "probe")
    json_path = os.path.join(probe_dir, "probe_summary.json")

    with open(json_path) as f:
        summary = json.load(f)

    log("Generating probe plots...")

    plot_accuracy_curves(summary, probe_dir, log)
    plot_attn_minus_mlp(summary, probe_dir, log)
    plot_layer_profile_shapes(summary, probe_dir, log)

    log("\nDone. All plots saved to results/probe/")


if __name__ == "__main__":
    main()
