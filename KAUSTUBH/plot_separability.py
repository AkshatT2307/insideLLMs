"""
plot_separability.py — Visualize FDR results.

Usage:
    python plot_separability.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import COMPONENTS, NUM_LAYERS, RESULTS_DIR
from logger import setup_logger

COLORS = {"attn": "#E74C3C", "mlp": "#2ECC71", "layer": "#3498DB"}
LABELS = {"attn": "Attention (Δa)", "mlp": "MLP (Δm)", "layer": "Residual (x)"}


def plot_fdr_curves(data, kind, title_suffix, out_dir, log):
    """Plot FDR curves for all 3 components."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    layers = np.arange(NUM_LAYERS)

    for comp in COMPONENTS:
        fdr = data[f"{kind}_fdr_{comp}"]
        peak = int(np.argmax(fdr))
        cv = float(np.std(fdr) / np.mean(fdr)) if np.mean(fdr) > 0 else 0
        ax.plot(layers, fdr, color=COLORS[comp], linewidth=2.2,
                label=f"{LABELS[comp]}  (peak L{peak}, CV={cv:.2f})",
                marker="o", markersize=4)
        ax.plot(peak, fdr[peak], "D", color=COLORS[comp], markersize=8,
                zorder=5, markeredgecolor="white", markeredgewidth=1.5)

    ax.set_title(f"Fisher Discriminant Ratio — {title_suffix}\n"
                 f"FDR = tr(S_B) / tr(S_W), higher = better separation",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("FDR", fontsize=12)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()

    path = os.path.join(out_dir, f"fdr_{kind}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")


def plot_raw_vs_norm(data, out_dir, log):
    """Side-by-side: raw vs normalized FDR for attn and mlp."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    layers = np.arange(NUM_LAYERS)

    for ax, comp in zip(axes, ["attn", "mlp"]):
        raw = data[f"raw_fdr_{comp}"]
        norm = data[f"norm_fdr_{comp}"]

        ax.plot(layers, raw, color=COLORS[comp], linewidth=2, alpha=0.5,
                linestyle="--", label="Raw FDR", marker="o", markersize=3)
        ax.plot(layers, norm, color=COLORS[comp], linewidth=2.5,
                label="Normalized FDR", marker="s", markersize=3)

        ax.set_title(f"{LABELS[comp]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("FDR", fontsize=11)
        ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")

    fig.suptitle("Raw vs Normalized FDR — Effect of Removing Magnitude",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, "fdr_raw_vs_norm.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")


def plot_norm_attn_minus_mlp(data, out_dir, log):
    """Bar chart: normalized attn FDR minus normalized mlp FDR per layer."""
    fig, ax = plt.subplots(figsize=(12, 4))
    layers = np.arange(NUM_LAYERS)

    attn = data["norm_fdr_attn"]
    mlp = data["norm_fdr_mlp"]
    diff = attn - mlp
    colors = ["#E74C3C" if d > 0 else "#2ECC71" for d in diff]

    ax.bar(layers, diff, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)

    ax.set_title("Normalized FDR: Attention − MLP (per layer)\n"
                 "Red = attention directionally better, Green = MLP directionally better",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Δ Normalized FDR", fontsize=12)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    fig.tight_layout()

    path = os.path.join(out_dir, "fdr_norm_attn_minus_mlp.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")


def main():
    log = setup_logger("plot_separability")
    sep_dir = os.path.join(RESULTS_DIR, "separability")
    data = np.load(os.path.join(sep_dir, "fdr_results.npz"))

    log("Generating separability plots...")
    plot_fdr_curves(data, "raw", "Raw Activations", sep_dir, log)
    plot_fdr_curves(data, "norm", "Unit-Normalized Activations (direction only)", sep_dir, log)
    plot_raw_vs_norm(data, sep_dir, log)
    plot_norm_attn_minus_mlp(data, sep_dir, log)
    log("\nDone.")


if __name__ == "__main__":
    main()
