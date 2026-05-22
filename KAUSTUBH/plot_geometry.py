"""
plot_geometry.py — Visualize geometry results.

Usage:
    python plot_geometry.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import DOMAINS, COMPONENTS, NUM_LAYERS, RESULTS_DIR
from logger import setup_logger

COLORS = {"attn": "#E74C3C", "mlp": "#2ECC71", "layer": "#3498DB"}
LABELS = {"attn": "Attention (Δa)", "mlp": "MLP (Δm)", "layer": "Residual (x)"}
DOMAIN_COLORS = ["#E74C3C", "#F39C12", "#2ECC71", "#3498DB", "#9B59B6", "#1ABC9C"]


def plot_simplex_deviation(data, out_dir, log):
    """Simplex deviation curves for all 3 components."""
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(NUM_LAYERS)

    for comp in COMPONENTS:
        sd = data[f"simplex_{comp}"]
        best = int(np.argmin(sd))
        ax.plot(layers, sd, color=COLORS[comp], linewidth=2.2,
                label=f"{LABELS[comp]}  (best L{best}: {sd[best]:.2f})",
                marker="o", markersize=4)
        ax.plot(best, sd[best], "D", color=COLORS[comp], markersize=8,
                zorder=5, markeredgecolor="white", markeredgewidth=1.5)

    ax.set_title("Simplex Deviation — Distance from Ideal Equidistant Geometry\n"
                 "Lower = domains more symmetrically separated",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Frobenius Distance from Ideal Simplex", fontsize=12)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()

    path = os.path.join(out_dir, "simplex_deviation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")


def plot_concept_norms(data, out_dir, log):
    """Concept vector norm profiles — one subplot per component."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    layers = np.arange(NUM_LAYERS)

    for ax, comp in zip(axes, COMPONENTS):
        norms = data[f"norms_{comp}"]  # (L, K)
        for i, domain in enumerate(DOMAINS):
            ax.plot(layers, norms[:, i], color=DOMAIN_COLORS[i],
                    linewidth=1.8, label=domain, alpha=0.85)
        ax.set_title(LABELS[comp], fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel("‖d_k‖ (concept vector norm)", fontsize=10)
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle("Domain Concept Vector Norms Across Layers",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, "concept_norms.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")


def plot_cosine_heatmaps(data, out_dir, log):
    """Cosine similarity heatmaps at selected layers for attn vs mlp."""
    key_layers = [0, 9, 14, 22, 27]

    for comp in ["attn", "mlp"]:
        fig, axes = plt.subplots(1, len(key_layers), figsize=(18, 3.5))
        for ax, layer in zip(axes, key_layers):
            cos = data[f"cosine_{comp}"][layer]
            im = ax.imshow(cos, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
            ax.set_title(f"L{layer}", fontsize=11, fontweight="bold")
            ax.set_xticks(range(len(DOMAINS)))
            ax.set_xticklabels(DOMAINS, fontsize=7, rotation=45)
            ax.set_yticks(range(len(DOMAINS)))
            ax.set_yticklabels(DOMAINS, fontsize=7)

        fig.suptitle(f"Domain Cosine Similarity — {LABELS[comp]}",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.colorbar(im, ax=axes, shrink=0.8, label="Cosine Similarity")
        fig.tight_layout()

        path = os.path.join(out_dir, f"cosine_heatmaps_{comp}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: {path}")


def main():
    log = setup_logger("plot_geometry")
    geo_dir = os.path.join(RESULTS_DIR, "geometry")
    data = np.load(os.path.join(geo_dir, "geometry_results.npz"))

    log("Generating geometry plots...")
    plot_simplex_deviation(data, geo_dir, log)
    plot_concept_norms(data, geo_dir, log)
    plot_cosine_heatmaps(data, geo_dir, log)
    log("\nDone.")


if __name__ == "__main__":
    main()
