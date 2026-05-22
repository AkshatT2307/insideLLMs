"""
plot_paper.py — Generate EMNLP-quality figures (PDF) for probing experiment.

Main paper: 2 figures (attn vs mlp only)
Appendix:   all figures with full detail

Usage:
    python plot_paper.py
    python plot_paper.py --config ../config.yaml
"""
import argparse
import os, sys, json
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.plot_style import (
    plt, ATTN_COLOR, MLP_COLOR, RES_COLOR, DOMAIN_COLORS_LIST,
    TEXT_WIDTH, COL_WIDTH, save_fig, setup_style,
)
from utils.logger import setup_logger
from utils.model_loader import get_model_slug

# Ensure style is applied
setup_style()

ATTN = ATTN_COLOR
MLP  = MLP_COLOR
RES  = RES_COLOR
DCOLORS = DOMAIN_COLORS_LIST


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Generate paper figures for probing")
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    return p.parse_args()


def load_data(results_dir):
    with open(os.path.join(results_dir, "probe/probe_summary.json")) as f:
        probe = json.load(f)
    fdr = np.load(os.path.join(results_dir, "separability/fdr_results.npz"))
    geo = np.load(os.path.join(results_dir, "geometry/geometry_results.npz"))
    return probe, fdr, geo


# ════════════════════════════════════════════════════════════
#  MAIN PAPER FIGURES
# ════════════════════════════════════════════════════════════

def main_fig1(probe, fdr, fig_dir, log, num_layers):
    """Figure 1: Probe accuracy + FDR (attn vs mlp)."""
    L = np.arange(num_layers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6))

    aa = probe["components"]["attn"]["accuracy_per_layer"]
    ma = probe["components"]["mlp"]["accuracy_per_layer"]
    ax1.plot(L, aa, color=ATTN, marker="o", ms=3, label="Attention ($\\Delta a$)")
    ax1.plot(L, ma, color=MLP, marker="s", ms=3, label="MLP ($\\Delta m$)")
    ax1.set_title("(a) Linear Probe Accuracy", fontweight="bold")
    ax1.set(xlabel="Layer", ylabel="Test Accuracy (6-class)",
            xlim=(-0.5, num_layers - 0.5))
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.2, ls="--")

    af = fdr["norm_fdr_attn"]
    mf = fdr["norm_fdr_mlp"]
    ax2.plot(L, af, color=ATTN, marker="o", ms=3, label="Attention ($\\Delta a$)")
    ax2.plot(L, mf, color=MLP, marker="s", ms=3, label="MLP ($\\Delta m$)")
    ax2.set_title("(b) Fisher Discriminant Ratio", fontweight="bold")
    ax2.set(xlabel="Layer", ylabel="FDR",
            xlim=(-0.5, num_layers - 0.5))
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.2, ls="--")

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, fig_dir, "probe_main_fig1", log)


def main_fig2(geo, fig_dir, log, num_layers, domains):
    """Figure 2: Concept vector norms (attn vs mlp), per domain."""
    L = np.arange(num_layers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6))

    for i, d in enumerate(domains):
        ax1.plot(L, geo["norms_attn"][:, i], color=DCOLORS[i], lw=1.4,
                 label=d, alpha=0.85)
        ax2.plot(L, geo["norms_mlp"][:, i], color=DCOLORS[i], lw=1.4,
                 label=d, alpha=0.85)

    ax1.set_title("(a) Attention $\\|\\mathbf{d}_k\\|$", fontweight="bold")
    ax1.set(xlabel="Layer", ylabel="Concept Vector Norm", xlim=(-0.5, num_layers - 0.5))
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.grid(True, alpha=0.2, ls="--")

    ax2.set_title("(b) MLP $\\|\\mathbf{d}_k\\|$", fontweight="bold")
    ax2.set(xlabel="Layer", ylabel="Concept Vector Norm", xlim=(-0.5, num_layers - 0.5))
    ax2.legend(fontsize=7, ncol=2, loc="upper left")
    ax2.grid(True, alpha=0.2, ls="--")

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, fig_dir, "probe_main_fig2", log)


# ════════════════════════════════════════════════════════════
#  APPENDIX FIGURES
# ════════════════════════════════════════════════════════════

def app_accuracy_all(probe, fig_dir, log, num_layers):
    """Probe accuracy — all 3 components."""
    L = np.arange(num_layers)
    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.8, 2.8))
    colors = {"attn": ATTN, "mlp": MLP, "layer": RES}
    labels = {"attn": "Attention ($\\Delta a$)", "mlp": "MLP ($\\Delta m$)",
              "layer": "Residual ($x$)"}
    for comp in ["layer", "attn", "mlp"]:
        ax.plot(L, probe["components"][comp]["accuracy_per_layer"],
                color=colors[comp], marker="o", ms=3, label=labels[comp])
    ax.set(xlabel="Layer", ylabel="Test Accuracy", xlim=(-0.5, num_layers - 0.5))
    ax.axhline(y=1/6, color="gray", ls=":", alpha=0.4, label="Chance")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.2, ls="--")
    ax.set_title("Linear Probe Accuracy — All Components", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, fig_dir, "probe_app_accuracy_all", log)


def app_difference(probe, fig_dir, log, num_layers):
    """Probe accuracy difference: attn - mlp."""
    L = np.arange(num_layers)
    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.8, 2.2))
    aa = np.array(probe["components"]["attn"]["accuracy_per_layer"])
    ma = np.array(probe["components"]["mlp"]["accuracy_per_layer"])
    diff = aa - ma
    colors = [ATTN if d > 0 else MLP for d in diff]
    ax.bar(L, diff, color=colors, alpha=0.8, edgecolor="white", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Layer", ylabel="$\\Delta$ Accuracy (Attn $-$ MLP)",
           xlim=(-0.5, num_layers - 0.5))
    ax.set_title("Probe Accuracy: Attention $-$ MLP", fontweight="bold")
    ax.grid(True, alpha=0.2, ls="--", axis="y")
    fig.tight_layout()
    save_fig(fig, fig_dir, "probe_app_difference", log)


def app_shape(probe, fig_dir, log, num_layers):
    """Normalized shape comparison."""
    L = np.arange(num_layers)
    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.8, 2.8))
    colors = {"attn": ATTN, "mlp": MLP, "layer": RES}
    labels = {"attn": "Attention ($\\Delta a$)", "mlp": "MLP ($\\Delta m$)",
              "layer": "Residual ($x$)"}
    for comp in ["attn", "mlp", "layer"]:
        a = np.array(probe["components"][comp]["accuracy_per_layer"])
        normed = (a - a.min()) / (a.max() - a.min() + 1e-9)
        ax.plot(L, normed, color=colors[comp], lw=1.8, marker="o", ms=3,
                label=labels[comp])
    ax.set(xlabel="Layer", ylabel="Normalized Accuracy [0,1]", xlim=(-0.5, num_layers - 0.5))
    ax.set_title("Probe Accuracy — Normalized Shape", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.2, ls="--")
    fig.tight_layout()
    save_fig(fig, fig_dir, "probe_app_shape", log)


def app_fdr(fdr, fig_dir, log, num_layers):
    """Raw and normalized FDR — all 3 components."""
    L = np.arange(num_layers)
    for kind, title in [("raw", "Raw"), ("norm", "Normalized")]:
        fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.8, 2.8))
        colors = {"attn": ATTN, "mlp": MLP, "layer": RES}
        labels = {"attn": "Attention", "mlp": "MLP", "layer": "Residual"}
        for comp in ["attn", "mlp", "layer"]:
            ax.plot(L, fdr[f"{kind}_fdr_{comp}"], color=colors[comp],
                    marker="o", ms=3, label=labels[comp])
        ax.set(xlabel="Layer", ylabel="FDR", xlim=(-0.5, num_layers - 0.5))
        ax.set_title(f"Fisher Discriminant Ratio — {title}", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.2, ls="--")
        fig.tight_layout()
        save_fig(fig, fig_dir, f"probe_app_fdr_{kind}", log)


def app_fdr_rawvsnorm(fdr, fig_dir, log, num_layers):
    """Raw vs normalized FDR for attn and mlp."""
    L = np.arange(num_layers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6))
    for ax, comp, col in [(ax1, "attn", ATTN), (ax2, "mlp", MLP)]:
        ax.plot(L, fdr[f"raw_fdr_{comp}"], color=col, ls="--", alpha=0.5,
                marker="o", ms=2, label="Raw")
        ax.plot(L, fdr[f"norm_fdr_{comp}"], color=col, lw=2,
                marker="s", ms=2, label="Normalized")
        name = "Attention ($\\Delta a$)" if comp == "attn" else "MLP ($\\Delta m$)"
        ax.set_title(name, fontweight="bold")
        ax.set(xlabel="Layer", ylabel="FDR", xlim=(-0.5, num_layers - 0.5))
        ax.legend(); ax.grid(True, alpha=0.2, ls="--")
    fig.suptitle("Raw vs Normalized FDR", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, fig_dir, "probe_app_fdr_rawvsnorm", log)


def app_cosine(geo, fig_dir, log, num_layers, domains):
    """Cosine heatmaps at key layers."""
    K = len(domains)
    # Pick 5 evenly spaced layers
    key_layers = [0, num_layers // 3, num_layers // 2,
                  int(num_layers * 0.8), num_layers - 1]
    for comp, col in [("attn", ATTN), ("mlp", MLP)]:
        fig, axes = plt.subplots(1, 5, figsize=(TEXT_WIDTH, 2.2))
        for ax, layer in zip(axes, key_layers):
            cos = geo[f"cosine_{comp}"][layer]
            im = ax.imshow(cos, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
            ax.set_title(f"L{layer}", fontsize=9, fontweight="bold")
            ax.set_xticks(range(K)); ax.set_xticklabels(domains, fontsize=6, rotation=45)
            ax.set_yticks(range(K)); ax.set_yticklabels(domains, fontsize=6)
        name = "Attention" if comp == "attn" else "MLP"
        fig.suptitle(f"Domain Cosine Similarity — {name}", fontweight="bold", y=1.04)
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.tight_layout()
        save_fig(fig, fig_dir, f"probe_app_cosine_{comp}", log)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_slug = get_model_slug(cfg["model"]["name"])
    results_base = os.path.join(os.path.dirname(args.config), cfg.get("results_dir", "./results"))
    results_dir = os.path.join(results_base, model_slug)

    domains = cfg["data"]["domains"]

    log = setup_logger("plot_paper", log_dir=os.path.join(results_dir, "logs"))
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    probe, fdr, geo = load_data(results_dir)

    # Auto-detect num_layers
    num_layers = len(probe["components"]["attn"]["accuracy_per_layer"])

    log("=== Main paper figures ===")
    main_fig1(probe, fdr, fig_dir, log, num_layers)
    main_fig2(geo, fig_dir, log, num_layers, domains)

    log("\n=== Appendix figures ===")
    app_accuracy_all(probe, fig_dir, log, num_layers)
    app_difference(probe, fig_dir, log, num_layers)
    app_shape(probe, fig_dir, log, num_layers)
    app_fdr(fdr, fig_dir, log, num_layers)
    app_fdr_rawvsnorm(fdr, fig_dir, log, num_layers)
    app_cosine(geo, fig_dir, log, num_layers, domains)

    log(f"\nDone. All figures in {fig_dir}/")


if __name__ == "__main__":
    main()
