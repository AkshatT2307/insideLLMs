#!/usr/bin/env python3
"""Plot all Experiment 3 (Domain Alignment) results."""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ───────────────────────────────────────────────────────────────
DOMAINS = ["cs", "eess", "math", "physics", "q-bio", "stat"]
EPOCHS = [1, 2, 3]
NUM_LAYERS = 28
MODES = ["last", "mean"]
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]

DATA_DIR = "results/experiment3_alignment"
PLOTS_DIR = "FineTuning/experiments/plots/4_domain_alignment"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
})

DOMAIN_COLORS = {
    "cs": "#2196F3", "eess": "#FF9800", "math": "#4CAF50",
    "physics": "#9C27B0", "q-bio": "#F44336", "stat": "#00BCD4",
}
DOMAIN_LABELS = {
    "cs": "CS", "eess": "EESS", "math": "Math",
    "physics": "Physics", "q-bio": "Q-Bio", "stat": "Stat",
}

def ensure(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_json(path):
    with open(path) as f:
        return json.load(f)

# ── Data loaders ─────────────────────────────────────────────────────────

def load_block_pooled(epoch, mode, domain):
    return load_json(os.path.join(DATA_DIR, "block_pooled", f"epoch_{epoch}", mode, f"{domain}.json"))

def load_per_module(epoch, mode, domain):
    return load_json(os.path.join(DATA_DIR, "per_module", f"epoch_{epoch}", mode, f"{domain}.json"))

def load_svd(epoch, mode, domain):
    return load_json(os.path.join(DATA_DIR, "svd_overlap", f"epoch_{epoch}", mode, f"{domain}.json"))

def load_cross_domain(epoch, mode, domain):
    return load_json(os.path.join(DATA_DIR, "cross_domain", f"epoch_{epoch}", mode, f"{domain}.json"))

# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: Per-domain layer-wise alignment (projection ρ)
# ═══════════════════════════════════════════════════════════════════════

def plot_projection_layer_curves():
    """For each domain: MLP vs Attention ρ across layers."""
    print("\n[1] Projection metric: per-domain layer curves")
    for epoch in EPOCHS:
        for mode in MODES:
            out = ensure(os.path.join(PLOTS_DIR, "projection_metric", f"epoch_{epoch}", mode))
            for domain in DOMAINS:
                bp = load_block_pooled(epoch, mode, domain)
                layers = list(range(NUM_LAYERS))
                mlp_rho = [bp[str(l)]["mlp"]["pooled_rho_frobenius_weighted"] for l in layers]
                attn_rho = [bp[str(l)]["attn"]["pooled_rho_frobenius_weighted"] for l in layers]

                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(layers, attn_rho, "o-", color="#1976D2", label="Attention (pooled)", linewidth=2, markersize=4)
                ax.plot(layers, mlp_rho, "s-", color="#E65100", label="MLP (pooled)", linewidth=2, markersize=4)
                ax.fill_between(layers, attn_rho, mlp_rho, alpha=0.08, color="#E65100")
                ax.set_xlabel("Layer")
                ax.set_ylabel("ρ (Domain Alignment)")
                ax.set_title(f"{DOMAIN_LABELS[domain]} — Domain Alignment ρ (Epoch {epoch}, {mode} pool)")
                ax.legend(loc="best")
                ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"{domain}_alignment.png"))
                plt.close(fig)

            # All domains overlay
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            for domain in DOMAINS:
                bp = load_block_pooled(epoch, mode, domain)
                layers = list(range(NUM_LAYERS))
                mlp_rho = [bp[str(l)]["mlp"]["pooled_rho_frobenius_weighted"] for l in layers]
                attn_rho = [bp[str(l)]["attn"]["pooled_rho_frobenius_weighted"] for l in layers]
                c = DOMAIN_COLORS[domain]
                lb = DOMAIN_LABELS[domain]
                axes[0].plot(layers, attn_rho, "-", color=c, label=lb, linewidth=1.8, alpha=0.85)
                axes[1].plot(layers, mlp_rho, "-", color=c, label=lb, linewidth=1.8, alpha=0.85)
            axes[0].set_title(f"Attention ρ (Epoch {epoch}, {mode})")
            axes[1].set_title(f"MLP ρ (Epoch {epoch}, {mode})")
            for ax in axes:
                ax.set_xlabel("Layer"); ax.set_ylabel("ρ")
                ax.legend(loc="best", fontsize=8); ax.set_xlim(-0.5, NUM_LAYERS-0.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            fig.suptitle(f"All Domains — Domain Alignment ρ (Epoch {epoch}, {mode})", fontsize=14, fontweight="bold", y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "all_domains_overlay.png"))
            plt.close(fig)
            print(f"  epoch {epoch}, {mode}: {len(DOMAINS)+1} plots")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: SVD overlap metric per domain
# ═══════════════════════════════════════════════════════════════════════

def plot_svd_layer_curves():
    """For each domain: SVD total_overlap_sq for MLP vs Attention modules."""
    print("\n[2] SVD metric: per-domain layer curves")
    for epoch in EPOCHS:
        for mode in MODES:
            out = ensure(os.path.join(PLOTS_DIR, "svd_metric", f"epoch_{epoch}", mode))
            for domain in DOMAINS:
                svd = load_svd(epoch, mode, domain)
                layers = list(range(NUM_LAYERS))

                # Frobenius-weighted SVD overlap per block
                pm = load_per_module(epoch, mode, domain)
                mlp_svd, attn_svd = [], []
                for l in layers:
                    # MLP
                    tw, ws = 0.0, 0.0
                    for m in MLP_MODULES:
                        w = pm[str(l)][m]["frobenius_norm"]
                        s = svd[str(l)][m]["total_overlap_sq"]
                        tw += w; ws += w * s
                    mlp_svd.append(ws / tw if tw > 1e-12 else 0)
                    # Attn
                    tw, ws = 0.0, 0.0
                    for m in ATTN_MODULES:
                        w = pm[str(l)][m]["frobenius_norm"]
                        s = svd[str(l)][m]["total_overlap_sq"]
                        tw += w; ws += w * s
                    attn_svd.append(ws / tw if tw > 1e-12 else 0)

                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(layers, attn_svd, "o-", color="#1976D2", label="Attention (pooled)", linewidth=2, markersize=4)
                ax.plot(layers, mlp_svd, "s-", color="#E65100", label="MLP (pooled)", linewidth=2, markersize=4)
                ax.fill_between(layers, attn_svd, mlp_svd, alpha=0.08, color="#E65100")
                ax.set_xlabel("Layer")
                ax.set_ylabel("SVD Overlap (squared)")
                ax.set_title(f"{DOMAIN_LABELS[domain]} — SVD Overlap with Domain Vector (Epoch {epoch}, {mode})")
                ax.legend(loc="best")
                ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"{domain}_svd.png"))
                plt.close(fig)
            print(f"  epoch {epoch}, {mode}: {len(DOMAINS)} plots")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: Cross-domain 6×6 heatmaps at most-distinguishing layer
# ═══════════════════════════════════════════════════════════════════════

def find_most_distinguishing_layer(epoch, mode):
    """Layer where same-domain ρ most exceeds cross-domain ρ for MLP."""
    best_layer, best_gap = 0, -1
    for l in range(NUM_LAYERS):
        same, cross_vals = [], []
        for ad in DOMAINS:
            cd = load_cross_domain(epoch, mode, ad)
            for m in MLP_MODULES:
                for td in DOMAINS:
                    val = cd[str(l)][m][td]
                    if td == ad:
                        same.append(val)
                    else:
                        cross_vals.append(val)
        gap = np.mean(same) - np.mean(cross_vals)
        if gap > best_gap:
            best_gap = gap; best_layer = l
    return best_layer

def build_cross_matrix(epoch, mode, layer, modules):
    """Build 6×6 matrix: row=adapter domain, col=target domain vector."""
    n = len(DOMAINS)
    mat = np.zeros((n, n))
    for i, ad in enumerate(DOMAINS):
        cd = load_cross_domain(epoch, mode, ad)
        for j, td in enumerate(DOMAINS):
            vals = [cd[str(layer)][m][td] for m in modules]
            mat[i, j] = np.mean(vals)
    return mat

def plot_cross_domain_heatmaps():
    """Side-by-side Attention vs MLP heatmaps at most-distinguishing layer."""
    print("\n[3] Cross-domain heatmaps")
    for epoch in EPOCHS:
        for mode in MODES:
            out = ensure(os.path.join(PLOTS_DIR, "cross_domain_heatmaps", f"epoch_{epoch}", mode))
            best_l = find_most_distinguishing_layer(epoch, mode)
            labels = [DOMAIN_LABELS[d] for d in DOMAINS]

            attn_mat = build_cross_matrix(epoch, mode, best_l, ATTN_MODULES)
            mlp_mat = build_cross_matrix(epoch, mode, best_l, MLP_MODULES)
            vmin = min(np.min(attn_mat), np.min(mlp_mat))
            vmax = max(np.max(attn_mat), np.max(mlp_mat))

            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            for ax, mat, title in zip(axes, [attn_mat, mlp_mat],
                                       [f"Attention — Layer {best_l}", f"MLP — Layer {best_l}"]):
                im = ax.imshow(mat, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal")
                for i in range(6):
                    for j in range(6):
                        tc = "white" if mat[i,j] > (vmin+vmax)/2*1.3 else "black"
                        ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center",
                                fontsize=8, fontweight="bold", color=tc)
                ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticks(range(6)); ax.set_yticklabels(labels)
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_ylabel("Adapter Domain"); ax.set_xlabel("Target Domain Vector")

            fig.suptitle(f"Cross-Domain Alignment ρ — Most Distinguishing Layer {best_l}\n(Epoch {epoch}, {mode} pool)",
                         fontsize=13, fontweight="bold", y=1.04)
            fig.colorbar(im, ax=axes, shrink=0.8, label="ρ (alignment)")
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"heatmap_layer{best_l}.png"))
            plt.close(fig)

            # Also: average across all layers
            attn_avg = np.mean([build_cross_matrix(epoch, mode, l, ATTN_MODULES) for l in range(NUM_LAYERS)], axis=0)
            mlp_avg = np.mean([build_cross_matrix(epoch, mode, l, MLP_MODULES) for l in range(NUM_LAYERS)], axis=0)
            vmin2 = min(np.min(attn_avg), np.min(mlp_avg))
            vmax2 = max(np.max(attn_avg), np.max(mlp_avg))

            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            for ax, mat, title in zip(axes, [attn_avg, mlp_avg],
                                       ["Attention (avg all layers)", "MLP (avg all layers)"]):
                im = ax.imshow(mat, cmap="RdYlBu_r", vmin=vmin2, vmax=vmax2, aspect="equal")
                for i in range(6):
                    for j in range(6):
                        tc = "white" if mat[i,j] > (vmin2+vmax2)/2*1.3 else "black"
                        ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center",
                                fontsize=8, fontweight="bold", color=tc)
                ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticks(range(6)); ax.set_yticklabels(labels)
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_ylabel("Adapter Domain"); ax.set_xlabel("Target Domain Vector")
            fig.suptitle(f"Cross-Domain Alignment ρ — Average Across All Layers\n(Epoch {epoch}, {mode} pool)",
                         fontsize=13, fontweight="bold", y=1.04)
            fig.colorbar(im, ax=axes, shrink=0.8, label="ρ (alignment)")
            fig.tight_layout()
            fig.savefig(os.path.join(out, "heatmap_avg_all_layers.png"))
            plt.close(fig)
            print(f"  epoch {epoch}, {mode}: best layer={best_l}, 2 heatmaps saved")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: Summary — MLP vs Attention bar comparison
# ═══════════════════════════════════════════════════════════════════════

def plot_mlp_vs_attn_summary():
    """Bar chart: mean ρ for MLP vs Attention per domain, across epochs."""
    print("\n[4] Summary: MLP vs Attention bar charts")
    for mode in MODES:
        out = ensure(os.path.join(PLOTS_DIR, "summary", mode))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for col, epoch in enumerate(EPOCHS):
            ax = axes[col]
            mlp_means, attn_means = [], []
            for domain in DOMAINS:
                bp = load_block_pooled(epoch, mode, domain)
                mlp_vals = [bp[str(l)]["mlp"]["pooled_rho_frobenius_weighted"] for l in range(NUM_LAYERS)]
                attn_vals = [bp[str(l)]["attn"]["pooled_rho_frobenius_weighted"] for l in range(NUM_LAYERS)]
                mlp_means.append(np.mean(mlp_vals))
                attn_means.append(np.mean(attn_vals))

            x = np.arange(len(DOMAINS))
            w = 0.35
            ax.bar(x - w/2, attn_means, w, label="Attention", color="#1976D2", alpha=0.85)
            ax.bar(x + w/2, mlp_means, w, label="MLP", color="#E65100", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS], rotation=45, ha="right")
            ax.set_ylabel("Mean ρ (across layers)")
            ax.set_title(f"Epoch {epoch}")
            ax.legend(loc="best")
        fig.suptitle(f"MLP vs Attention: Mean Domain Alignment ({mode} pool)", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "mlp_vs_attn_bar.png"))
        plt.close(fig)
        print(f"  {mode}: saved")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: Epoch progression
# ═══════════════════════════════════════════════════════════════════════

def plot_epoch_progression():
    """How ρ evolves across epochs for each domain."""
    print("\n[5] Epoch progression")
    for mode in MODES:
        out = ensure(os.path.join(PLOTS_DIR, "epoch_progression", mode))
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for idx, domain in enumerate(DOMAINS):
            ax = axes[idx // 3, idx % 3]
            for epoch in EPOCHS:
                bp = load_block_pooled(epoch, mode, domain)
                layers = list(range(NUM_LAYERS))
                mlp_rho = [bp[str(l)]["mlp"]["pooled_rho_frobenius_weighted"] for l in layers]
                attn_rho = [bp[str(l)]["attn"]["pooled_rho_frobenius_weighted"] for l in layers]
                alpha = 0.4 + 0.2 * epoch
                ax.plot(layers, mlp_rho, "-", color="#E65100", alpha=alpha, linewidth=1.5+0.3*epoch,
                        label=f"MLP ep{epoch}")
                ax.plot(layers, attn_rho, "--", color="#1976D2", alpha=alpha, linewidth=1.5+0.3*epoch,
                        label=f"Attn ep{epoch}")
            ax.set_title(DOMAIN_LABELS[domain], fontweight="bold")
            ax.set_xlabel("Layer"); ax.set_ylabel("ρ")
            ax.set_xlim(-0.5, NUM_LAYERS-0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            if idx == 0:
                ax.legend(loc="best", fontsize=7, ncol=2)
        fig.suptitle(f"Epoch Progression of Domain Alignment ({mode} pool)", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "epoch_progression_all.png"))
        plt.close(fig)
        print(f"  {mode}: saved")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 6: Per-module breakdown
# ═══════════════════════════════════════════════════════════════════════

MODULE_COLORS = {
    "q_proj": "#1565C0", "k_proj": "#42A5F5", "v_proj": "#0D47A1", "o_proj": "#64B5F6",
    "gate_proj": "#E65100", "up_proj": "#FF9800", "down_proj": "#FFB74D",
}
MODULE_LS = {
    "q_proj": "-", "k_proj": "--", "v_proj": "-.", "o_proj": ":",
    "gate_proj": "-", "up_proj": "--", "down_proj": "-.",
}

def plot_per_module_breakdown():
    """Detailed per-module ρ across layers for each domain."""
    print("\n[6] Per-module breakdown")
    for epoch in EPOCHS:
        for mode in MODES:
            out = ensure(os.path.join(PLOTS_DIR, "per_module", f"epoch_{epoch}", mode))
            for domain in DOMAINS:
                pm = load_per_module(epoch, mode, domain)
                layers = list(range(NUM_LAYERS))
                fig, ax = plt.subplots(figsize=(12, 5))
                for mod in ATTN_MODULES + MLP_MODULES:
                    vals = [pm[str(l)][mod]["rho"] for l in layers]
                    grp = "attn" if mod in ATTN_MODULES else "mlp"
                    ax.plot(layers, vals, color=MODULE_COLORS[mod], linestyle=MODULE_LS[mod],
                            label=f"{mod} ({grp})", linewidth=1.8, alpha=0.85)
                ax.set_xlabel("Layer"); ax.set_ylabel("ρ")
                ax.set_title(f"{DOMAIN_LABELS[domain]} — Per-Module ρ (Epoch {epoch}, {mode})")
                ax.legend(loc="best", ncol=2, fontsize=8)
                ax.set_xlim(-0.5, NUM_LAYERS-0.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"{domain}_per_module.png"))
                plt.close(fig)
            print(f"  epoch {epoch}, {mode}: {len(DOMAINS)} plots")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 7: MLP with/without down_proj comparison
# ═══════════════════════════════════════════════════════════════════════

def plot_down_proj_effect():
    """Compare MLP ρ with and without down_proj pooling."""
    print("\n[7] Down_proj pooling effect")
    for mode in MODES:
        out = ensure(os.path.join(PLOTS_DIR, "down_proj_effect", mode))
        for epoch in EPOCHS:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            for idx, domain in enumerate(DOMAINS):
                ax = axes[idx // 3, idx % 3]
                bp = load_block_pooled(epoch, mode, domain)
                layers = list(range(NUM_LAYERS))
                with_dp = [bp[str(l)]["mlp"]["pooled_rho_frobenius_weighted"] for l in layers]
                without_dp = [bp[str(l)]["mlp"]["pooled_rho_no_down_proj"] for l in layers]
                ax.plot(layers, with_dp, "-", color="#E65100", label="MLP (with down_proj)", linewidth=2)
                ax.plot(layers, without_dp, "--", color="#FF9800", label="MLP (no down_proj)", linewidth=2)
                ax.set_title(DOMAIN_LABELS[domain], fontweight="bold")
                ax.set_xlabel("Layer"); ax.set_ylabel("ρ")
                ax.set_xlim(-0.5, NUM_LAYERS-0.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
                if idx == 0: ax.legend(loc="best", fontsize=8)
            fig.suptitle(f"Effect of down_proj on MLP ρ (Epoch {epoch}, {mode})", fontsize=14, fontweight="bold", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"epoch_{epoch}.png"))
            plt.close(fig)
        print(f"  {mode}: {len(EPOCHS)} plots")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 8: Cross-domain specificity ratio
# ═══════════════════════════════════════════════════════════════════════

def plot_specificity_ratio():
    """Ratio of same-domain ρ to mean cross-domain ρ per layer."""
    print("\n[8] Specificity ratio (same/cross)")
    for epoch in EPOCHS:
        for mode in MODES:
            out = ensure(os.path.join(PLOTS_DIR, "specificity_ratio", f"epoch_{epoch}", mode))
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            for idx, domain in enumerate(DOMAINS):
                ax = axes[idx // 3, idx % 3]
                cd = load_cross_domain(epoch, mode, domain)
                layers = list(range(NUM_LAYERS))
                mlp_ratio, attn_ratio = [], []
                for l in layers:
                    for modules, ratios in [(MLP_MODULES, mlp_ratio), (ATTN_MODULES, attn_ratio)]:
                        same = np.mean([cd[str(l)][m][domain] for m in modules])
                        cross = np.mean([cd[str(l)][m][td] for m in modules for td in DOMAINS if td != domain])
                        ratios.append(same / cross if cross > 1e-12 else 0)
                ax.plot(layers, mlp_ratio, "s-", color="#E65100", label="MLP", linewidth=2, markersize=3)
                ax.plot(layers, attn_ratio, "o-", color="#1976D2", label="Attention", linewidth=2, markersize=3)
                ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
                ax.set_title(DOMAIN_LABELS[domain], fontweight="bold")
                ax.set_xlabel("Layer"); ax.set_ylabel("Same/Cross ρ ratio")
                ax.set_xlim(-0.5, NUM_LAYERS-0.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
                if idx == 0: ax.legend(loc="best", fontsize=8)
            fig.suptitle(f"Domain Specificity Ratio (Epoch {epoch}, {mode})", fontsize=14, fontweight="bold", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "specificity_ratio.png"))
            plt.close(fig)
            print(f"  epoch {epoch}, {mode}: saved")

# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Generating Experiment 3 (Domain Alignment) Plots")
    print("=" * 60)

    plot_projection_layer_curves()
    plot_svd_layer_curves()
    plot_cross_domain_heatmaps()
    plot_mlp_vs_attn_summary()
    plot_epoch_progression()
    plot_per_module_breakdown()
    plot_down_proj_effect()
    plot_specificity_ratio()

    total = sum(1 for r, _, fs in os.walk(PLOTS_DIR) for f in fs if f.endswith(".png"))
    print(f"\n{'='*60}")
    print(f"  Done! {total} plots saved to {PLOTS_DIR}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
