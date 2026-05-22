#!/usr/bin/env python3
"""
exp3_all_pairs_analysis.py — Consistency analysis across all domain pairs.

Reads the all-pairs patching results and produces:
  1. Per-pair propagation curves (grid of 30 subplots)
  2. Mean propagation by patch layer (averaged over all pairs)
  3. Consistency heatmap: mean post-patch shift per (source, target, patch_layer)
  4. Three-zone classification for each pair
  5. Pair-similarity matrix: do all pairs show the same propagation pattern?
  6. Source-domain and target-domain aggregated views
  7. Combined summary figure
"""

import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.plot_style import (
    plt, setup_style, save_fig, ATTN_COLOR, MLP_COLOR, RES_COLOR,
    DOMAIN_COLORS, TEXT_WIDTH, COL_WIDTH,
)
import matplotlib.gridspec as gridspec

setup_style()

DOMAIN_COLOURS = DOMAIN_COLORS
PCMAP = plt.cm.plasma


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=str,
                    default="./results/exp3_causal_alldomains/patching_all_pairs")
    p.add_argument("--vectors-dir", type=str, default="./results/exp3_causal_alldomains")
    p.add_argument("--output-dir", type=str,
                    default="./results/exp3_causal_alldomains/plots")
    return p.parse_args()


def load_results(results_dir):
    with open(os.path.join(results_dir, "all_pairs_results.json")) as f:
        meta = json.load(f)
    tensor = torch.load(os.path.join(results_dir, "shift_scores_all_pairs.pt"),
                         weights_only=True).numpy()
    return meta, tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mean propagation curves averaged over all pairs
# ─────────────────────────────────────────────────────────────────────────────
def plot_mean_propagation(meta, tensor, output_dir):
    """Mean ± std of shift scores across all pairs, per patch layer."""
    patch_layers = meta["patch_layers"]
    num_layers = tensor.shape[2]
    n_pairs = tensor.shape[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Mean Propagation Across All {n_pairs} Domain Pairs",
                 fontsize=14, fontweight="bold")

    cvals = np.linspace(0.15, 0.85, len(patch_layers))
    for j, pl in enumerate(patch_layers):
        scores = tensor[:, j, :]  # (n_pairs, num_layers)
        mean = scores.mean(axis=0)
        std = scores.std(axis=0)
        post = list(range(pl, num_layers))
        c = PCMAP(cvals[j])
        ax.plot(post, mean[post], "-o", color=c, label=f"ℓ*={pl}",
                markersize=3, linewidth=2, alpha=0.85)
        ax.fill_between(post, mean[post] - std[post], mean[post] + std[post],
                         alpha=0.15, color=c)
        ax.axvline(pl, color=c, ls="--", alpha=0.2, lw=1)

    ax.axhline(0, color="#484f58", alpha=0.5)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mean Shift Score s$^{\\ell}$ (± 1 std)")
    ax.legend(framealpha=0.7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_layers - 1)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_mean_propagation.png"))
    plt.close(fig)
    print("  ✓ 01_mean_propagation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-pair propagation grid
# ─────────────────────────────────────────────────────────────────────────────
def plot_pair_grid(meta, tensor, output_dir):
    """Grid of propagation curves, one subplot per pair."""
    pair_keys = meta["pair_keys"]
    patch_layers = meta["patch_layers"]
    domains = meta["domains"]
    num_layers = tensor.shape[2]
    K = len(domains)

    fig, axes = plt.subplots(K, K - 1, figsize=(3 * (K - 1), 2.5 * K), squeeze=False)
    fig.suptitle("Propagation Curves for All Domain Pairs",
                 fontsize=14, fontweight="bold", y=1.01)

    pair_idx_map = {pk: i for i, pk in enumerate(pair_keys)}
    cvals = np.linspace(0.15, 0.85, len(patch_layers))

    for row, source in enumerate(domains):
        targets = [d for d in domains if d != source]
        for col, target in enumerate(targets):
            ax = axes[row, col]
            pk = f"{source}->{target}"
            idx = pair_idx_map[pk]

            for j, pl in enumerate(patch_layers):
                scores = tensor[idx, j]
                post = list(range(pl, num_layers))
                ax.plot(post, scores[post], "-", color=PCMAP(cvals[j]),
                        linewidth=1, alpha=0.8)

            ax.axhline(0, color="#484f58", alpha=0.4, lw=0.5)
            ax.set_title(f"{source}→{target}", fontsize=9, fontweight="bold")
            ax.set_xlim(0, num_layers - 1)
            if row == K - 1:
                ax.set_xlabel("Layer", fontsize=8)
            if col == 0:
                ax.set_ylabel("s$^{\\ell}$", fontsize=8)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_pair_grid.png"))
    plt.close(fig)
    print("  ✓ 02_pair_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Consistency heatmap: mean post-patch shift
# ─────────────────────────────────────────────────────────────────────────────
def plot_consistency_heatmap(meta, tensor, output_dir):
    """For each patch layer, show a source×target heatmap of mean post-patch shift."""
    pair_keys = meta["pair_keys"]
    patch_layers = meta["patch_layers"]
    domains = meta["domains"]
    K = len(domains)
    num_layers = tensor.shape[2]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Mean Post-Patch Shift by Domain Pair",
                 fontsize=14, fontweight="bold", y=1.01)

    pair_idx_map = {pk: i for i, pk in enumerate(pair_keys)}

    for ax_idx, pl in enumerate(patch_layers):
        ax = axes[ax_idx // 3, ax_idx % 3]
        mat = np.full((K, K), np.nan)
        for si, src in enumerate(domains):
            for ti, tgt in enumerate(domains):
                if src == tgt:
                    continue
                pk = f"{src}->{tgt}"
                idx = pair_idx_map[pk]
                j = patch_layers.index(pl)
                post_scores = tensor[idx, j, pl + 1:]
                mat[si, ti] = post_scores.mean()

        vmax = max(abs(np.nanmax(mat)), abs(np.nanmin(mat)))
        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(K))
        ax.set_xticklabels(domains, fontsize=8)
        ax.set_yticks(range(K))
        ax.set_yticklabels(domains, fontsize=8)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        ax.set_title(f"Patch @ L{pl}", fontweight="bold")

        # Annotate cells
        for si in range(K):
            for ti in range(K):
                if not np.isnan(mat[si, ti]):
                    ax.text(ti, si, f"{mat[si, ti]:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if abs(mat[si, ti]) < vmax * 0.5 else "white")

        fig.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_consistency_heatmap.png"))
    plt.close(fig)
    print("  ✓ 03_consistency_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Three-zone classification
# ─────────────────────────────────────────────────────────────────────────────
def plot_zone_classification(meta, tensor, output_dir):
    """Classify each pair's propagation pattern per patch layer into
    amplification (>0.1), persistence (-0.1 to 0.1), decay (<-0.1)."""
    pair_keys = meta["pair_keys"]
    patch_layers = meta["patch_layers"]
    num_layers = tensor.shape[2]

    categories = {"amplify": [], "persist": [], "decay": [], "reverse": []}
    zone_data = []

    for i, pk in enumerate(pair_keys):
        for j, pl in enumerate(patch_layers):
            post_scores = tensor[i, j, pl + 1:]
            if len(post_scores) == 0:
                continue
            mean_shift = float(post_scores.mean())
            # Check trend: does it grow or shrink?
            if len(post_scores) >= 3:
                first_third = post_scores[:len(post_scores) // 3].mean()
                last_third = post_scores[-len(post_scores) // 3:].mean()
                trend = last_third - first_third
            else:
                trend = 0

            if mean_shift > 0.1 and trend > 0:
                zone = "amplify"
            elif mean_shift > 0.05:
                zone = "persist"
            elif mean_shift < -0.05:
                zone = "reverse"
            else:
                zone = "decay"

            zone_data.append({"pair": pk, "patch_layer": pl, "zone": zone,
                              "mean_shift": mean_shift, "trend": float(trend)})

    # Count zones per patch layer
    zone_names = ["amplify", "persist", "decay", "reverse"]
    zone_colors = ["#7ee787", "#79c0ff", "#ffa657", "#f778ba"]
    counts = {pl: {z: 0 for z in zone_names} for pl in patch_layers}
    for zd in zone_data:
        counts[zd["patch_layer"]][zd["zone"]] += 1

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Propagation Zone Classification Across All Pairs",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(patch_layers))
    width = 0.18
    for zi, (zone, color) in enumerate(zip(zone_names, zone_colors)):
        vals = [counts[pl][zone] for pl in patch_layers]
        ax.bar(x + zi * width, vals, width, label=zone.capitalize(), color=color,
               alpha=0.85, edgecolor="#30363d")

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"L{pl}" for pl in patch_layers])
    ax.set_xlabel("Patch Layer")
    ax.set_ylabel("Number of Pairs")
    ax.legend(framealpha=0.7)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_zone_classification.png"))
    plt.close(fig)
    print("  ✓ 04_zone_classification.png")

    # Save classification data
    with open(os.path.join(output_dir, "zone_classification.json"), "w") as f:
        json.dump({"zone_data": zone_data, "counts": {str(k): v for k, v in counts.items()}},
                  f, indent=2)

    return zone_data, counts


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pair-to-pair correlation of propagation profiles
# ─────────────────────────────────────────────────────────────────────────────
def plot_pair_correlation(meta, tensor, output_dir):
    """Correlation matrix of propagation profiles across all pairs."""
    n_pairs = tensor.shape[0]
    pair_keys = meta["pair_keys"]
    # Flatten (patch_layers, num_layers) into a single vector per pair
    profiles = tensor.reshape(n_pairs, -1)  # (n_pairs, n_pl * n_layers)
    # Cosine similarity
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = profiles / norms
    corr = normed @ normed.T

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.suptitle("Pair-to-Pair Propagation Profile Similarity (Cosine)",
                 fontsize=14, fontweight="bold")
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(pair_keys, rotation=90, fontsize=6)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_keys, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_pair_correlation.png"))
    plt.close(fig)
    print("  ✓ 05_pair_correlation.png")

    return corr


# ─────────────────────────────────────────────────────────────────────────────
# 6. Source- and target-aggregated views
# ─────────────────────────────────────────────────────────────────────────────
def plot_aggregated_views(meta, tensor, output_dir):
    """Aggregate shift scores by source domain and by target domain."""
    pair_keys = meta["pair_keys"]
    patch_layers = meta["patch_layers"]
    domains = meta["domains"]
    num_layers = tensor.shape[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Domain-Aggregated Propagation",
                 fontsize=14, fontweight="bold", y=1.02)

    # Use patch_layer=24 (late, where amplification is strongest)
    pl_idx = patch_layers.index(24) if 24 in patch_layers else -1

    # By source
    for src in domains:
        idxs = [i for i, pk in enumerate(pair_keys) if pk.startswith(f"{src}->")]
        if not idxs:
            continue
        mean_curve = tensor[idxs, pl_idx, :].mean(axis=0)
        post = list(range(24, num_layers))
        ax1.plot(post, mean_curve[post], "-o", color=DOMAIN_COLOURS.get(src, "#8b949e"),
                 label=src, markersize=3, linewidth=2)

    ax1.axhline(0, color="#484f58", alpha=0.5)
    ax1.set_title("By Source Domain (patch@L24)", fontweight="bold")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean Shift Score")
    ax1.legend(framealpha=0.7)
    ax1.grid(True, alpha=0.3)

    # By target
    for tgt in domains:
        idxs = [i for i, pk in enumerate(pair_keys) if pk.endswith(f"->{tgt}")]
        if not idxs:
            continue
        mean_curve = tensor[idxs, pl_idx, :].mean(axis=0)
        post = list(range(24, num_layers))
        ax2.plot(post, mean_curve[post], "-o", color=DOMAIN_COLOURS.get(tgt, "#8b949e"),
                 label=tgt, markersize=3, linewidth=2)

    ax2.axhline(0, color="#484f58", alpha=0.5)
    ax2.set_title("By Target Domain (patch@L24)", fontweight="bold")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Shift Score")
    ax2.legend(framealpha=0.7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_aggregated_views.png"))
    plt.close(fig)
    print("  ✓ 06_aggregated_views.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_summary_stats(meta, tensor, output_dir, zone_data, corr):
    """Compute and save consistency statistics."""
    pair_keys = meta["pair_keys"]
    patch_layers = meta["patch_layers"]
    num_layers = tensor.shape[2]

    stats = {
        "num_pairs": len(pair_keys),
        "num_patch_layers": len(patch_layers),
        "overall_consistency": {},
    }

    # Per patch layer: what fraction of pairs show positive mean shift?
    for j, pl in enumerate(patch_layers):
        post_means = [tensor[i, j, pl + 1:].mean() for i in range(len(pair_keys))]
        pos_frac = sum(1 for m in post_means if m > 0) / len(post_means)
        neg_frac = sum(1 for m in post_means if m < 0) / len(post_means)
        stats["overall_consistency"][f"L{pl}"] = {
            "mean_across_pairs": float(np.mean(post_means)),
            "std_across_pairs": float(np.std(post_means)),
            "fraction_positive": round(pos_frac, 3),
            "fraction_negative": round(neg_frac, 3),
            "min": float(np.min(post_means)),
            "max": float(np.max(post_means)),
        }

    # Zone consistency
    zone_counts_by_pl = {}
    for zd in zone_data:
        pl = zd["patch_layer"]
        zone = zd["zone"]
        if pl not in zone_counts_by_pl:
            zone_counts_by_pl[pl] = {}
        zone_counts_by_pl[pl][zone] = zone_counts_by_pl[pl].get(zone, 0) + 1

    stats["zone_consistency"] = {str(k): v for k, v in zone_counts_by_pl.items()}

    # Pair correlation stats
    upper_tri = corr[np.triu_indices_from(corr, k=1)]
    stats["pair_correlation"] = {
        "mean": float(upper_tri.mean()),
        "std": float(upper_tri.std()),
        "min": float(upper_tri.min()),
        "max": float(upper_tri.max()),
    }

    # Three-zone test: for majority of pairs, is early=decay, mid=reverse, late=amplify?
    early_pls = [pl for pl in patch_layers if pl <= 12]
    mid_pls = [pl for pl in patch_layers if 13 <= pl <= 20]
    late_pls = [pl for pl in patch_layers if pl >= 21]

    zone_pattern = {"early_decay": 0, "mid_reverse": 0, "late_amplify": 0, "total_checks": 0}
    for zd in zone_data:
        pl = zd["patch_layer"]
        if pl in early_pls and zd["zone"] in ["decay", "persist"]:
            zone_pattern["early_decay"] += 1
        if pl in mid_pls and zd["zone"] in ["reverse", "decay"]:
            zone_pattern["mid_reverse"] += 1
        if pl in late_pls and zd["zone"] in ["amplify", "persist"]:
            zone_pattern["late_amplify"] += 1
        zone_pattern["total_checks"] += 1

    n_early = len(pair_keys) * len(early_pls)
    n_mid = len(pair_keys) * len(mid_pls)
    n_late = len(pair_keys) * len(late_pls)
    stats["three_zone_test"] = {
        "early_decay_fraction": round(zone_pattern["early_decay"] / max(n_early, 1), 3),
        "mid_reverse_fraction": round(zone_pattern["mid_reverse"] / max(n_mid, 1), 3),
        "late_amplify_fraction": round(zone_pattern["late_amplify"] / max(n_late, 1), 3),
    }

    with open(os.path.join(output_dir, "consistency_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("  ✓ consistency_stats.json")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("All-Pairs Consistency Analysis")
    print(f"{'='*60}\n")

    meta, tensor = load_results(args.results_dir)
    print(f"  Loaded: {tensor.shape} — {meta['num_pairs']} pairs, "
          f"{len(meta['patch_layers'])} patch layers, {tensor.shape[2]} layers")

    plot_mean_propagation(meta, tensor, args.output_dir)
    plot_pair_grid(meta, tensor, args.output_dir)
    plot_consistency_heatmap(meta, tensor, args.output_dir)
    zone_data, counts = plot_zone_classification(meta, tensor, args.output_dir)
    corr = plot_pair_correlation(meta, tensor, args.output_dir)
    plot_aggregated_views(meta, tensor, args.output_dir)
    stats = compute_summary_stats(meta, tensor, args.output_dir, zone_data, corr)

    # Print key findings
    print(f"\n{'='*60}")
    print("KEY CONSISTENCY RESULTS")
    print(f"{'='*60}")
    print(f"\nPair-to-pair correlation: mean={stats['pair_correlation']['mean']:.3f} "
          f"± {stats['pair_correlation']['std']:.3f}")
    print(f"\nThree-zone test:")
    print(f"  Early (L4-12) decay:    {stats['three_zone_test']['early_decay_fraction']:.1%}")
    print(f"  Mid (L16-20) reverse:   {stats['three_zone_test']['mid_reverse_fraction']:.1%}")
    print(f"  Late (L24) amplify:     {stats['three_zone_test']['late_amplify_fraction']:.1%}")
    print(f"\nPer patch layer:")
    for pl_key, pl_stats in stats["overall_consistency"].items():
        print(f"  {pl_key}: mean={pl_stats['mean_across_pairs']:.4f} "
              f"± {pl_stats['std_across_pairs']:.4f}  "
              f"(+:{pl_stats['fraction_positive']:.0%} / −:{pl_stats['fraction_negative']:.0%})")

    print(f"\n{'='*60}")
    print(f"All plots saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
