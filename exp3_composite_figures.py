#!/usr/bin/env python3
"""
exp3_composite_figures.py — Gap-filling analysis for Experiment 3.

Generates:
  1. The "Key Figure" from §8.3: hotspot vs non-hotspot vs MLP propagation
  2. Perturbation magnitude validation: ‖perturbation‖ / √tr(S_W)
  3. Bootstrap confidence intervals for key shift metrics
  4. Updated final-shift comparison with Attn/MLP ratio annotations

Requires:
  - Attention propagation: patching_all_pairs/shift_scores_all_pairs.pt
  - MLP propagation:       patching_mlp_pairs/shift_scores_mlp_pairs.pt  (if available)
  - Final shift:           final_shift/final_shift_scores.pt
  - FDR data:              exp1_fdr/ (for S_W values)
"""

import argparse, json, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": "#c9d1d9", "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d", "legend.fontsize": 9,
    "grid.color": "#21262d", "grid.alpha": 0.6,
    "font.family": "sans-serif", "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "#0e1117",
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results/exp3_causal_alldomains")
    p.add_argument("--fdr-dir", default="./results/exp1_fdr")
    p.add_argument("--output-dir", default="./results/exp3_causal_alldomains/composite_figures")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Hotspot vs Non-Hotspot Propagation (the "Key Figure" from §8.3)
# ─────────────────────────────────────────────────────────────────────────────
def plot_key_figure(results_dir, output_dir):
    """
    §8.3 specifies: 'one curve for patches at hotspot attention layers and
    one for patches at non-hotspot layers (and MLP).'
    
    Hotspot layers: L20, L24 (FDR peak band L19-24)
    Non-hotspot layers: L4, L8, L12
    """
    # Load attention propagation
    attn_tensor = torch.load(
        os.path.join(results_dir, "patching_all_pairs/shift_scores_all_pairs.pt"),
        weights_only=True, map_location="cpu").numpy()

    with open(os.path.join(results_dir, "patching_all_pairs/all_pairs_results.json")) as f:
        attn_meta = json.load(f)

    patch_layers = attn_meta["patch_layers"]  # [4, 8, 12, 16, 20, 24]
    num_layers = attn_tensor.shape[2]
    n_pairs = attn_tensor.shape[0]

    # Define hotspot and non-hotspot groups
    hotspot_pls = [pl for pl in patch_layers if pl >= 20]   # [20, 24]
    nonhot_pls  = [pl for pl in patch_layers if pl <= 12]   # [4, 8, 12]

    hotspot_idxs = [patch_layers.index(pl) for pl in hotspot_pls]
    nonhot_idxs  = [patch_layers.index(pl) for pl in nonhot_pls]

    # Check if MLP propagation exists
    mlp_path = os.path.join(results_dir, "patching_mlp_pairs/shift_scores_mlp_pairs.pt")
    has_mlp = os.path.exists(mlp_path)
    if has_mlp:
        mlp_tensor = torch.load(mlp_path, weights_only=True, map_location="cpu").numpy()
        print(f"  MLP propagation data found: {mlp_tensor.shape}")
    else:
        print("  ⚠ No MLP propagation data found — plotting attention-only comparison")

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Hotspot vs Non-Hotspot Propagation (Attention Patches)",
                 fontsize=14, fontweight="bold")

    # For the composite curve, we align by "layers after patch"
    max_post = max(num_layers - pl for pl in patch_layers)

    # ── Hotspot attention curve (mean over hotspot patch layers × all pairs) ──
    hotspot_curves = []
    for pl_idx in hotspot_idxs:
        pl = patch_layers[pl_idx]
        for pair_idx in range(n_pairs):
            scores = attn_tensor[pair_idx, pl_idx, pl:]  # post-patch
            # Pad to max_post length
            padded = np.full(max_post, np.nan)
            padded[:len(scores)] = scores
            hotspot_curves.append(padded)
    hotspot_arr = np.array(hotspot_curves)
    hotspot_mean = np.nanmean(hotspot_arr, axis=0)
    hotspot_std = np.nanstd(hotspot_arr, axis=0)
    hotspot_x = np.arange(max_post)

    ax.plot(hotspot_x, hotspot_mean, "-o", color="#7ee787", linewidth=2.5,
            markersize=4, label=f"Attn Hotspot (L{','.join(map(str, hotspot_pls))})",
            zorder=5)
    ax.fill_between(hotspot_x,
                     hotspot_mean - hotspot_std,
                     hotspot_mean + hotspot_std,
                     alpha=0.15, color="#7ee787")

    # ── Non-hotspot attention curve ──
    nonhot_curves = []
    for pl_idx in nonhot_idxs:
        pl = patch_layers[pl_idx]
        for pair_idx in range(n_pairs):
            scores = attn_tensor[pair_idx, pl_idx, pl:]
            padded = np.full(max_post, np.nan)
            padded[:len(scores)] = scores
            nonhot_curves.append(padded)
    nonhot_arr = np.array(nonhot_curves)
    nonhot_mean = np.nanmean(nonhot_arr, axis=0)
    nonhot_std = np.nanstd(nonhot_arr, axis=0)

    ax.plot(hotspot_x, nonhot_mean, "-s", color="#8b949e", linewidth=2,
            markersize=3, label=f"Attn Non-Hotspot (L{','.join(map(str, nonhot_pls))})",
            alpha=0.8, zorder=4)
    ax.fill_between(hotspot_x,
                     nonhot_mean - nonhot_std,
                     nonhot_mean + nonhot_std,
                     alpha=0.1, color="#8b949e")

    # ── MLP hotspot curve (if available) ──
    if has_mlp:
        mlp_hotspot_curves = []
        for pl_idx in hotspot_idxs:
            pl = patch_layers[pl_idx]
            for pair_idx in range(n_pairs):
                scores = mlp_tensor[pair_idx, pl_idx, pl:]
                padded = np.full(max_post, np.nan)
                padded[:len(scores)] = scores
                mlp_hotspot_curves.append(padded)
        mlp_arr = np.array(mlp_hotspot_curves)
        mlp_mean = np.nanmean(mlp_arr, axis=0)
        mlp_std = np.nanstd(mlp_arr, axis=0)

        ax.plot(hotspot_x, mlp_mean, "--D", color="#ffa657", linewidth=2,
                markersize=3,
                label=f"MLP Hotspot (L{','.join(map(str, hotspot_pls))})",
                alpha=0.85, zorder=3)
        ax.fill_between(hotspot_x,
                         mlp_mean - mlp_std,
                         mlp_mean + mlp_std,
                         alpha=0.1, color="#ffa657")

    ax.axhline(0, color="#484f58", alpha=0.5, ls="-")
    ax.set_xlabel("Layers After Patch (ℓ − ℓ*)")
    ax.set_ylabel("Mean Shift Score s$^{\\ell}$")
    ax.legend(framealpha=0.8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Annotate
    ax.annotate("Hotspot patches amplify\nthrough downstream layers",
                xy=(3, hotspot_mean[3]), xytext=(6, hotspot_mean[3] + 2),
                arrowprops=dict(arrowstyle="->", color="#7ee787", lw=1.5),
                fontsize=9, color="#7ee787", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_key_figure_hotspot_vs_nonhotspot.png"))
    plt.close(fig)
    print("  ✓ 01_key_figure_hotspot_vs_nonhotspot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Final Domain Shift with Attn/MLP ratio annotations
# ─────────────────────────────────────────────────────────────────────────────
def plot_final_shift_annotated(results_dir, output_dir):
    """Enhanced final shift plot with Attn/MLP ratio at key layers."""
    t = torch.load(os.path.join(results_dir, "final_shift/final_shift_scores.pt"),
                    weights_only=True, map_location="cpu")
    num_layers = t.shape[2]
    layers = np.arange(num_layers)

    # Mean and std across pairs
    attn_mean = t[:, 0, :].mean(dim=0).numpy()
    attn_std  = t[:, 0, :].std(dim=0).numpy()
    mlp_mean  = t[:, 1, :].mean(dim=0).numpy()
    mlp_std   = t[:, 1, :].std(dim=0).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Final Domain Shift by Patch Layer — Attention vs MLP",
                 fontsize=14, fontweight="bold", y=0.98)

    # Main shift curves
    ax1.plot(layers, attn_mean, "-o", color="#58a6ff", linewidth=2.5,
             markersize=4, label="Attention Patch", zorder=5)
    ax1.fill_between(layers, attn_mean - attn_std, attn_mean + attn_std,
                      alpha=0.12, color="#58a6ff")
    ax1.plot(layers, mlp_mean, "--s", color="#ffa657", linewidth=2,
             markersize=3, label="MLP Patch", alpha=0.85, zorder=4)
    ax1.fill_between(layers, mlp_mean - mlp_std, mlp_mean + mlp_std,
                      alpha=0.1, color="#ffa657")

    ax1.axhline(0, color="#484f58", alpha=0.5)

    # Annotate key layers
    key_layers = [9, 19, 22, 24]
    for l in key_layers:
        ratio = attn_mean[l] / max(mlp_mean[l], 0.01)
        color = "#7ee787" if ratio > 1.5 else "#ffa657" if ratio < 0.7 else "#c9d1d9"
        ax1.annotate(f"L{l}: {ratio:.1f}×",
                     xy=(l, max(attn_mean[l], mlp_mean[l]) + 1),
                     fontsize=8, fontweight="bold", color=color,
                     ha="center", va="bottom")

    # Highlight FDR hotspot band
    ax1.axvspan(19, 24, alpha=0.08, color="#7ee787", label="FDR Hotspot Band (L19-L24)")

    ax1.set_ylabel("Final Domain Shift")
    ax1.legend(framealpha=0.8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, num_layers - 0.5)

    # Attn/MLP ratio subplot
    ratio = np.where(np.abs(mlp_mean) > 0.01,
                      attn_mean / np.abs(mlp_mean), np.nan)
    valid = ~np.isnan(ratio) & (np.abs(ratio) < 20)

    ax2.bar(layers[valid], ratio[valid], color=np.where(ratio[valid] > 1, "#58a6ff", "#ffa657"),
            alpha=0.7, edgecolor="#30363d", linewidth=0.5)
    ax2.axhline(1.0, color="#c9d1d9", ls="--", alpha=0.5, label="Parity (1.0)")
    ax2.axhspan(19, 24, alpha=0.08, color="#7ee787")
    ax2.set_xlabel("Patch Layer ℓ*")
    ax2.set_ylabel("Attn / MLP Ratio")
    ax2.legend(framealpha=0.7, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, num_layers - 0.5)
    ax2.set_ylim(-2, 8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_final_shift_annotated.png"))
    plt.close(fig)
    print("  ✓ 02_final_shift_annotated.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Perturbation magnitude validation
# ─────────────────────────────────────────────────────────────────────────────
def plot_perturbation_validation(results_dir, fdr_dir, output_dir):
    """
    §8: 'The magnitude of the injected perturbation ‖x̃^{ℓ*} - x^{ℓ*}‖ should
    be small relative to √tr(S_W^{ℓ*}).'
    
    Approximation: perturbation magnitude ≈ |alpha| * ‖d̂_B‖ + |proj_A| * ‖d̂_A‖
    Since d̂ are unit vectors, perturbation ≈ alpha + avg_proj ≈ 2 * alpha
    """
    # Load alpha values from domain vectors
    domains = ["cs", "eess", "math", "physics", "q-bio", "stat"]
    alphas_attn = []
    for d in domains:
        d_dir = os.path.join(results_dir, d)
        alpha_path = os.path.join(d_dir, "alpha_projection_magnitudes.pt")
        if os.path.exists(alpha_path):
            a = torch.load(alpha_path, weights_only=True, map_location="cpu")
            alphas_attn.append(a.float())

    if not alphas_attn:
        print("  ⚠ No alpha data found, skipping perturbation validation")
        return

    # Mean alpha across domains per layer
    alpha_mean = torch.stack(alphas_attn).mean(dim=0).numpy()
    num_layers = len(alpha_mean)

    # Load S_W from FDR npz results
    fdr_npz = os.path.join(fdr_dir, "exp1_fdr_scores.npz")
    if not os.path.exists(fdr_npz):
        print(f"  ⚠ FDR scores not found at {fdr_npz}, skipping")
        return

    fdr_data = np.load(fdr_npz)
    sw_attn = fdr_data["last_attn_tr_sw"]  # (28,) tr(S_W) per layer

    # Perturbation magnitude ≈ 2 * alpha (remove + inject, both unit-magnitude operations)
    perturbation = 2.0 * np.abs(alpha_mean)
    spread = np.sqrt(np.maximum(sw_attn, 1e-8))
    ratio = perturbation / spread

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle("Perturbation Magnitude Validation (Linear Regime Check)",
                 fontsize=14, fontweight="bold")

    layers = np.arange(num_layers)

    # Top: absolute magnitudes
    ax1.semilogy(layers, perturbation, "-o", color="#58a6ff", label="‖perturbation‖ ≈ 2α",
                 markersize=3, linewidth=2)
    ax1.semilogy(layers, spread, "-s", color="#ffa657", label="√tr(S_W) (within-domain spread)",
                 markersize=3, linewidth=2)
    ax1.axvspan(19, 24, alpha=0.08, color="#7ee787")
    ax1.set_ylabel("Magnitude (log scale)")
    ax1.legend(framealpha=0.8)
    ax1.grid(True, alpha=0.3)

    # Bottom: ratio
    ax2.bar(layers, ratio, color=np.where(ratio < 0.1, "#7ee787", "#ffa657"),
            alpha=0.7, edgecolor="#30363d", linewidth=0.5)
    ax2.axhline(0.1, color="#f778ba", ls="--", alpha=0.7, label="10% threshold")
    ax2.axhline(0.01, color="#7ee787", ls="--", alpha=0.7, label="1% threshold")
    ax2.axvspan(19, 24, alpha=0.08, color="#7ee787")
    ax2.set_xlabel("Layer ℓ*")
    ax2.set_ylabel("‖perturbation‖ / √tr(S_W)")
    ax2.legend(framealpha=0.7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_perturbation_validation.png"))
    plt.close(fig)
    print("  ✓ 03_perturbation_validation.png")

    # Print key values
    print(f"    Perturbation/Spread ratio at hotspot layers:")
    for l in [19, 20, 21, 22, 23, 24]:
        print(f"      L{l}: ratio = {ratio[l]:.4f} ({ratio[l]*100:.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────
def plot_bootstrap_ci(results_dir, output_dir, n_bootstrap=10000):
    """Bootstrap 95% CI for key metrics across domain pairs."""
    t = torch.load(os.path.join(results_dir, "final_shift/final_shift_scores.pt"),
                    weights_only=True, map_location="cpu")
    n_pairs = t.shape[0]
    rng = np.random.default_rng(42)

    key_layers = [4, 8, 12, 16, 19, 20, 22, 24]
    results = {"attn": {}, "mlp": {}}

    for comp_idx, comp in enumerate(["attn", "mlp"]):
        for l in key_layers:
            values = t[:, comp_idx, l].numpy()
            boot_means = np.array([
                values[rng.choice(n_pairs, n_pairs, replace=True)].mean()
                for _ in range(n_bootstrap)
            ])
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            results[comp][l] = {
                "mean": float(values.mean()),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "std": float(values.std()),
            }

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Final Domain Shift — 95% Bootstrap Confidence Intervals",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(key_layers))
    width = 0.35

    # Attention bars
    attn_means = [results["attn"][l]["mean"] for l in key_layers]
    attn_lo = [results["attn"][l]["mean"] - results["attn"][l]["ci_lo"] for l in key_layers]
    attn_hi = [results["attn"][l]["ci_hi"] - results["attn"][l]["mean"] for l in key_layers]
    ax.bar(x - width/2, attn_means, width, label="Attention",
           color="#58a6ff", alpha=0.8, edgecolor="#30363d",
           yerr=[attn_lo, attn_hi], capsize=4, error_kw={"color": "#c9d1d9", "linewidth": 1.5})

    # MLP bars
    mlp_means = [results["mlp"][l]["mean"] for l in key_layers]
    mlp_lo = [results["mlp"][l]["mean"] - results["mlp"][l]["ci_lo"] for l in key_layers]
    mlp_hi = [results["mlp"][l]["ci_hi"] - results["mlp"][l]["mean"] for l in key_layers]
    ax.bar(x + width/2, mlp_means, width, label="MLP",
           color="#ffa657", alpha=0.8, edgecolor="#30363d",
           yerr=[mlp_lo, mlp_hi], capsize=4, error_kw={"color": "#c9d1d9", "linewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in key_layers])
    ax.set_xlabel("Patch Layer ℓ*")
    ax.set_ylabel("Final Domain Shift (mean ± 95% CI)")
    ax.axhline(0, color="#484f58", alpha=0.5)
    ax.legend(framealpha=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate L19 dominance
    l19_idx = key_layers.index(19)
    if results["attn"][19]["ci_lo"] > results["mlp"][19]["ci_hi"]:
        ax.annotate("Attn > MLP\n(non-overlapping CIs)",
                     xy=(l19_idx, results["attn"][19]["mean"]),
                     xytext=(l19_idx + 1.5, results["attn"][19]["mean"] + 3),
                     arrowprops=dict(arrowstyle="->", color="#7ee787", lw=1.5),
                     fontsize=9, color="#7ee787", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_bootstrap_ci.png"))
    plt.close(fig)
    print("  ✓ 04_bootstrap_ci.png")

    # Save CI data
    with open(os.path.join(output_dir, "bootstrap_ci.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  ✓ bootstrap_ci.json")

    return results


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Experiment 3 — Composite Figures & Validation")
    print(f"{'='*60}\n")

    plot_key_figure(args.results_dir, args.output_dir)
    plot_final_shift_annotated(args.results_dir, args.output_dir)
    plot_perturbation_validation(args.results_dir, args.fdr_dir, args.output_dir)
    ci_results = plot_bootstrap_ci(args.results_dir, args.output_dir)

    # Print key bootstrap results
    print(f"\n{'='*60}")
    print("KEY BOOTSTRAP RESULTS")
    print(f"{'='*60}")
    for l in [19, 22, 24]:
        a = ci_results["attn"][l]
        m = ci_results["mlp"][l]
        overlap = a["ci_lo"] < m["ci_hi"] and m["ci_lo"] < a["ci_hi"]
        print(f"  L{l}: Attn={a['mean']:.2f} [{a['ci_lo']:.2f}, {a['ci_hi']:.2f}] | "
              f"MLP={m['mean']:.2f} [{m['ci_lo']:.2f}, {m['ci_hi']:.2f}] | "
              f"CIs {'overlap' if overlap else 'DO NOT overlap'}")

    print(f"\n  All figures saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
