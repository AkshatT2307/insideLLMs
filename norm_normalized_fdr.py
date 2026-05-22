#!/usr/bin/env python3
"""
norm_normalized_fdr.py

Computes FDR on unit-normalized activations to test whether MLP's separability
advantage is purely a scale effect.

For each activation x, we replace x → x/||x|| (project onto unit sphere),
then compute FDR = tr(S_B) / tr(S_W) on the normalized vectors.

If attention's normalized FDR > MLP's normalized FDR, it proves attention
is directionally more discriminative and MLP wins only via norm growth.

Two-pass algorithm:
  Pass 1: Compute per-domain means of normalized vectors (μ̂_k)
  Pass 2: Stream to compute within-class scatter tr(S_W)
  Between-class scatter tr(S_B) is computed directly from means (no streaming needed)
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def compute_normalized_fdr(activations_dir, vectors_file, output_dir, batch_size=1000):
    """Compute FDR on unit-normalized activations."""

    from calculate_separability import load_domain_data

    os.makedirs(output_dir, exist_ok=True)

    # Load metadata (we need domains and sample counts, not the vectors themselves)
    _, _, sample_counts, domains = load_domain_data(vectors_file, activations_dir)

    components = ["attn", "mlp", "layer"]
    modes = ["mean", "last"]
    K = len(domains)

    # Get num_layers from first file
    with h5py.File(os.path.join(activations_dir, f"{domains[0]}_activations.h5"), "r") as f:
        num_layers = f.attrs["num_layers"]
        hidden_dim = f.attrs["hidden_dim"]

    print(f"Domains: {domains}")
    print(f"Layers: {num_layers}, Hidden dim: {hidden_dim}")
    print(f"Computing FDR on unit-normalized activations\n")

    all_results = {}

    for mode in modes:
        print(f"{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")
        all_results[mode] = {}

        for comp in components:
            print(f"\n  Component: {comp}")

            # ── PASS 1: Compute normalized domain means ──────────────────
            print(f"    Pass 1: Computing normalized domain means...")
            # norm_means[domain] = (L, D) mean of x/||x||
            norm_means = {}
            for d in domains:
                act_path = os.path.join(activations_dir, f"{d}_activations.h5")
                accum = np.zeros((num_layers, hidden_dim), dtype=np.float64)
                n = 0

                with h5py.File(act_path, "r") as f:
                    dataset = f[f"{comp}/{mode}"]
                    N = dataset.shape[0]

                    for start in range(0, N, batch_size):
                        end = min(start + batch_size, N)
                        batch = dataset[start:end].astype(np.float64)  # (B, L, D)

                        # Normalize each vector: x/||x||
                        norms = np.linalg.norm(batch, axis=2, keepdims=True)  # (B, L, 1)
                        norms = np.maximum(norms, 1e-12)  # avoid div by zero
                        batch_normed = batch / norms  # (B, L, D)

                        accum += batch_normed.sum(axis=0)  # (L, D)
                        n += (end - start)

                norm_means[d] = accum / n
                print(f"      {d}: {n} samples ✓")

            # Global mean of normalized vectors
            n_total = sum(sample_counts[d] for d in domains)
            global_mean = np.zeros((num_layers, hidden_dim), dtype=np.float64)
            for d in domains:
                global_mean += norm_means[d] * (sample_counts[d] / n_total)

            # Between-class scatter: tr(S_B) = Σ_k n_k ||d̂_k||²
            tr_sb = np.zeros(num_layers, dtype=np.float64)
            for d in domains:
                dk = norm_means[d] - global_mean  # (L, D)
                tr_sb += sample_counts[d] * np.sum(dk ** 2, axis=1)

            # ── PASS 2: Compute within-class scatter ─────────────────────
            print(f"    Pass 2: Computing within-class scatter...")
            tr_sw = np.zeros(num_layers, dtype=np.float64)

            for d in domains:
                mu_k = norm_means[d]  # (L, D)
                act_path = os.path.join(activations_dir, f"{d}_activations.h5")

                with h5py.File(act_path, "r") as f:
                    dataset = f[f"{comp}/{mode}"]
                    N = dataset.shape[0]

                    for start in range(0, N, batch_size):
                        end = min(start + batch_size, N)
                        batch = dataset[start:end].astype(np.float64)

                        # Normalize
                        norms = np.linalg.norm(batch, axis=2, keepdims=True)
                        norms = np.maximum(norms, 1e-12)
                        batch_normed = batch / norms

                        # Within-class scatter
                        diff = batch_normed - mu_k[np.newaxis, :, :]
                        tr_sw += np.sum(diff ** 2, axis=(0, 2))

                        if start % (batch_size * 5) == 0:
                            print(f"      {d}: {end}/{N}", end="\r", flush=True)

                print(f"      {d}: {N}/{N} ✓")

            # FDR
            fdr = np.where(tr_sw > 0, tr_sb / tr_sw, 0.0)
            all_results[mode][comp] = {
                "fdr": fdr,
                "tr_sb": tr_sb,
                "tr_sw": tr_sw,
            }
            print(f"    Normalized FDR range: [{fdr.min():.6f}, {fdr.max():.6f}]")

    # ── Also compute mean activation norm per layer/comp/mode ────────────
    print(f"\n  Computing mean activation norms...")
    norm_profiles = {}
    for mode in modes:
        norm_profiles[mode] = {}
        for comp in components:
            layer_norms = np.zeros(num_layers, dtype=np.float64)
            total_n = 0
            for d in domains:
                act_path = os.path.join(activations_dir, f"{d}_activations.h5")
                with h5py.File(act_path, "r") as f:
                    dataset = f[f"{comp}/{mode}"]
                    N = dataset.shape[0]
                    for start in range(0, N, batch_size):
                        end = min(start + batch_size, N)
                        batch = dataset[start:end].astype(np.float64)
                        batch_norms = np.linalg.norm(batch, axis=2)  # (B, L)
                        layer_norms += batch_norms.sum(axis=0)  # (L,)
                        total_n += (end - start)
            norm_profiles[mode][comp] = layer_norms / total_n
            print(f"    {mode}/{comp}: mean norm range [{norm_profiles[mode][comp].min():.1f}, "
                  f"{norm_profiles[mode][comp].max():.1f}]")

    # ── Save ─────────────────────────────────────────────────────────────
    save_dict = {}
    for mode in modes:
        for comp in components:
            for key in ["fdr", "tr_sb", "tr_sw"]:
                save_dict[f"{mode}_{comp}_{key}"] = all_results[mode][comp][key]
            save_dict[f"{mode}_{comp}_mean_norm"] = norm_profiles[mode][comp]
    save_dict["domains"] = np.array(domains)
    npz_path = os.path.join(output_dir, "normalized_fdr_scores.npz")
    np.savez(npz_path, **save_dict)
    print(f"\n  Saved: {npz_path}")

    # ── Plot 1: Normalized FDR curves ────────────────────────────────────
    COLORS = {"attn": "#FF6B6B", "mlp": "#4ECDC4", "layer": "#6C5CE7"}
    LABELS = {"attn": "Attention Only", "mlp": "MLP Only", "layer": "Full Residual Stream"}
    layers = np.arange(num_layers)

    for mode in modes:
        fig, ax = plt.subplots(figsize=(12, 5))
        for comp in ["attn", "mlp", "layer"]:
            fdr = all_results[mode][comp]["fdr"]
            ax.plot(layers, fdr, color=COLORS[comp], linewidth=2.2,
                    label=LABELS[comp], marker="o", markersize=4, alpha=0.9)
        mode_title = "Mean-Pool" if mode == "mean" else "Last-Token"
        ax.set_title(f"Norm-Normalized FDR (unit sphere) — {mode_title}",
                     fontsize=16, fontweight="bold", pad=12)
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_ylabel("FDR on unit-normalized activations", fontsize=13)
        ax.set_xlim(-0.5, num_layers - 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.legend(fontsize=12, framealpha=0.9, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=11)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        path = os.path.join(output_dir, f"normalized_fdr_{mode}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {path}")

    # ── Plot 2: Mean activation norm profiles ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, mode in zip(axes, ["mean", "last"]):
        for comp in ["attn", "mlp", "layer"]:
            ax.semilogy(layers, norm_profiles[mode][comp], color=COLORS[comp],
                        linewidth=2.2, label=LABELS[comp], marker="o", markersize=4)
        mode_title = "Mean-Pool" if mode == "mean" else "Last-Token"
        ax.set_title(f"Mean Activation Norm — {mode_title}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("||x|| (log scale)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#FAFAFA")
        ax.set_xlim(-0.5, num_layers - 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = os.path.join(output_dir, "activation_norm_profiles.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")

    # ── Plot 3: Side-by-side original vs normalized FDR ──────────────────
    try:
        d_orig = np.load(os.path.join(output_dir, "fdr_scores.npz"))
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        for col, mode in enumerate(["mean", "last"]):
            mode_title = "Mean-Pool" if mode == "mean" else "Last-Token"

            # Row 0: Original FDR
            ax = axes[0, col]
            for comp in ["attn", "mlp", "layer"]:
                ax.plot(layers, d_orig[f"{mode}_{comp}_fdr"], color=COLORS[comp],
                        linewidth=2.2, label=LABELS[comp], marker="o", markersize=3)
            ax.set_title(f"Original FDR — {mode_title}", fontsize=13, fontweight="bold")
            ax.set_ylabel("FDR (raw)", fontsize=11)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_facecolor("#FAFAFA"); ax.set_xlim(-0.5, num_layers-0.5)

            # Row 1: Normalized FDR
            ax = axes[1, col]
            for comp in ["attn", "mlp", "layer"]:
                ax.plot(layers, all_results[mode][comp]["fdr"], color=COLORS[comp],
                        linewidth=2.2, label=LABELS[comp], marker="o", markersize=3)
            ax.set_title(f"Norm-Normalized FDR — {mode_title}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Layer", fontsize=11)
            ax.set_ylabel("FDR (unit-normalized)", fontsize=11)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_facecolor("#FAFAFA"); ax.set_xlim(-0.5, num_layers-0.5)

        fig.patch.set_facecolor("white")
        fig.suptitle("Effect of Norm Removal: Original vs Unit-Normalized FDR",
                     fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        path = os.path.join(output_dir, "fdr_original_vs_normalized.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {path}")
    except FileNotFoundError:
        print("  (Original FDR scores not found in same dir, skipping comparison plot)")

    print("\nDone!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--activations-dir", type=str, default="./activations")
    p.add_argument("--vectors-file", type=str, default="./results/domain_vectors.h5")
    p.add_argument("--output-dir", type=str, default="./results/sep")
    p.add_argument("--batch-size", type=int, default=1000)
    args = p.parse_args()

    compute_normalized_fdr(
        activations_dir=args.activations_dir,
        vectors_file=args.vectors_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
