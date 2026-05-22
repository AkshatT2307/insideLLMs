#!/usr/bin/env python3
"""
calculate_separability.py

Computes multi-class separability metrics (default: Fisher Discriminant Ratio)
across transformer layers for domain-specific activations.

The FDR at layer ℓ is:
    FDR^ℓ = tr(S_B^ℓ) / tr(S_W^ℓ)

where:
    tr(S_B) = Σ_k n_k ||d_k||²           (between-class scatter, from domain vectors)
    tr(S_W) = Σ_k Σ_{x∈D_k} ||x - μ_k||² (within-class scatter, streamed from activations)

Usage:
    python calculate_separability.py \
        --activations-dir ./activations \
        --vectors-file ./results/domain_vectors.h5 \
        --output-dir ./results \
        --batch-size 500
"""

import argparse
import os
import glob
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# Pluggable Metric Interface
# ─────────────────────────────────────────────────────────────────────────────

class SeparabilityMetric(ABC):
    """
    Abstract base class for multi-class separability metrics.

    To implement a new metric, subclass this and implement:
        - name (property): human-readable name for plot titles/legends
        - compute_per_layer(...): returns a 1-D array of metric values, one per layer

    The compute_per_layer method receives:
        - domain_vectors: dict  {domain: {comp: {mode: array(num_layers, hidden_dim)}}}
        - domain_means:   dict  {domain: {comp: {mode: array(num_layers, hidden_dim)}}}
        - sample_counts:  dict  {domain: int}
        - activations_dir: str  path to raw activation HDF5 files
        - component: str        one of 'attn', 'mlp', 'layer'
        - mode: str             one of 'mean', 'last'
        - batch_size: int       batch size for streaming
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name for plots and file naming."""
        pass

    @abstractmethod
    def compute_per_layer(
        self,
        domain_vectors: Dict,
        domain_means: Dict,
        sample_counts: Dict,
        activations_dir: str,
        component: str,
        mode: str,
        batch_size: int = 500,
    ) -> np.ndarray:
        """
        Compute the metric value at every layer.

        Returns
        -------
        scores : np.ndarray, shape (num_layers,)
        """
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Fisher Discriminant Ratio
# ─────────────────────────────────────────────────────────────────────────────

class FisherDiscriminantRatio(SeparabilityMetric):
    """
    Multiclass Fisher Discriminant Ratio:  FDR = tr(S_B) / tr(S_W)

    Between-class scatter trace:
        tr(S_B) = Σ_k  n_k  ||d_k||²
    where d_k is the domain concept vector (domain mean minus global mean).

    Within-class scatter trace:
        tr(S_W) = Σ_k Σ_{x ∈ D_k}  ||x - μ_k||²
    computed by streaming through the raw activation files.
    """

    @property
    def name(self) -> str:
        return "Fisher Discriminant Ratio"

    # ── between-class scatter ────────────────────────────────────────────
    @staticmethod
    def _between_class_scatter(
        domain_vectors: Dict,
        sample_counts: Dict,
        component: str,
        mode: str,
    ) -> np.ndarray:
        """
        Compute tr(S_B) per layer.

        Parameters
        ----------
        domain_vectors : {domain: {comp: {mode: array(L, D)}}}
        sample_counts  : {domain: int}

        Returns
        -------
        tr_sb : np.ndarray, shape (num_layers,)
        """
        domains = list(domain_vectors.keys())
        dv0 = domain_vectors[domains[0]][component][mode]
        num_layers = dv0.shape[0]

        tr_sb = np.zeros(num_layers, dtype=np.float64)
        for d in domains:
            dv = domain_vectors[d][component][mode].astype(np.float64)  # (L, D)
            nk = sample_counts[d]
            tr_sb += nk * np.sum(dv ** 2, axis=1)  # per-layer squared norm

        return tr_sb

    # ── within-class scatter (streamed) ──────────────────────────────────
    @staticmethod
    def _within_class_scatter(
        domain_means: Dict,
        sample_counts: Dict,
        activations_dir: str,
        component: str,
        mode: str,
        batch_size: int = 500,
    ) -> np.ndarray:
        """
        Compute tr(S_W) per layer by streaming through activation files.

        tr(S_W)^ℓ = Σ_k Σ_{x ∈ D_k} ||x^ℓ - μ_k^ℓ||²

        Parameters
        ----------
        domain_means   : {domain: {comp: {mode: array(L, D)}}}
        sample_counts  : {domain: int}
        activations_dir: path to directory with {domain}_activations.h5 files

        Returns
        -------
        tr_sw : np.ndarray, shape (num_layers,)
        """
        domains = list(domain_means.keys())
        mu0 = domain_means[domains[0]][component][mode]
        num_layers = mu0.shape[0]

        tr_sw = np.zeros(num_layers, dtype=np.float64)

        for d in domains:
            mu_k = domain_means[d][component][mode].astype(np.float64)  # (L, D)
            act_path = os.path.join(activations_dir, f"{d}_activations.h5")

            with h5py.File(act_path, "r") as f:
                dataset = f[f"{component}/{mode}"]  # (N, L, D)
                N = dataset.shape[0]

                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    batch = dataset[start:end].astype(np.float64)  # (B, L, D)
                    diff = batch - mu_k[np.newaxis, :, :]  # (B, L, D)
                    # ||diff||² per sample per layer, then sum over batch & dim
                    tr_sw += np.sum(diff ** 2, axis=(0, 2))  # (L,)

                    if start % (batch_size * 10) == 0:
                        print(f"      {d}: {end}/{N}", end="\r", flush=True)

            print(f"      {d}: {N}/{N} ✓")

        return tr_sw

    # ── main compute ─────────────────────────────────────────────────────
    def compute_per_layer(
        self,
        domain_vectors: Dict,
        domain_means: Dict,
        sample_counts: Dict,
        activations_dir: str,
        component: str,
        mode: str,
        batch_size: int = 500,
    ) -> np.ndarray:
        """Compute FDR = tr(S_B) / tr(S_W) per layer."""

        tr_sb = self._between_class_scatter(
            domain_vectors, sample_counts, component, mode
        )
        tr_sw = self._within_class_scatter(
            domain_means, sample_counts, activations_dir, component, mode, batch_size
        )

        # Avoid division by zero (shouldn't happen with real data)
        fdr = np.where(tr_sw > 0, tr_sb / tr_sw, 0.0)
        return fdr, tr_sb, tr_sw


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_domain_data(vectors_file: str, activations_dir: str):
    """
    Load domain vectors, domain means, and sample counts from the
    pre-computed domain_vectors.h5 and raw activation files.

    Returns
    -------
    domain_vectors : {domain: {comp: {mode: ndarray(L, D)}}}
    domain_means   : {domain: {comp: {mode: ndarray(L, D)}}}
    sample_counts  : {domain: int}
    """
    components = ["attn", "mlp", "layer"]
    modes = ["mean", "last"]

    with h5py.File(vectors_file, "r") as f:
        domains = list(f.attrs["domains"])
        sample_counts = {d: int(f.attrs[f"num_samples_{d}"]) for d in domains}

        # Load domain vectors  (d_k = μ_k - μ_global)
        domain_vectors = {}
        for d in domains:
            domain_vectors[d] = {}
            for comp in components:
                domain_vectors[d][comp] = {}
                for mode in modes:
                    key = f"domain_vectors/{d}/{comp}/{mode}"
                    domain_vectors[d][comp][mode] = f[key][:].astype(np.float32)

        # Load global means
        global_means = {}
        for comp in components:
            global_means[comp] = {}
            for mode in modes:
                key = f"global_mean/{comp}/{mode}"
                global_means[comp][mode] = f[key][:].astype(np.float32)

    # Reconstruct domain means: μ_k = d_k + μ_global
    domain_means = {}
    for d in domains:
        domain_means[d] = {}
        for comp in components:
            domain_means[d][comp] = {}
            for mode in modes:
                domain_means[d][comp][mode] = (
                    domain_vectors[d][comp][mode] + global_means[comp][mode]
                )

    return domain_vectors, domain_means, sample_counts, domains


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_fdr_single_mode(
    results: Dict,
    mode: str,
    output_path: str,
    num_layers: int,
):
    """
    Plot FDR across layers for attn, mlp, and layer (total) for a single mode.

    Parameters
    ----------
    results : {component: {'fdr': array, 'tr_sb': array, 'tr_sw': array}}
    mode    : 'mean' or 'last'
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    layers = np.arange(num_layers)

    # Color palette
    colors = {
        "attn":  "#FF6B6B",   # coral red
        "mlp":   "#4ECDC4",   # teal
        "layer": "#6C5CE7",   # purple
    }
    labels = {
        "attn":  "Attention Only",
        "mlp":   "MLP Only",
        "layer": "Full Residual Stream",
    }

    for comp in ["attn", "mlp", "layer"]:
        fdr = results[comp]["fdr"]
        ax.plot(
            layers, fdr,
            color=colors[comp],
            linewidth=2.2,
            label=labels[comp],
            marker="o",
            markersize=4,
            alpha=0.9,
        )

    # Styling
    mode_title = "Mean-Pool" if mode == "mean" else "Last-Token"
    ax.set_title(
        f"Fisher Discriminant Ratio — {mode_title} Strategy",
        fontsize=16, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("FDR  (tr(S_B) / tr(S_W))", fontsize=13)
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=12, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=11)

    # Light background
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def compute_separability(
    metric: SeparabilityMetric,
    activations_dir: str,
    vectors_file: str,
    output_dir: str,
    batch_size: int = 500,
):
    """
    Compute a separability metric for all components and modes.

    Parameters
    ----------
    metric          : instance of SeparabilityMetric
    activations_dir : path to raw activation HDF5 files
    vectors_file    : path to domain_vectors.h5
    output_dir      : directory for output plots and data

    Returns
    -------
    all_results : {mode: {component: {'fdr': array, 'tr_sb': array, 'tr_sw': array}}}
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Computing: {metric.name}")
    print(f"  Vectors file   : {vectors_file}")
    print(f"  Activations dir: {activations_dir}")
    print()

    # Load data
    domain_vectors, domain_means, sample_counts, domains = load_domain_data(
        vectors_file, activations_dir
    )
    print(f"  Domains: {domains}")
    print(f"  Sample counts: {sample_counts}")

    num_layers = domain_vectors[domains[0]]["attn"]["mean"].shape[0]
    print(f"  Num layers: {num_layers}")
    print()

    components = ["attn", "mlp", "layer"]
    modes = ["mean", "last"]

    all_results = {}

    for mode in modes:
        print(f"  === Mode: {mode} ===")
        all_results[mode] = {}

        for comp in components:
            print(f"    Component: {comp}")
            fdr, tr_sb, tr_sw = metric.compute_per_layer(
                domain_vectors=domain_vectors,
                domain_means=domain_means,
                sample_counts=sample_counts,
                activations_dir=activations_dir,
                component=comp,
                mode=mode,
                batch_size=batch_size,
            )
            all_results[mode][comp] = {
                "fdr": fdr,
                "tr_sb": tr_sb,
                "tr_sw": tr_sw,
            }
            print(f"      FDR range: [{fdr.min():.6f}, {fdr.max():.6f}]")
            print()

    # ── Save numerical results ───────────────────────────────────────────
    save_dict = {}
    for mode in modes:
        for comp in components:
            for key in ["fdr", "tr_sb", "tr_sw"]:
                save_dict[f"{mode}_{comp}_{key}"] = all_results[mode][comp][key]
    save_dict["domains"] = np.array(domains)
    save_dict["sample_counts"] = np.array([sample_counts[d] for d in domains])

    npz_path = os.path.join(output_dir, "fdr_scores.npz")
    np.savez(npz_path, **save_dict)
    print(f"  Scores saved: {npz_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    for mode in modes:
        plot_path = os.path.join(output_dir, f"fdr_{mode}.png")
        plot_fdr_single_mode(all_results[mode], mode, plot_path, num_layers)

    return all_results


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute multi-class separability metrics across transformer layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--activations-dir", type=str, default="./activations",
        help="Directory containing {domain}_activations.h5 files.",
    )
    p.add_argument(
        "--vectors-file", type=str, default="./results/domain_vectors.h5",
        help="Path to pre-computed domain_vectors.h5.",
    )
    p.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory for output plots and data files.",
    )
    p.add_argument(
        "--batch-size", type=int, default=500,
        help="Batch size for streaming through activation files.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    metric = FisherDiscriminantRatio()
    compute_separability(
        metric=metric,
        activations_dir=args.activations_dir,
        vectors_file=args.vectors_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
