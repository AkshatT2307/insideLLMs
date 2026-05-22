#!/usr/bin/env python3
"""
domain_specificity.py — Experiment 2, Metric 2: Domain-Specificity of Weight Updates

Computes pairwise cosine similarity between vectorised weight updates from different
domains, separately for each module and grouped (attn, mlp).

From the paper (Section 7.3):
    sim(k, k')^ℓ = <vec(ΔW_ℓ^(k)), vec(ΔW_ℓ^(k'))> / (||vec(ΔW_ℓ^(k))|| · ||vec(ΔW_ℓ^(k'))||)

Produces 6×6 cross-domain update similarity matrices at each layer.

Prediction: MLP updates should be more domain-differentiated (lower cross-domain
cosine similarity) than attention updates.

Usage:
    python domain_specificity.py                                # all available domains, epoch_3
    python domain_specificity.py --domains cs eess math         # specific domains
    python domain_specificity.py --epoch 1                      # specific epoch
"""

import argparse
import os
import sys
import json
from datetime import datetime
from collections import defaultdict
from itertools import combinations

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_utils import (
    NUM_LAYERS,
    ATTN_MODULES,
    MLP_MODULES,
    ALL_MODULES,
    MODULE_GROUP,
    DEFAULT_DOMAINS,
    load_lora_deltas,
    save_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2 Metric 2: Domain Specificity")
    p.add_argument(
        "--adapter-dir", type=str, default="../adapters",
        help="Root directory containing domain adapter subdirectories.",
    )
    p.add_argument(
        "--domains", nargs="+", default=None,
        help="Domains to analyse. Default: auto-detect from adapter-dir.",
    )
    p.add_argument(
        "--epoch", type=int, default=3,
        help="Which epoch checkpoint to use (default: 3).",
    )
    p.add_argument(
        "--output-dir", type=str, default="../experiments/results",
        help="Directory to save result logs.",
    )
    return p.parse_args()


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D tensors."""
    dot = torch.dot(v1, v2)
    norm1 = v1.norm()
    norm2 = v2.norm()
    if norm1 == 0 or norm2 == 0:
        return 0.0
    sim = (dot / (norm1 * norm2)).item()
    return max(-1.0, min(1.0, sim))  # clamp for float precision


def compute_similarity_matrix(
    domain_vectors: dict,
    domains: list,
) -> dict:
    """
    Compute pairwise cosine similarity matrix for a set of domain vectors.
    
    Parameters
    ----------
    domain_vectors : dict
        domain → 1D tensor (vectorised ΔW)
    domains : list
        Ordered list of domain names
    
    Returns
    -------
    dict with 'matrix' (list of lists), 'mean_off_diagonal', 'pairwise' details
    """
    n = len(domains)
    matrix = [[0.0] * n for _ in range(n)]
    pairwise = {}
    off_diag_vals = []

    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if d1 not in domain_vectors or d2 not in domain_vectors:
                matrix[i][j] = None
                continue
            sim = cosine_similarity(domain_vectors[d1], domain_vectors[d2])
            matrix[i][j] = round(sim, 6)
            if i != j:
                off_diag_vals.append(sim)
            if i < j:
                pairwise[f"{d1}_vs_{d2}"] = round(sim, 6)

    return {
        "domains": domains,
        "matrix": matrix,
        "mean_off_diagonal": round(np.mean(off_diag_vals), 6) if off_diag_vals else None,
        "std_off_diagonal": round(np.std(off_diag_vals), 6) if off_diag_vals else None,
        "min_off_diagonal": round(min(off_diag_vals), 6) if off_diag_vals else None,
        "max_off_diagonal": round(max(off_diag_vals), 6) if off_diag_vals else None,
        "pairwise": pairwise,
    }


def compute_domain_specificity(
    adapter_dir: str,
    domains: list,
    epoch: int,
) -> dict:
    """
    Compute pairwise cosine similarity of weight updates across domains.
    
    For each layer, produces similarity matrices:
      - Per-module (q_proj, k_proj, ... individually)
      - Grouped (all attn modules concatenated, all mlp modules concatenated)
    """

    # ── Step 1: Load ΔW for all domains ───────────────────────────────────
    all_deltas = {}  # domain → { (layer, module): ΔW }

    for domain in domains:
        epoch_dir = os.path.join(adapter_dir, domain, f"epoch_{epoch}")
        if not os.path.exists(epoch_dir):
            print(f"  [SKIP] {domain}/epoch_{epoch} — not found")
            continue

        print(f"  Loading ΔW for domain: {domain} (epoch {epoch})")
        deltas = load_lora_deltas(epoch_dir)
        all_deltas[domain] = deltas
        print(f"    Loaded {len(deltas)} ΔW matrices")

    available_domains = [d for d in domains if d in all_deltas]
    if len(available_domains) < 2:
        print(f"  [ERROR] Need at least 2 domains for similarity analysis, got {len(available_domains)}")
        return {"error": "insufficient_domains", "available": available_domains}

    print(f"\n  Computing similarities across {len(available_domains)} domains: {available_domains}")

    # ── Step 2: Per-module similarity at each layer ───────────────────────
    per_module_results = {}  # layer → module → similarity_matrix

    for layer_idx in range(NUM_LAYERS):
        layer_results = {}

        for module in ALL_MODULES:
            # Vectorise ΔW for each domain
            domain_vectors = {}
            for domain in available_domains:
                key = (layer_idx, module)
                if key in all_deltas[domain]:
                    delta_w = all_deltas[domain][key]
                    domain_vectors[domain] = delta_w.flatten()

            if len(domain_vectors) >= 2:
                sim_result = compute_similarity_matrix(domain_vectors, available_domains)
                layer_results[module] = sim_result

        per_module_results[layer_idx] = layer_results

    # ── Step 3: Grouped similarity (concatenated attn / mlp) ──────────────
    grouped_results = {}  # layer → { 'attn': similarity_matrix, 'mlp': ... }

    for layer_idx in range(NUM_LAYERS):
        layer_grouped = {}

        for group_name, module_list in [("attn", ATTN_MODULES), ("mlp", MLP_MODULES)]:
            # Concatenate all module ΔWs within the group for each domain
            domain_vectors = {}
            for domain in available_domains:
                parts = []
                all_present = True
                for module in module_list:
                    key = (layer_idx, module)
                    if key in all_deltas[domain]:
                        parts.append(all_deltas[domain][key].flatten())
                    else:
                        all_present = False
                        break
                if all_present and parts:
                    domain_vectors[domain] = torch.cat(parts)

            if len(domain_vectors) >= 2:
                sim_result = compute_similarity_matrix(domain_vectors, available_domains)
                layer_grouped[group_name] = sim_result

        grouped_results[layer_idx] = layer_grouped

    # ── Step 4: Summary statistics across layers ──────────────────────────
    summary = {"attn": {}, "mlp": {}, "per_module": {}}

    for group_name in ["attn", "mlp"]:
        mean_off_diags = [
            grouped_results[l][group_name]["mean_off_diagonal"]
            for l in range(NUM_LAYERS)
            if group_name in grouped_results.get(l, {})
            and grouped_results[l][group_name]["mean_off_diagonal"] is not None
        ]
        if mean_off_diags:
            summary[group_name] = {
                "grand_mean_similarity": round(np.mean(mean_off_diags), 6),
                "std_across_layers": round(np.std(mean_off_diags), 6),
                "max_similarity_layer": int(np.argmax(mean_off_diags)),
                "min_similarity_layer": int(np.argmin(mean_off_diags)),
                "max_similarity": round(max(mean_off_diags), 6),
                "min_similarity": round(min(mean_off_diags), 6),
            }

    # Per-module summaries
    for module in ALL_MODULES:
        mean_off_diags = [
            per_module_results[l][module]["mean_off_diagonal"]
            for l in range(NUM_LAYERS)
            if module in per_module_results.get(l, {})
            and per_module_results[l][module]["mean_off_diagonal"] is not None
        ]
        if mean_off_diags:
            summary["per_module"][module] = {
                "grand_mean_similarity": round(np.mean(mean_off_diags), 6),
                "std_across_layers": round(np.std(mean_off_diags), 6),
                "group": MODULE_GROUP[module],
            }

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n  {'─' * 55}")
    print(f"  Summary: Mean Off-Diagonal Cosine Similarity (Averaged Across Layers)")
    print(f"  {'─' * 55}")

    if "grand_mean_similarity" in summary.get("attn", {}):
        print(f"  Attention (grouped): {summary['attn']['grand_mean_similarity']:.6f}")
    if "grand_mean_similarity" in summary.get("mlp", {}):
        print(f"  MLP (grouped):       {summary['mlp']['grand_mean_similarity']:.6f}")
    
    print(f"\n  Per-module breakdown:")
    for module in ALL_MODULES:
        if module in summary.get("per_module", {}):
            s = summary["per_module"][module]
            group_label = "attn" if s["group"] == "attn" else "mlp "
            print(f"    [{group_label}] {module:<12}: {s['grand_mean_similarity']:.6f}")

    if ("grand_mean_similarity" in summary.get("attn", {}) and
        "grand_mean_similarity" in summary.get("mlp", {})):
        attn_sim = summary["attn"]["grand_mean_similarity"]
        mlp_sim = summary["mlp"]["grand_mean_similarity"]
        print(f"\n  Prediction check (MLP should be LOWER than Attn):")
        print(f"    Attn mean sim: {attn_sim:.6f}")
        print(f"    MLP  mean sim: {mlp_sim:.6f}")
        if mlp_sim < attn_sim:
            print(f"    ✅ MLP updates are more domain-differentiated (lower similarity)")
        else:
            print(f"    ⚠️  MLP updates are NOT more differentiated than attention")

    return {
        "per_module": {int(k): v for k, v in per_module_results.items()},
        "grouped": {int(k): v for k, v in grouped_results.items()},
        "summary": summary,
        "available_domains": available_domains,
    }


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.normpath(os.path.join(script_dir, args.adapter_dir))
    output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))

    # Auto-detect domains if not specified
    if args.domains:
        domains = args.domains
    else:
        domains = [
            d for d in DEFAULT_DOMAINS
            if os.path.isdir(os.path.join(adapter_dir, d, f"epoch_{args.epoch}"))
        ]

    if len(domains) < 2:
        print(f"[ERROR] Need at least 2 domains. Found: {domains}")
        print(f"  Adapter dir: {adapter_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Experiment 2 — Metric 2: Domain Specificity of Updates")
    print(f"  Adapter dir: {adapter_dir}")
    print(f"  Domains: {domains}")
    print(f"  Epoch: {args.epoch}")
    print("=" * 60)

    results = compute_domain_specificity(
        adapter_dir=adapter_dir,
        domains=domains,
        epoch=args.epoch,
    )

    if "error" in results:
        print(f"\n[ERROR] {results['error']}")
        sys.exit(1)

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "experiment": "domain_specificity",
        "metric": "pairwise_cosine_similarity",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "adapter_dir": adapter_dir,
            "epoch": args.epoch,
            "domains": results["available_domains"],
        },
        "summary": results["summary"],
        "per_layer_grouped": results["grouped"],
        "per_layer_per_module": results["per_module"],
    }

    out_path = os.path.join(output_dir, f"domain_specificity_epoch{args.epoch}.json")
    save_json(output, out_path)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
