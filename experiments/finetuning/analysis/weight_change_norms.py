#!/usr/bin/env python3
"""
weight_change_norms.py — Experiment 2, Metric 1: Relative Weight Change Norms

Computes δ = ||ΔW||_F / ||W_base||_F for each LoRA-adapted module at every layer,
for each domain. Reports per-module (q_proj, k_proj, ...) and grouped (attn, mlp).

From the paper (Section 7.2):
    δ_mlp^{(k),ℓ} = ||ΔW_mlp^{(k),ℓ}||_F / ||W_mlp,base^ℓ||_F
    δ_attn^{(k),ℓ} = ||ΔW_attn^{(k),ℓ}||_F / ||W_attn,base^ℓ||_F

Usage:
    python weight_change_norms.py                               # all available domains, epoch_3
    python weight_change_norms.py --domains cs eess              # specific domains
    python weight_change_norms.py --epoch 1                      # specific epoch
    python weight_change_norms.py --adapter-dir ../adapters      # custom adapter location
"""

import argparse
import os
import sys
import math
import json
from datetime import datetime
from collections import defaultdict

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
    load_base_weight_norms,
    group_by_layer_and_type,
    save_json,
    set_num_layers,
)

# Model slug utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.model_loader import get_model_slug


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2 Metric 1: Weight Change Norms")
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
        "--model-name", type=str, default="Qwen/Qwen2.5-7B",
        help="Base model name (for locating weights in HF cache).",
    )
    p.add_argument(
        "--output-dir", type=str, default="../experiments/results",
        help="Directory to save result logs.",
    )
    return p.parse_args()


def compute_weight_change_norms(
    adapter_dir: str,
    domains: list,
    epoch: int,
    model_name: str,
) -> dict:
    """
    Compute relative weight change norms for all domains, all layers, all modules.
    
    Returns a structured dict with per-module and grouped results.
    """
    
    # ── Step 1: Load base model weight norms ──────────────────────────────
    print("\n  Loading base model weight norms...")
    base_norms = load_base_weight_norms(model_name)
    print(f"  Loaded norms for {len(base_norms)} base weight matrices.")

    # ── Step 2: For each domain, load LoRA deltas and compute δ ───────────
    results = {}

    for domain in domains:
        epoch_dir = os.path.join(adapter_dir, domain, f"epoch_{epoch}")
        if not os.path.exists(epoch_dir):
            print(f"  [SKIP] {domain}/epoch_{epoch} — not found")
            continue

        print(f"\n  Processing domain: {domain} (epoch {epoch})")

        # Load ΔW = (alpha/r) * B @ A
        deltas = load_lora_deltas(epoch_dir)
        print(f"    Loaded {len(deltas)} ΔW matrices")

        # Compute per-module relative norms
        domain_result = {
            "domain": domain,
            "epoch": epoch,
            "per_module": {},    # { layer: { module: { delta_norm, base_norm, relative_norm } } }
            "grouped": {},       # { layer: { attn: { delta_norm, base_norm, relative_norm }, mlp: ... } }
            "summary": {},       # aggregate stats
        }

        # Per-module results
        per_module_by_layer = defaultdict(dict)
        for (layer_idx, module), delta_w in deltas.items():
            delta_norm = delta_w.norm().item()
            base_norm = base_norms.get((layer_idx, module), None)
            
            if base_norm is None or base_norm == 0:
                relative_norm = float("inf")
            else:
                relative_norm = delta_norm / base_norm

            per_module_by_layer[layer_idx][module] = {
                "delta_norm_F": round(delta_norm, 6),
                "base_norm_F": round(base_norm, 6) if base_norm else None,
                "relative_norm": round(relative_norm, 8),
            }

        domain_result["per_module"] = {
            int(k): v for k, v in sorted(per_module_by_layer.items())
        }

        # Grouped results: combine attn modules and mlp modules per layer
        grouped_by_layer = {}
        for layer_idx in range(NUM_LAYERS):
            layer_data = per_module_by_layer.get(layer_idx, {})
            
            grouped = {"attn": {}, "mlp": {}}
            for group_name, module_list in [("attn", ATTN_MODULES), ("mlp", MLP_MODULES)]:
                # Combined Frobenius norm: sqrt(sum of squared individual norms)
                delta_sq_sum = 0.0
                base_sq_sum = 0.0
                module_details = {}

                for module in module_list:
                    if module in layer_data:
                        d = layer_data[module]
                        delta_sq_sum += d["delta_norm_F"] ** 2
                        if d["base_norm_F"] is not None:
                            base_sq_sum += d["base_norm_F"] ** 2
                        module_details[module] = d["relative_norm"]

                combined_delta = math.sqrt(delta_sq_sum)
                combined_base = math.sqrt(base_sq_sum) if base_sq_sum > 0 else 0
                combined_relative = (combined_delta / combined_base) if combined_base > 0 else float("inf")

                grouped[group_name] = {
                    "combined_delta_norm_F": round(combined_delta, 6),
                    "combined_base_norm_F": round(combined_base, 6),
                    "combined_relative_norm": round(combined_relative, 8),
                    "per_module": module_details,
                }

            grouped_by_layer[layer_idx] = grouped

        domain_result["grouped"] = {int(k): v for k, v in sorted(grouped_by_layer.items())}

        # Summary statistics across layers
        attn_relatives = [
            grouped_by_layer[l]["attn"]["combined_relative_norm"]
            for l in range(NUM_LAYERS)
            if l in grouped_by_layer
        ]
        mlp_relatives = [
            grouped_by_layer[l]["mlp"]["combined_relative_norm"]
            for l in range(NUM_LAYERS)
            if l in grouped_by_layer
        ]

        domain_result["summary"] = {
            "attn": {
                "mean_relative_norm": round(np.mean(attn_relatives), 8) if attn_relatives else None,
                "std_relative_norm": round(np.std(attn_relatives), 8) if attn_relatives else None,
                "max_relative_norm": round(max(attn_relatives), 8) if attn_relatives else None,
                "min_relative_norm": round(min(attn_relatives), 8) if attn_relatives else None,
                "max_layer": int(np.argmax(attn_relatives)) if attn_relatives else None,
            },
            "mlp": {
                "mean_relative_norm": round(np.mean(mlp_relatives), 8) if mlp_relatives else None,
                "std_relative_norm": round(np.std(mlp_relatives), 8) if mlp_relatives else None,
                "max_relative_norm": round(max(mlp_relatives), 8) if mlp_relatives else None,
                "min_relative_norm": round(min(mlp_relatives), 8) if mlp_relatives else None,
                "max_layer": int(np.argmax(mlp_relatives)) if mlp_relatives else None,
            },
            "mlp_to_attn_ratio": round(
                np.mean(mlp_relatives) / np.mean(attn_relatives), 4
            ) if (attn_relatives and mlp_relatives and np.mean(attn_relatives) > 0) else None,
        }

        results[domain] = domain_result

        # Print summary
        s = domain_result["summary"]
        print(f"    Summary:")
        print(f"      δ_attn (mean): {s['attn']['mean_relative_norm']:.6f}")
        print(f"      δ_mlp  (mean): {s['mlp']['mean_relative_norm']:.6f}")
        print(f"      MLP/Attn ratio: {s['mlp_to_attn_ratio']:.2f}x")

    return results


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    finetuning_dir = os.path.dirname(script_dir)  # experiments/finetuning/
    model_slug = get_model_slug(args.model_name)
    results_base = os.path.join(finetuning_dir, "results", model_slug)

    adapter_dir = os.path.normpath(os.path.join(results_base, "adapters"))
    output_dir = os.path.normpath(os.path.join(results_base, "analysis_results"))

    # Auto-detect domains if not specified
    if args.domains:
        domains = args.domains
    else:
        domains = [
            d for d in DEFAULT_DOMAINS
            if os.path.isdir(os.path.join(adapter_dir, d, f"epoch_{args.epoch}"))
        ]

    if not domains:
        print(f"[ERROR] No adapter directories found in {adapter_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Experiment 2 — Metric 1: Relative Weight Change Norms")
    print(f"  Adapter dir: {adapter_dir}")
    print(f"  Domains: {domains}")
    print(f"  Epoch: {args.epoch}")
    print(f"  Base model: {args.model_name}")
    print("=" * 60)

    results = compute_weight_change_norms(
        adapter_dir=adapter_dir,
        domains=domains,
        epoch=args.epoch,
        model_name=args.model_name,
    )

    # ── Cross-domain comparison ───────────────────────────────────────────
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("Cross-Domain Comparison")
        print("=" * 60)
        print(f"  {'Domain':<12} {'δ_attn (mean)':>15} {'δ_mlp (mean)':>15} {'MLP/Attn':>10}")
        print("  " + "─" * 55)
        for domain, r in sorted(results.items()):
            s = r["summary"]
            print(
                f"  {domain:<12} "
                f"{s['attn']['mean_relative_norm']:>15.6f} "
                f"{s['mlp']['mean_relative_norm']:>15.6f} "
                f"{s['mlp_to_attn_ratio']:>10.2f}x"
            )

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "experiment": "weight_change_norms",
        "metric": "relative_frobenius_norm",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "adapter_dir": adapter_dir,
            "epoch": args.epoch,
            "base_model": args.model_name,
            "domains": domains,
        },
        "results": results,
    }

    out_path = os.path.join(output_dir, f"weight_change_norms_epoch{args.epoch}.json")
    save_json(output, out_path)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
