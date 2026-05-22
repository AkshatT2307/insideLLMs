#!/usr/bin/env python3
"""
exp3_compute_domain_vectors.py — Part 1 of Experiment 3: Causal Activation Patching

Computes domain concept vectors for CS and q-bio using 500 prompts from each domain,
running them through Qwen 2.5 7B.

For each layer ℓ, we compute:
  - Global mean:  µ^ℓ = E_x[x^ℓ(x)]
  - Domain concept vector:  d^ℓ_k = E_{x~D_k}[x^ℓ(x)] - µ^ℓ       (Eq. 2)
  - Attention contribution vector:  d^{attn,ℓ}_k                     (Eq. 3)
  - MLP contribution vector:  d^{mlp,ℓ}_k                            (Eq. 4)

All vectors are saved as PyTorch tensors (.pt) and metadata as JSON for
downstream use in causal activation patching (Part 2).

Usage
-----
    python exp3_compute_domain_vectors.py \
        --model Qwen/Qwen2.5-7B \
        --data-dir ./arxiv_data \
        --output-dir ./results/exp3_causal \
        --samples-per-domain 500 \
        --batch-size 16 \
        --max-length 384
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import load_domain_data, tokenize_domain, make_dataloader
from hooks import ActivationStore


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute domain concept vectors for Experiment 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    p.add_argument("--data-dir", type=str, default="./arxiv_data")
    p.add_argument("--output-dir", type=str, default="./results/exp3_causal")
    p.add_argument("--samples-per-domain", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--domains", type=str, nargs="*", default=["cs", "q-bio"],
                    help="Domains to compute vectors for (default: cs q-bio)")
    p.add_argument("--pooling", type=str, default="mean",
                    choices=["last", "mean"],
                    help="Pooling mode for activations (mean recommended for concept vectors)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_name: str, dtype_str: str, device_map: str):
    """Load model in eval mode and configure tokenizer for left-padding."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_str]

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"  dtype  : {dtype_str}")
    print(f"  device : {device_map}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    print(f"  Model loaded — {num_layers} layers, d={hidden_dim}\n")

    return model, tokenizer, num_layers, hidden_dim


# ─────────────────────────────────────────────────────────────────────────────
# Accumulate activations for a domain
# ─────────────────────────────────────────────────────────────────────────────
def accumulate_domain_activations(
    domain: str,
    df: pd.DataFrame,
    model,
    tokenizer,
    store: ActivationStore,
    args: argparse.Namespace,
    num_layers: int,
    hidden_dim: int,
) -> Dict[str, torch.Tensor]:
    """
    Run forward passes over a domain's data and accumulate mean activations.

    Returns
    -------
    dict with keys:
        'layer_mean'  : (num_layers, hidden_dim) — mean residual stream
        'attn_mean'   : (num_layers, hidden_dim) — mean attention contribution (Δa)
        'mlp_mean'    : (num_layers, hidden_dim) — mean MLP contribution (Δm)
    All in float64 for numerical stability.
    """
    pooling = args.pooling
    print(f"\n{'─'*60}")
    print(f"Accumulating activations for domain: {domain}  ({len(df):,} samples)")
    print(f"  Pooling mode: {pooling}")
    print(f"{'─'*60}")

    # Tokenize
    t0 = time.time()
    dataset = tokenize_domain(df, tokenizer, max_length=args.max_length)
    print(f"  Tokenized in {time.time() - t0:.1f}s")

    loader = make_dataloader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Running sums for online mean computation
    layer_sum = torch.zeros(num_layers, hidden_dim, dtype=torch.float64)
    attn_sum = torch.zeros(num_layers, hidden_dim, dtype=torch.float64)
    mlp_sum = torch.zeros(num_layers, hidden_dim, dtype=torch.float64)
    total_samples = 0

    pbar = tqdm(loader, desc=f"  [{domain}]", unit="batch", dynamic_ncols=True)

    for batch in pbar:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bs = input_ids.size(0)

        # Move to model device
        try:
            target_device = model.model.embed_tokens.weight.device
        except AttributeError:
            target_device = next(model.parameters()).device

        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device)

        store.set_mask(attention_mask)

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )

        results = store.get_batch_results()

        # results[key] has shape (B, num_layers, hidden_dim) in float16 on CPU
        # Sum over batch dimension for online mean
        layer_batch = results[f"layer/{pooling}"].to(torch.float64)  # (B, L, d)
        attn_batch = results[f"attn/{pooling}"].to(torch.float64)
        mlp_batch = results[f"mlp/{pooling}"].to(torch.float64)

        layer_sum += layer_batch.sum(dim=0)  # (L, d)
        attn_sum += attn_batch.sum(dim=0)
        mlp_sum += mlp_batch.sum(dim=0)
        total_samples += bs

        del input_ids, attention_mask, results, layer_batch, attn_batch, mlp_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.set_postfix({"processed": f"{total_samples:,}/{len(dataset):,}"})

    print(f"  ✓ Domain '{domain}' — {total_samples:,} samples accumulated")

    return {
        "layer_mean": layer_sum / total_samples,  # (L, d)
        "attn_mean": attn_sum / total_samples,
        "mlp_mean": mlp_sum / total_samples,
        "num_samples": total_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compute concept vectors (Equations 2-4)
# ─────────────────────────────────────────────────────────────────────────────
def compute_concept_vectors(
    domain_means: Dict[str, Dict[str, torch.Tensor]],
) -> Dict:
    """
    Compute domain concept vectors by subtracting the global mean.

    Parameters
    ----------
    domain_means : dict
        {domain_name: {'layer_mean': (L,d), 'attn_mean': (L,d),
                       'mlp_mean': (L,d), 'num_samples': int}}

    Returns
    -------
    dict with:
        'global_mean_layer' : (L, d) — µ^ℓ for residual stream
        'global_mean_attn'  : (L, d) — E[Δa^ℓ] global
        'global_mean_mlp'   : (L, d) — E[Δm^ℓ] global
        'domain_vectors'    : {domain: {'layer': (L,d), 'attn': (L,d), 'mlp': (L,d)}}
        'domain_vectors_unit' : same but unit-normalised per layer
    """
    domains = list(domain_means.keys())
    total_samples = sum(domain_means[d]["num_samples"] for d in domains)

    # Weighted global mean (Eq. 2 denominator)
    components = ["layer", "attn", "mlp"]
    global_means = {}
    for comp in components:
        key = f"{comp}_mean"
        global_means[comp] = sum(
            domain_means[d][key] * (domain_means[d]["num_samples"] / total_samples)
            for d in domains
        )

    # Domain concept vectors (Eq. 2, 3, 4)
    domain_vectors = {}
    domain_vectors_unit = {}
    for d in domains:
        dv = {}
        dv_unit = {}
        for comp in components:
            key = f"{comp}_mean"
            vec = domain_means[d][key] - global_means[comp]  # (L, d)
            dv[comp] = vec

            # Unit normalise per layer: ˆd^ℓ_k = d^ℓ_k / ‖d^ℓ_k‖
            norms = vec.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (L, 1)
            dv_unit[comp] = vec / norms

        domain_vectors[d] = dv
        domain_vectors_unit[d] = dv_unit

    return {
        "global_mean_layer": global_means["layer"],
        "global_mean_attn": global_means["attn"],
        "global_mean_mlp": global_means["mlp"],
        "domain_vectors": domain_vectors,
        "domain_vectors_unit": domain_vectors_unit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compute projection magnitudes for calibration (α in Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────
def compute_projection_stats(
    domain: str,
    df: pd.DataFrame,
    model,
    tokenizer,
    store: ActivationStore,
    args: argparse.Namespace,
    num_layers: int,
    hidden_dim: int,
    global_mean_attn: torch.Tensor,
    domain_unit_attn: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the typical projection magnitude of domain-k attention contributions
    onto their own concept direction, per layer.

    This is α in Equation 6 — the calibration magnitude so the injected
    domain-B signal is "natural" in scale.

    Returns
    -------
    alpha : (num_layers,) — mean |⟨Δa^ℓ(x) - E[Δa^ℓ], ˆd^{attn,ℓ}_k⟩| for x ~ D_k
    """
    pooling = args.pooling
    print(f"\n  Computing projection magnitudes (α) for domain '{domain}'...")

    dataset = tokenize_domain(df, tokenizer, max_length=args.max_length)
    loader = make_dataloader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    proj_sum = torch.zeros(num_layers, dtype=torch.float64)
    total_samples = 0

    for batch in tqdm(loader, desc=f"  [α-{domain}]", unit="batch", dynamic_ncols=True):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bs = input_ids.size(0)

        try:
            target_device = model.model.embed_tokens.weight.device
        except AttributeError:
            target_device = next(model.parameters()).device

        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device)
        store.set_mask(attention_mask)

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )

        results = store.get_batch_results()
        attn_batch = results[f"attn/{pooling}"].to(torch.float64)  # (B, L, d)

        # Centre: subtract global mean of attention contributions
        centred = attn_batch - global_mean_attn.unsqueeze(0)  # (B, L, d)

        # Project onto unit domain vector: ⟨centred, ˆd⟩ for each layer
        # domain_unit_attn is (L, d), centred is (B, L, d)
        projections = (centred * domain_unit_attn.unsqueeze(0)).sum(dim=2)  # (B, L)

        # We want the typical magnitude (use mean of absolute projections)
        proj_sum += projections.abs().sum(dim=0)  # (L,)
        total_samples += bs

        del input_ids, attention_mask, results, attn_batch, centred, projections
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    alpha = proj_sum / total_samples  # (L,)
    print(f"  ✓ α computed for domain '{domain}' — shape {alpha.shape}")
    return alpha


# ─────────────────────────────────────────────────────────────────────────────
# Save everything
# ─────────────────────────────────────────────────────────────────────────────
def save_results(output_dir: str, concept_data: Dict, domain_means: Dict,
                 alpha_values: Dict, args: argparse.Namespace):
    """Save all domain vectors and metadata as .pt tensors and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    domains = list(domain_means.keys())

    # ── Save global means ────────────────────────────────────────────────
    torch.save(concept_data["global_mean_layer"],
               os.path.join(output_dir, "global_mean_layer.pt"))
    torch.save(concept_data["global_mean_attn"],
               os.path.join(output_dir, "global_mean_attn.pt"))
    torch.save(concept_data["global_mean_mlp"],
               os.path.join(output_dir, "global_mean_mlp.pt"))

    # ── Save per-domain vectors ──────────────────────────────────────────
    for d in domains:
        domain_dir = os.path.join(output_dir, d)
        os.makedirs(domain_dir, exist_ok=True)

        # Raw domain vectors (un-normalised)
        for comp in ["layer", "attn", "mlp"]:
            torch.save(
                concept_data["domain_vectors"][d][comp],
                os.path.join(domain_dir, f"domain_vector_{comp}.pt"),
            )
            torch.save(
                concept_data["domain_vectors_unit"][d][comp],
                os.path.join(domain_dir, f"domain_vector_unit_{comp}.pt"),
            )

        # Domain means
        for comp in ["layer", "attn", "mlp"]:
            torch.save(
                domain_means[d][f"{comp}_mean"],
                os.path.join(domain_dir, f"domain_mean_{comp}.pt"),
            )

        # Alpha values (projection magnitudes for calibration)
        if d in alpha_values:
            torch.save(
                alpha_values[d],
                os.path.join(domain_dir, "alpha_projection_magnitudes.pt"),
            )

    # ── Save metadata ────────────────────────────────────────────────────
    metadata = {
        "model": args.model,
        "domains": domains,
        "samples_per_domain": {d: int(domain_means[d]["num_samples"]) for d in domains},
        "max_length": args.max_length,
        "pooling_mode": args.pooling,
        "seed": args.seed,
        "dtype": args.dtype,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add vector norms for diagnostic purposes
    norms_info = {}
    for d in domains:
        norms_info[d] = {}
        for comp in ["layer", "attn", "mlp"]:
            vec = concept_data["domain_vectors"][d][comp]
            layer_norms = vec.norm(dim=1).tolist()  # per-layer norms
            norms_info[d][comp] = {
                "mean_norm": float(np.mean(layer_norms)),
                "max_norm": float(np.max(layer_norms)),
                "min_norm": float(np.min(layer_norms)),
                "per_layer_norms": [round(n, 6) for n in layer_norms],
            }
        if d in alpha_values:
            norms_info[d]["alpha"] = {
                "per_layer": alpha_values[d].tolist(),
                "mean": float(alpha_values[d].mean()),
            }

    metadata["domain_vector_norms"] = norms_info

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  All results saved to: {output_dir}")
    print(f"  Files per domain: domain_vector_*.pt, domain_mean_*.pt, alpha_*.pt")
    print(f"  Global: global_mean_*.pt, metadata.json")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Load model ───────────────────────────────────────────────────────
    model, tokenizer, num_layers, hidden_dim = load_model_and_tokenizer(
        args.model, args.dtype, args.device,
    )

    # ── Register hooks ───────────────────────────────────────────────────
    store = ActivationStore(num_layers=num_layers)
    store.register_hooks(model)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading dataset …")
    domain_data = load_domain_data(
        data_dir=args.data_dir,
        samples_per_domain=args.samples_per_domain,
        seed=args.seed,
    )

    # Filter to requested domains
    domain_data = {k: v for k, v in domain_data.items() if k in args.domains}
    if not domain_data:
        print(f"[ERROR] No matching domains for: {args.domains}")
        return

    print(f"\nDomains to process: {list(domain_data.keys())}")
    print(f"Samples per domain: {args.samples_per_domain}")

    # ── Phase 1: Accumulate per-domain means ─────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1: Computing per-domain activation means")
    print(f"{'='*60}")

    domain_means = {}
    t_start = time.time()

    for domain, df in domain_data.items():
        domain_means[domain] = accumulate_domain_activations(
            domain=domain,
            df=df,
            model=model,
            tokenizer=tokenizer,
            store=store,
            args=args,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )

    elapsed = time.time() - t_start
    print(f"\n  Phase 1 complete in {elapsed:.1f}s")

    # ── Phase 2: Compute concept vectors ─────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2: Computing domain concept vectors (Eq. 2-4)")
    print(f"{'='*60}")

    concept_data = compute_concept_vectors(domain_means)

    # Print diagnostic info
    for d in domain_data.keys():
        print(f"\n  Domain '{d}':")
        for comp in ["layer", "attn", "mlp"]:
            vec = concept_data["domain_vectors"][d][comp]
            norms = vec.norm(dim=1)
            print(f"    {comp:>5s} concept vector — "
                  f"mean ‖d‖={norms.mean():.4f}, "
                  f"max ‖d‖={norms.max():.4f}, "
                  f"min ‖d‖={norms.min():.4f}")

    # ── Phase 3: Compute projection magnitudes (α) ───────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3: Computing projection magnitudes (α for Eq. 6)")
    print(f"{'='*60}")

    alpha_values = {}
    for domain, df in domain_data.items():
        alpha_values[domain] = compute_projection_stats(
            domain=domain,
            df=df,
            model=model,
            tokenizer=tokenizer,
            store=store,
            args=args,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            global_mean_attn=concept_data["global_mean_attn"],
            domain_unit_attn=concept_data["domain_vectors_unit"][domain]["attn"],
        )

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    save_results(args.output_dir, concept_data, domain_means, alpha_values, args)

    store.remove_hooks()

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
