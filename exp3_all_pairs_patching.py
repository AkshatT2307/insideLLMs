#!/usr/bin/env python3
"""
exp3_all_pairs_patching.py — Causal patching for ALL directed domain pairs.

Efficiently processes all K*(K-1) directed pairs by caching clean forward
passes per source domain. For each source domain batch, runs one clean pass
then K-1 patched passes per patch layer.

Saves:
  - shift_scores_all_pairs.pt: (num_pairs, num_patch_layers, num_layers)
  - all_pairs_results.json: full summary with per-pair shift scores
"""

import argparse, json, os, time, itertools
from typing import Dict, List, Optional
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_domain_data, tokenize_domain, make_dataloader


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--vectors-dir", type=str, default="./results/exp3_causal_alldomains")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    p.add_argument("--data-dir", type=str, default="./arxiv_data")
    p.add_argument("--output-dir", type=str, default="./results/exp3_causal_alldomains/patching_all_pairs")
    p.add_argument("--patch-layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24])
    p.add_argument("--num-samples", type=int, default=50,
                    help="Samples per source domain (50 keeps runtime manageable)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pooling", type=str, default="mean", choices=["last", "mean"])
    p.add_argument("--domains", type=str, nargs="*",
                    default=["cs", "eess", "math", "physics", "q-bio", "stat"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Load domain vectors for all domains
# ─────────────────────────────────────────────────────────────────────────────
def load_all_domain_vectors(vectors_dir: str, domains: List[str]):
    data = {}
    for d in domains:
        d_dir = os.path.join(vectors_dir, d)
        data[d] = {
            "attn_unit": torch.load(os.path.join(d_dir, "domain_vector_unit_attn.pt"),
                                     weights_only=True),
            "layer_vec": torch.load(os.path.join(d_dir, "domain_vector_layer.pt"),
                                     weights_only=True),
            "alpha": torch.load(os.path.join(d_dir, "alpha_projection_magnitudes.pt"),
                                 weights_only=True),
        }
    data["global_mean_attn"] = torch.load(
        os.path.join(vectors_dir, "global_mean_attn.pt"), weights_only=True)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Hook manager with cached clean pass support
# ─────────────────────────────────────────────────────────────────────────────
class AllPairsHookManager:
    def __init__(self, model, num_layers, hidden_dim, pooling="mean"):
        self.model = model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self._handles = []
        self.attention_mask = None
        self.patch_layer = None
        self.d_A_unit = None
        self.d_B_unit = None
        self.alpha = None
        self.patching_enabled = False
        self.residual_streams = {}

    def _pool(self, act):
        if self.pooling == "last":
            return act[:, -1, :].float().cpu()
        mask = self.attention_mask.to(device=act.device, dtype=torch.float32).unsqueeze(-1)
        return ((act.float() * mask).sum(1) / mask.sum(1).clamp(min=1.0)).cpu()

    def set_mask(self, mask):
        self.attention_mask = mask

    def configure_patch(self, patch_layer, d_A_unit, d_B_unit, alpha):
        self.patch_layer = patch_layer
        self.d_A_unit = d_A_unit
        self.d_B_unit = d_B_unit
        self.alpha = alpha
        self.patching_enabled = True

    def disable_patch(self):
        self.patching_enabled = False

    def register_hooks(self):
        for i, layer in enumerate(self.model.model.layers):
            self._handles.append(
                layer.self_attn.register_forward_hook(self._make_attn_hook(i)))
            self._handles.append(
                layer.register_forward_hook(self._make_layer_hook(i)))
        print(f"  Registered {len(self._handles)} hooks")

    def _make_attn_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if not self.patching_enabled or layer_idx != self.patch_layer:
                return output
            act = output[0] if isinstance(output, tuple) else output
            device, dtype = act.device, act.dtype
            d_A = self.d_A_unit.to(device=device, dtype=torch.float32)
            d_B = self.d_B_unit.to(device=device, dtype=torch.float32)
            act_f = act.float()
            proj_A = (act_f * d_A).sum(-1, keepdim=True)
            patched = act_f - proj_A * d_A + self.alpha * d_B
            patched = patched.to(dtype)
            return (patched,) + output[1:] if isinstance(output, tuple) else patched
        return hook_fn

    def _make_layer_hook(self, layer_idx):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            self.residual_streams[layer_idx] = self._pool(act).to(torch.float64)
        return hook_fn

    def get_streams(self):
        result = dict(self.residual_streams)
        self.residual_streams.clear()
        return result

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def compute_shift_scores(clean, patched, d_B_layer, d_A_layer, num_layers):
    """Eq. 7 shift scores across all layers."""
    scores = torch.zeros(num_layers, dtype=torch.float64)
    for l in range(num_layers):
        if l not in clean or l not in patched:
            continue
        diff = patched[l] - clean[l]
        direction = d_B_layer[l] - d_A_layer[l]
        dnorm = direction.norm().clamp(min=1e-8)
        scores[l] = (diff * (direction / dnorm)).sum(1).mean()
    return scores


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    domains = args.domains
    pairs = [(s, t) for s in domains for t in domains if s != t]
    print(f"\n{'='*60}")
    print(f"All-Pairs Causal Patching: {len(pairs)} directed pairs")
    print(f"Domains: {domains}")
    print(f"Patch layers: {args.patch_layers}")
    print(f"{'='*60}")

    # Load domain vectors
    dv = load_all_domain_vectors(args.vectors_dir, domains)
    print("  Domain vectors loaded")

    # Load model
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype],
        device_map=args.device, trust_remote_code=True)
    model.eval()
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Model loaded: {num_layers} layers, d={hidden_dim}")

    hook_mgr = AllPairsHookManager(model, num_layers, hidden_dim, args.pooling)
    hook_mgr.register_hooks()

    # Load all domain data
    print("  Loading data...")
    all_data = load_domain_data(args.data_dir, args.num_samples, args.seed)
    all_data = {k: v for k, v in all_data.items() if k in domains}

    # Results storage
    all_results = {}  # (source, target) -> {patch_layer: shift_scores}

    try:
        target_device = model.model.embed_tokens.weight.device
    except AttributeError:
        target_device = next(model.parameters()).device

    t_global = time.time()

    # Process by source domain (cache clean passes)
    for src_idx, source in enumerate(domains):
        print(f"\n{'━'*60}")
        print(f"Source domain: {source} ({src_idx+1}/{len(domains)})")
        print(f"{'━'*60}")

        dataset = tokenize_domain(all_data[source], tokenizer, max_length=args.max_length)
        loader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        targets = [d for d in domains if d != source]

        # Initialize accumulators for all (target, patch_layer) combos
        accum = {}
        for target in targets:
            accum[target] = {}
            for pl in args.patch_layers:
                accum[target][pl] = torch.zeros(num_layers, dtype=torch.float64)

        total_samples = 0

        for batch in tqdm(loader, desc=f"  [{source}]", unit="batch", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(target_device)
            attention_mask = batch["attention_mask"].to(target_device)
            bs = input_ids.size(0)
            hook_mgr.set_mask(attention_mask)

            # ── Clean forward pass (shared for all targets/patch_layers) ──
            hook_mgr.disable_patch()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False, output_attentions=False, output_hidden_states=False)
            clean_streams = hook_mgr.get_streams()

            # ── Patched passes for each (target, patch_layer) ──
            for target in targets:
                for pl in args.patch_layers:
                    hook_mgr.configure_patch(
                        patch_layer=pl,
                        d_A_unit=dv[source]["attn_unit"][pl],
                        d_B_unit=dv[target]["attn_unit"][pl],
                        alpha=dv[target]["alpha"][pl].item(),
                    )
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False, output_attentions=False,
                              output_hidden_states=False)
                    patched_streams = hook_mgr.get_streams()
                    hook_mgr.disable_patch()

                    shifts = compute_shift_scores(
                        clean_streams, patched_streams,
                        dv[target]["layer_vec"], dv[source]["layer_vec"],
                        num_layers)
                    accum[target][pl] += shifts * bs

            total_samples += bs
            del input_ids, attention_mask, clean_streams
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Finalize means
        for target in targets:
            pair_key = f"{source}->{target}"
            all_results[pair_key] = {}
            for pl in args.patch_layers:
                all_results[pair_key][pl] = (accum[target][pl] / total_samples).clone()

        print(f"  ✓ {source}: {total_samples} samples × {len(targets)} targets × {len(args.patch_layers)} patches")

    elapsed = time.time() - t_global
    print(f"\n  Total patching time: {elapsed:.1f}s")

    # ── Save results ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving results...")

    pair_keys = sorted(all_results.keys())
    patch_layers = sorted(args.patch_layers)
    n_pairs = len(pair_keys)
    n_pl = len(patch_layers)

    # Build 3D tensor: (pairs, patch_layers, num_layers)
    shift_tensor = torch.zeros(n_pairs, n_pl, num_layers, dtype=torch.float64)
    for i, pk in enumerate(pair_keys):
        for j, pl in enumerate(patch_layers):
            shift_tensor[i, j] = all_results[pk][pl]

    torch.save(shift_tensor, os.path.join(args.output_dir, "shift_scores_all_pairs.pt"))

    # Build JSON summary
    summary = {
        "experiment": "causal_patching_all_pairs",
        "model": args.model,
        "domains": domains,
        "num_pairs": n_pairs,
        "pair_keys": pair_keys,
        "patch_layers": patch_layers,
        "num_samples_per_source": args.num_samples,
        "pooling": args.pooling,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "tensor_shape": list(shift_tensor.shape),
        "per_pair": {},
    }

    for i, pk in enumerate(pair_keys):
        pair_data = {}
        for j, pl in enumerate(patch_layers):
            scores = shift_tensor[i, j].tolist()
            post = shift_tensor[i, j, pl+1:].tolist()
            pair_data[str(pl)] = {
                "shift_scores": scores,
                "post_patch_mean": float(np.mean(post)) if post else 0.0,
                "post_patch_max": float(np.max(post)) if post else 0.0,
            }
        summary["per_pair"][pk] = pair_data

    with open(os.path.join(args.output_dir, "all_pairs_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ shift_scores_all_pairs.pt: {shift_tensor.shape}")
    print(f"  ✓ all_pairs_results.json")
    print(f"  Saved to: {args.output_dir}")

    hook_mgr.remove_hooks()
    print(f"\n{'='*60}")
    print("All-pairs patching complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
