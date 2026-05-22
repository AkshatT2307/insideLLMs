#!/usr/bin/env python3
"""
exp3_causal_patching.py — Part 2 of Experiment 3: Causal Activation Patching

Performs concept direction patching (Eq. 6) and propagation analysis (Eq. 7).

Given pre-computed domain vectors from Part 1, this script:
1. Loads source texts from domain A (e.g. CS)
2. At a chosen hotspot layer ℓ*, patches the attention contribution by:
   - Removing the domain-A component along ˆd^{attn,ℓ*}_A
   - Injecting a domain-B component of calibrated magnitude α
3. Continues the forward pass normally for layers after ℓ*
4. Tracks the domain shift score s^ℓ at every layer ℓ > ℓ* (Eq. 7)
5. Also measures the change in next-token logit distribution

Usage
-----
    python exp3_causal_patching.py \
        --vectors-dir ./results/exp3_causal \
        --model Qwen/Qwen2.5-7B \
        --data-dir ./arxiv_data \
        --output-dir ./results/exp3_causal/patching \
        --source-domain cs --target-domain q-bio \
        --patch-layers 4 8 12 16 20 24 \
        --num-samples 100 --batch-size 8
"""

import argparse, json, os, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys as _sys
_sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
try:
    from utils.data_utils import load_domain_data, tokenize_domain, make_dataloader
    from utils.hooks import ActivationStore
except ImportError:
    pass
from data_utils import load_domain_data, tokenize_domain, make_dataloader


def parse_args():
    p = argparse.ArgumentParser(
        description="Causal activation patching (Experiment 3, Part 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vectors-dir", type=str, default="./results/exp3_causal",
                    help="Directory with domain vectors from Part 1")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    p.add_argument("--data-dir", type=str, default="./arxiv_data")
    p.add_argument("--output-dir", type=str, default="./results/exp3_causal/patching")
    p.add_argument("--source-domain", type=str, default="cs")
    p.add_argument("--target-domain", type=str, default="q-bio")
    p.add_argument("--patch-layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24],
                    help="Layers at which to perform the patch")
    p.add_argument("--num-samples", type=int, default=100,
                    help="Number of source-domain texts to patch")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pooling", type=str, default="mean", choices=["last", "mean"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Load pre-computed domain vectors
# ─────────────────────────────────────────────────────────────────────────────
def load_domain_vectors(vectors_dir: str, source: str, target: str):
    """Load domain vectors, unit vectors, and alpha values from Part 1."""
    data = {}
    for domain in [source, target]:
        d_dir = os.path.join(vectors_dir, domain)
        data[domain] = {
            "attn_unit": torch.load(os.path.join(d_dir, "domain_vector_unit_attn.pt"),
                                     weights_only=True),
            "layer_vec": torch.load(os.path.join(d_dir, "domain_vector_layer.pt"),
                                     weights_only=True),
            "attn_vec": torch.load(os.path.join(d_dir, "domain_vector_attn.pt"),
                                    weights_only=True),
            "alpha": torch.load(os.path.join(d_dir, "alpha_projection_magnitudes.pt"),
                                 weights_only=True),
        }
    # Global means
    data["global_mean_layer"] = torch.load(
        os.path.join(vectors_dir, "global_mean_layer.pt"), weights_only=True)
    data["global_mean_attn"] = torch.load(
        os.path.join(vectors_dir, "global_mean_attn.pt"), weights_only=True)

    with open(os.path.join(vectors_dir, "metadata.json")) as f:
        data["metadata"] = json.load(f)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Hook-based patching mechanism
# ─────────────────────────────────────────────────────────────────────────────
class PatchingHookManager:
    """
    Manages hooks for:
      1. Patching attention output at layer ℓ* (Eq. 6)
      2. Capturing residual stream at every layer for propagation analysis
    """

    def __init__(self, model, num_layers: int, hidden_dim: int, pooling: str = "mean"):
        self.model = model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self._handles = []
        self.attention_mask: Optional[torch.Tensor] = None

        # Patching config (set before each patched forward pass)
        self.patch_layer: Optional[int] = None
        self.d_A_unit = None  # (d,) unit concept vector for source domain attn
        self.d_B_unit = None  # (d,) unit concept vector for target domain attn
        self.alpha = None     # scalar: calibrated injection magnitude
        self.patching_enabled = False

        # Captured residual streams: layer_idx -> (B, d) on CPU
        self.residual_streams: Dict[int, torch.Tensor] = {}

    def _pool(self, act: torch.Tensor) -> torch.Tensor:
        """Pool (B, S, d) -> (B, d) using configured pooling mode."""
        if self.pooling == "last":
            return act[:, -1, :].float().cpu()
        else:
            mask = self.attention_mask.to(device=act.device, dtype=torch.float32)
            mask = mask.unsqueeze(-1)
            pooled = (act.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return pooled.cpu()

    def set_mask(self, mask: torch.Tensor):
        self.attention_mask = mask

    def configure_patch(self, patch_layer: int, d_A_unit: torch.Tensor,
                        d_B_unit: torch.Tensor, alpha: float):
        """Configure which layer to patch and with what vectors."""
        self.patch_layer = patch_layer
        self.d_A_unit = d_A_unit
        self.d_B_unit = d_B_unit
        self.alpha = alpha
        self.patching_enabled = True

    def disable_patch(self):
        self.patching_enabled = False

    def register_hooks(self):
        """Register attention-patching hook and residual-stream capture hooks."""
        layers = self.model.model.layers

        for i, layer in enumerate(layers):
            # Hook on self_attn to patch attention output at ℓ*
            h1 = layer.self_attn.register_forward_hook(self._make_attn_hook(i))
            # Hook on full layer to capture residual stream
            h2 = layer.register_forward_hook(self._make_layer_hook(i))
            self._handles.extend([h1, h2])

        print(f"  Registered {len(self._handles)} hooks for patching + capture")

    def _make_attn_hook(self, layer_idx: int):
        """Create hook that patches attention output at ℓ* (Eq. 6)."""
        def hook_fn(module, input, output):
            if not self.patching_enabled or layer_idx != self.patch_layer:
                return output

            # output is tuple: (attn_output, attn_weights, ...)
            if isinstance(output, tuple):
                act = output[0]  # (B, S, d)
            else:
                act = output

            device = act.device
            dtype = act.dtype

            d_A = self.d_A_unit.to(device=device, dtype=torch.float32)  # (d,)
            d_B = self.d_B_unit.to(device=device, dtype=torch.float32)
            alpha = self.alpha

            act_f = act.float()  # (B, S, d)

            # Project onto d_A: ⟨act, ˆd_A⟩
            proj_A = (act_f * d_A).sum(dim=-1, keepdim=True)  # (B, S, 1)

            # Eq. 6: remove domain-A component, inject domain-B component
            patched = act_f - proj_A * d_A.unsqueeze(0).unsqueeze(0) + alpha * d_B.unsqueeze(0).unsqueeze(0)

            patched = patched.to(dtype)

            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        return hook_fn

    def _make_layer_hook(self, layer_idx: int):
        """Capture residual stream after each layer."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            self.residual_streams[layer_idx] = self._pool(act).to(torch.float64)
        return hook_fn

    def get_residual_streams(self) -> Dict[int, torch.Tensor]:
        """Return captured residual streams and clear buffer."""
        result = dict(self.residual_streams)
        self.residual_streams.clear()
        return result

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        print("  All patching hooks removed.")


# ─────────────────────────────────────────────────────────────────────────────
# Domain shift score (Eq. 7)
# ─────────────────────────────────────────────────────────────────────────────
def compute_domain_shift_scores(
    clean_streams: Dict[int, torch.Tensor],
    patched_streams: Dict[int, torch.Tensor],
    d_B_layer: torch.Tensor,
    d_A_layer: torch.Tensor,
    patch_layer: int,
    num_layers: int,
) -> torch.Tensor:
    """
    Eq. 7: s^ℓ = ⟨ x̃^ℓ - x^ℓ(xA), (d^ℓ_B - d^ℓ_A) / ‖d^ℓ_B - d^ℓ_A‖ ⟩

    Returns: (num_layers,) tensor of shift scores (only valid for ℓ > patch_layer).
    """
    shift_scores = torch.zeros(num_layers, dtype=torch.float64)

    for l in range(num_layers):
        if l not in clean_streams or l not in patched_streams:
            continue

        diff_residual = patched_streams[l] - clean_streams[l]  # (B, d)
        direction = d_B_layer[l] - d_A_layer[l]  # (d,)
        direction_norm = direction.norm().clamp(min=1e-8)
        direction_unit = direction / direction_norm

        # Project each sample's shift onto the direction, then mean over batch
        proj = (diff_residual * direction_unit.unsqueeze(0)).sum(dim=1)  # (B,)
        shift_scores[l] = proj.mean()

    return shift_scores


# ─────────────────────────────────────────────────────────────────────────────
# Run patching experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_patching_experiment(
    model, tokenizer, hook_mgr: PatchingHookManager,
    df, args, domain_data: dict,
    source: str, target: str,
    patch_layers: List[int],
    num_layers: int, hidden_dim: int,
) -> Dict:
    """Run the full patching experiment across all specified patch layers."""

    d_A_attn_unit = domain_data[source]["attn_unit"]  # (L, d)
    d_B_attn_unit = domain_data[target]["attn_unit"]
    d_A_layer = domain_data[source]["layer_vec"]      # (L, d)
    d_B_layer = domain_data[target]["layer_vec"]
    alpha_B = domain_data[target]["alpha"]             # (L,)

    dataset = tokenize_domain(df, tokenizer, max_length=args.max_length)
    loader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    all_results = {}

    for patch_layer in patch_layers:
        print(f"\n{'─'*60}")
        print(f"Patching at layer {patch_layer}")
        print(f"  Source: {source} → Target: {target}")
        print(f"  α (injection magnitude): {alpha_B[patch_layer].item():.6f}")
        print(f"{'─'*60}")

        shift_scores_accum = torch.zeros(num_layers, dtype=torch.float64)
        total_samples = 0

        for batch in tqdm(loader, desc=f"  [patch@L{patch_layer}]", unit="batch"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bs = input_ids.size(0)

            try:
                target_device = model.model.embed_tokens.weight.device
            except AttributeError:
                target_device = next(model.parameters()).device

            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)
            hook_mgr.set_mask(attention_mask)

            # ── Clean forward pass ───────────────────────────────────
            hook_mgr.disable_patch()
            with torch.no_grad():
                clean_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, output_attentions=False,
                    output_hidden_states=False,
                ).logits[:, -1, :]  # (B, vocab)
            clean_streams = hook_mgr.get_residual_streams()

            # ── Patched forward pass ─────────────────────────────────
            hook_mgr.configure_patch(
                patch_layer=patch_layer,
                d_A_unit=d_A_attn_unit[patch_layer],
                d_B_unit=d_B_attn_unit[patch_layer],
                alpha=alpha_B[patch_layer].item(),
            )
            with torch.no_grad():
                patched_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, output_attentions=False,
                    output_hidden_states=False,
                ).logits[:, -1, :]
            patched_streams = hook_mgr.get_residual_streams()
            hook_mgr.disable_patch()

            # ── Compute domain shift scores (Eq. 7) ─────────────────
            batch_shifts = compute_domain_shift_scores(
                clean_streams, patched_streams,
                d_B_layer, d_A_layer,
                patch_layer, num_layers,
            )
            shift_scores_accum += batch_shifts * bs
            total_samples += bs

            del input_ids, attention_mask, clean_streams, patched_streams
            del clean_logits, patched_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average over all samples
        shift_scores_mean = shift_scores_accum / total_samples

        layer_result = {
            "patch_layer": patch_layer,
            "alpha": float(alpha_B[patch_layer].item()),
            "num_samples": total_samples,
            "shift_scores": shift_scores_mean,  # (num_layers,)
        }
        all_results[patch_layer] = layer_result

        print(f"\n  Shift scores (layers {patch_layer+1}..{num_layers-1}):")
        for l in range(patch_layer + 1, num_layers):
            print(f"    Layer {l:2d}: s = {shift_scores_mean[l]:.6f}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
def save_patching_results(output_dir: str, results: Dict, args):
    os.makedirs(output_dir, exist_ok=True)

    # Save per-patch-layer results
    for patch_layer, res in results.items():
        fname = f"shift_scores_patch_L{patch_layer}.pt"
        torch.save(res["shift_scores"], os.path.join(output_dir, fname))

    # Build summary JSON
    summary = {
        "experiment": "causal_activation_patching",
        "model": args.model,
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "num_samples": args.num_samples,
        "patch_layers": args.patch_layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_patch_layer": {},
    }

    for patch_layer, res in results.items():
        scores = res["shift_scores"]
        summary["per_patch_layer"][str(patch_layer)] = {
            "alpha": res["alpha"],
            "num_samples": res["num_samples"],
            "shift_scores": scores.tolist(),
            "post_patch_mean_shift": float(scores[patch_layer+1:].mean()),
            "post_patch_max_shift": float(scores[patch_layer+1:].max()),
            "post_patch_max_layer": int(scores[patch_layer+1:].argmax().item()) + patch_layer + 1,
        }

    # Aggregate: shift scores matrix (patch_layers x num_layers)
    patch_layer_list = sorted(results.keys())
    num_layers = len(next(iter(results.values()))["shift_scores"])
    shift_matrix = torch.stack([results[pl]["shift_scores"] for pl in patch_layer_list])
    torch.save(shift_matrix, os.path.join(output_dir, "shift_scores_matrix.pt"))
    summary["shift_matrix_shape"] = list(shift_matrix.shape)
    summary["shift_matrix_rows"] = patch_layer_list

    with open(os.path.join(output_dir, "patching_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {output_dir}")
    print(f"  - shift_scores_patch_L*.pt (per patch layer)")
    print(f"  - shift_scores_matrix.pt ({shift_matrix.shape})")
    print(f"  - patching_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Load domain vectors from Part 1 ──────────────────────────────────
    print(f"\n{'='*60}")
    print("Loading pre-computed domain vectors")
    print(f"{'='*60}")
    domain_data = load_domain_vectors(args.vectors_dir, args.source_domain, args.target_domain)
    print(f"  Source: {args.source_domain}, Target: {args.target_domain}")
    print(f"  Metadata: {domain_data['metadata']['samples_per_domain']}")

    # ── Load model ───────────────────────────────────────────────────────
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype],
        device_map=args.device, trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  {num_layers} layers, d={hidden_dim}")

    # ── Register patching hooks ──────────────────────────────────────────
    hook_mgr = PatchingHookManager(model, num_layers, hidden_dim, args.pooling)
    hook_mgr.register_hooks()

    # ── Load source domain data ──────────────────────────────────────────
    print(f"\nLoading source domain data ({args.source_domain})...")
    all_data = load_domain_data(
        data_dir=args.data_dir,
        samples_per_domain=args.num_samples,
        seed=args.seed,
    )
    source_df = all_data[args.source_domain]
    print(f"  {len(source_df)} source samples loaded")

    # ── Run patching ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RUNNING CAUSAL ACTIVATION PATCHING")
    print(f"  Patch layers: {args.patch_layers}")
    print(f"  Direction: {args.source_domain} → {args.target_domain}")
    print(f"{'='*60}")

    t_start = time.time()
    results = run_patching_experiment(
        model=model, tokenizer=tokenizer, hook_mgr=hook_mgr,
        df=source_df, args=args, domain_data=domain_data,
        source=args.source_domain, target=args.target_domain,
        patch_layers=args.patch_layers,
        num_layers=num_layers, hidden_dim=hidden_dim,
    )
    elapsed = time.time() - t_start
    print(f"\n  Patching complete in {elapsed:.1f}s")

    # ── Save ─────────────────────────────────────────────────────────────
    save_patching_results(args.output_dir, results, args)

    hook_mgr.remove_hooks()

    print(f"\n{'='*60}")
    print("Experiment 3 — Causal Activation Patching — COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
