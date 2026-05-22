#!/usr/bin/env python3
"""
exp3_final_shift.py  (v3 — RMS-norm alpha + random-vector control)
────────────────────────────────────────────────────────────────────
Changes from v2:
  1. Alpha normalised by per-layer RMS activation norm (not median clamp).
     Injection is kept at a constant *fraction* of the layer's dynamic range
     across all 28 layers, making L3 / L27 comparable with mid-layers.
  2. Random-vector control: a third "component" (rand) uses a fixed random
     unit vector instead of the target domain direction.  If L3 shows the
     same spike with random vectors, the effect is a disruption artifact.
  3. All 5 targets batched in one forward pass (slice-conditional hook).
  4. Model loaded on a single GPU (no inter-GPU pipeline stall).
"""
import argparse, json, os, time
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_domain_data, tokenize_domain, make_dataloader

DOMAINS    = ["cs", "eess", "math", "physics", "q-bio", "stat"]
NUM_LAYERS = 28
HIDDEN_DIM = 3584          # Qwen2.5-7B
SEED       = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors-dir", default="./results/exp3_causal_alldomains")
    p.add_argument("--data-dir",    default="./arxiv_data")
    p.add_argument("--output-dir",  default="./results/exp3_causal_alldomains/final_shift")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--gpu",         type=int, default=1)
    return p.parse_args()


# ── RMS-normalised alpha ──────────────────────────────────────────────────────
def rms_normalise_alpha(raw_alpha: torch.Tensor,
                        rms_norms: torch.Tensor) -> torch.Tensor:
    """
    Rescale raw alpha so the injection is a constant fraction of the layer's
    dynamic range across all layers.

    raw_alpha[l]  = mean |<Δc(x), d̂_k>|  (scales with layer norm)
    rms_norms[l]  = mean ||Δc^l||          (the layer's typical output scale)

    fraction[l]   = raw_alpha[l] / rms_norms[l]   (dimensionless, ≈ same everywhere)
    normalised[l] = fraction[l] * median(rms_norms) (back to a consistent absolute scale)
    """
    fraction   = raw_alpha / rms_norms.clamp(min=1e-6)
    med_rms    = rms_norms.median()
    return fraction * med_rms


# ── Load vectors ─────────────────────────────────────────────────────────────
def load_vectors(vectors_dir):
    rms_attn = torch.load(f"{vectors_dir}/rms_norm_attn_per_layer.pt", weights_only=True).float()
    rms_mlp  = torch.load(f"{vectors_dir}/rms_norm_mlp_per_layer.pt",  weights_only=True).float()

    print(f"\nRMS norms (attn): min={rms_attn.min():.2f}  max={rms_attn.max():.2f}  median={rms_attn.median():.2f}")
    print(f"RMS norms (mlp):  min={rms_mlp.min():.2f}  max={rms_mlp.max():.2f}  median={rms_mlp.median():.2f}")

    vecs = {}
    for d in DOMAINS:
        dd = os.path.join(vectors_dir, d)
        raw_a = torch.load(f"{dd}/alpha_projection_magnitudes.pt",     weights_only=True)
        raw_m = torch.load(f"{dd}/alpha_mlp_projection_magnitudes.pt", weights_only=True)
        vecs[d] = {
            "attn_unit": torch.load(f"{dd}/domain_vector_unit_attn.pt", weights_only=True),
            "mlp_unit":  torch.load(f"{dd}/domain_vector_unit_mlp.pt",  weights_only=True),
            "layer_vec": torch.load(f"{dd}/domain_vector_layer.pt",     weights_only=True),
            "alpha_attn": rms_normalise_alpha(raw_a, rms_attn),
            "alpha_mlp":  rms_normalise_alpha(raw_m, rms_mlp),
        }
        print(f"  {d}: alpha_attn L3={vecs[d]['alpha_attn'][3]:.3f}  L19={vecs[d]['alpha_attn'][19]:.3f} | "
              f"alpha_mlp L3={vecs[d]['alpha_mlp'][3]:.3f}  L27={vecs[d]['alpha_mlp'][27]:.3f}")
    return vecs, rms_attn, rms_mlp


def make_random_control(rms_attn, rms_mlp, hidden_dim, num_layers, seed=SEED):
    """
    Fixed random unit vectors for the control condition.
    One random unit vector per layer for each of attn/mlp.
    Alpha is set to the median RMS norm (same scale as a typical real injection).
    """
    torch.manual_seed(seed)
    rand_unit_attn = torch.randn(num_layers, hidden_dim)
    rand_unit_attn = rand_unit_attn / rand_unit_attn.norm(dim=1, keepdim=True).clamp(min=1e-8)
    rand_unit_mlp  = torch.randn(num_layers, hidden_dim)
    rand_unit_mlp  = rand_unit_mlp / rand_unit_mlp.norm(dim=1, keepdim=True).clamp(min=1e-8)

    alpha_rand_attn = torch.full((num_layers,), rms_attn.median().item())
    alpha_rand_mlp  = torch.full((num_layers,), rms_mlp.median().item())

    return {
        "attn_unit": rand_unit_attn,
        "mlp_unit":  rand_unit_mlp,
        "alpha_attn": alpha_rand_attn,
        "alpha_mlp":  alpha_rand_mlp,
    }


# ── Slice-conditional hook manager ───────────────────────────────────────────
class SlicePatchHookManager:
    """
    Replicates the source batch N_TARGETS times.
    Slice [i*B : (i+1)*B] gets patched toward targets[i].
    One forward pass handles all targets simultaneously.
    """
    def __init__(self, model, n_targets, bs):
        self.model     = model
        self.n_targets = n_targets
        self.bs        = bs
        self._handles  = []
        self.attn_mask  = None
        self.patch_layer = None
        self.patch_comp  = None
        self.d_A_unit    = None   # (d,)
        self.d_B_units   = None   # (n, d)
        self.alphas      = None   # (n,)
        self.enabled     = False
        self.final_pool  = None

    def configure(self, patch_layer, comp, d_A_unit, d_B_units, alphas):
        self.patch_layer = patch_layer
        self.patch_comp  = comp
        self.d_A_unit    = d_A_unit
        self.d_B_units   = d_B_units
        self.alphas      = alphas
        self.enabled     = True

    def disable(self):
        self.enabled = False

    def set_mask(self, mask):
        self.attn_mask = mask

    def _pool(self, act):
        mask = self.attn_mask.to(device=act.device, dtype=torch.float32).unsqueeze(-1)
        return ((act.float() * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().to(torch.float64)

    def _patch_hook(self, layer_idx, comp_type):
        def fn(module, inp, out):
            if not self.enabled or layer_idx != self.patch_layer or comp_type != self.patch_comp:
                return out
            act  = out[0] if isinstance(out, tuple) else out
            dev, dt = act.device, act.dtype
            d_A  = self.d_A_unit.to(dev, torch.float32)   # (d,)
            d_Bs = self.d_B_units.to(dev, torch.float32)  # (n, d)
            alps = self.alphas.to(dev, torch.float32)      # (n,)
            af   = act.float()
            out_f = af.clone()
            for i in range(self.n_targets):
                s, e = i * self.bs, (i + 1) * self.bs
                sl   = af[s:e]
                proj = (sl * d_A).sum(-1, keepdim=True)
                out_f[s:e] = sl - proj * d_A + alps[i] * d_Bs[i]
            return (out_f.to(dt),) + out[1:] if isinstance(out, tuple) else out_f.to(dt)
        return fn

    def _final_hook(self):
        def fn(module, inp, out):
            act = out[0] if isinstance(out, tuple) else out
            self.final_pool = self._pool(act)
        return fn

    def register(self):
        for i, layer in enumerate(self.model.model.layers):
            self._handles.append(layer.self_attn.register_forward_hook(self._patch_hook(i, "attn")))
            self._handles.append(layer.mlp.register_forward_hook(self._patch_hook(i, "mlp")))
        self._handles.append(
            self.model.model.layers[NUM_LAYERS - 1].register_forward_hook(self._final_hook()))

    def remove(self):
        for h in self._handles: h.remove()
        self._handles.clear()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    vecs, rms_attn, rms_mlp = load_vectors(args.vectors_dir)

    # Random control vectors (fixed seed, same alpha scale as real)
    rand_ctrl = make_random_control(rms_attn, rms_mlp, HIDDEN_DIM, NUM_LAYERS)
    print("\nRandom control alpha (attn):", rand_ctrl["alpha_attn"][0].item(), "(all layers same)")

    device = f"cuda:{args.gpu}"
    print(f"\nLoading model onto {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_data  = load_domain_data(args.data_dir, args.num_samples, SEED)
    pair_keys = [f"{s}->{t}" for s in DOMAINS for t in DOMAINS if s != t]

    # Components: attn, mlp, rand_attn, rand_mlp
    COMPS = ["attn", "mlp", "rand_attn", "rand_mlp"]
    # shape: (30 pairs, 4 comps, 28 layers)
    shift_out = torch.zeros(len(pair_keys), len(COMPS), NUM_LAYERS, dtype=torch.float64)

    t0 = time.time()

    for src_idx, source in enumerate(DOMAINS):
        targets = [d for d in DOMAINS if d != source]
        n_tgts  = len(targets)
        bs      = args.num_samples

        hook_mgr = SlicePatchHookManager(model, n_tgts, bs)
        hook_mgr.register()

        print(f"\n{'='*60}")
        print(f"Source: {source} ({src_idx+1}/6)")

        dataset = tokenize_domain(all_data[source], tok, max_length=384)
        ids_1   = dataset["input_ids"]
        mask_1  = dataset["attention_mask"]

        ids_rep  = ids_1.repeat(n_tgts, 1).to(device)
        mask_rep = mask_1.repeat(n_tgts, 1).to(device)
        hook_mgr.set_mask(mask_rep)

        # Clean pass
        hook_mgr.disable()
        with torch.no_grad():
            model(input_ids=ids_rep, attention_mask=mask_rep,
                  use_cache=False, output_attentions=False, output_hidden_states=False)
        clean_final = hook_mgr.final_pool.clone()

        # Per-target final direction  (d_B^L - d_A^L) / norm
        norm_dirs = []
        for target in targets:
            diff = vecs[target]["layer_vec"][-1] - vecs[source]["layer_vec"][-1]
            norm_dirs.append(diff / diff.norm().clamp(min=1e-8))

        # accum[t_idx, c_idx, l] = sum of shifts
        accum = torch.zeros(n_tgts, len(COMPS), NUM_LAYERS, dtype=torch.float64)

        # Iterate over (real attn, real mlp, rand attn, rand mlp)
        for c_idx, comp_key in enumerate(COMPS):
            is_rand = comp_key.startswith("rand_")
            base    = comp_key.replace("rand_", "")   # "attn" or "mlp"

            if is_rand:
                # Random control: same d_A removal, random d_B injection
                d_A_unit_per_l = vecs[source][f"{base}_unit"]     # (L, d) - still remove real domain-A
                d_B_rand       = rand_ctrl[f"{base}_unit"]        # (L, d)
                alpha_rand     = rand_ctrl[f"alpha_{base}"]       # (L,)
                # All targets get the SAME random vector (control is target-agnostic)
                d_B_stack  = d_B_rand.unsqueeze(0).expand(n_tgts, -1, -1)   # (n, L, d)
                alpha_stack = alpha_rand.unsqueeze(0).expand(n_tgts, -1)    # (n, L)
            else:
                d_A_unit_per_l = vecs[source][f"{base}_unit"]    # (L, d)
                d_B_stack  = torch.stack([vecs[t][f"{base}_unit"]     for t in targets], dim=0)  # (n, L, d)
                alpha_stack = torch.stack([vecs[t][f"alpha_{base}"]   for t in targets], dim=0)  # (n, L)

            for pl in tqdm(range(NUM_LAYERS),
                           desc=f"  [{source}] {comp_key}", dynamic_ncols=True):
                hook_mgr.configure(
                    patch_layer = pl,
                    comp        = base,
                    d_A_unit    = d_A_unit_per_l[pl],       # (d,)
                    d_B_units   = d_B_stack[:, pl, :],      # (n, d)
                    alphas      = alpha_stack[:, pl],        # (n,)
                )
                with torch.no_grad():
                    model(input_ids=ids_rep, attention_mask=mask_rep,
                          use_cache=False, output_attentions=False, output_hidden_states=False)
                hook_mgr.disable()

                pat_final = hook_mgr.final_pool
                diff_all  = pat_final - clean_final

                for t_idx in range(n_tgts):
                    s, e  = t_idx * bs, (t_idx + 1) * bs
                    shift = (diff_all[s:e] * norm_dirs[t_idx]).sum(1).mean()
                    accum[t_idx, c_idx, pl] = shift

        hook_mgr.remove()

        for t_idx, target in enumerate(targets):
            idx = pair_keys.index(f"{source}->{target}")
            shift_out[idx] = accum[t_idx]

        print(f"  ✓ {source} done — elapsed {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    torch.save(shift_out, os.path.join(args.output_dir, "final_shift_scores.pt"))
    with open(os.path.join(args.output_dir, "final_shift_results.json"), "w") as f:
        json.dump({"pair_keys": pair_keys, "components": COMPS,
                   "num_layers": NUM_LAYERS, "elapsed_seconds": elapsed}, f, indent=2)
    print(f"Saved {shift_out.shape} → {args.output_dir}")


if __name__ == "__main__":
    main()
