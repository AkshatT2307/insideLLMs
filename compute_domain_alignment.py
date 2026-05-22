#!/usr/bin/env python3
"""
compute_domain_alignment.py  —  Experiment 3

Computes domain alignment ρ between LoRA weight updates (ΔW = α/r · B·A)
and domain concept vectors from Experiment 1.

Metrics computed:
  1. Per-module alignment  ρ = ‖ΔW · d̂‖ / ‖ΔW‖_F
  2. Cross-domain alignment (adapter_k vs domain_j for all j)
  3. Block-pooled (Frobenius-weighted) MLP vs Attention scores
  4. SVD overlap of top singular vectors with domain directions

Output structure:
  results/experiment3_alignment/
  ├── per_module/         epoch_N/mode/domain.json
  ├── cross_domain/       epoch_N/mode/adapter_domain.json
  ├── block_pooled/       epoch_N/mode/domain.json
  └── svd_overlap/        epoch_N/mode/domain.json
"""

import os
import json
import numpy as np
import h5py
import torch
from safetensors import safe_open

# ── Constants ────────────────────────────────────────────────────────────────
DOMAINS = ["cs", "eess", "math", "physics", "q-bio", "stat"]
EPOCHS = [1, 2, 3]
NUM_LAYERS = 28
LORA_ALPHA = 32
LORA_RANK = 16
SCALING = LORA_ALPHA / LORA_RANK  # 2.0

MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_MODULES = ATTN_MODULES + MLP_MODULES
POOLING_MODES = ["last", "mean"]

# Modules where output dim (not input) matches d_model → use ΔW^T · d̂
TRANSPOSE_MODULES = {"down_proj"}

# Which domain-vector component to use for each module type
COMP_FOR_MODULE = {m: "mlp" for m in MLP_MODULES}
COMP_FOR_MODULE.update({m: "attn" for m in ATTN_MODULES})

ADAPTER_DIR = "FineTuning/adapters"
DOMAIN_VECTORS_PATH = "results/domain_vectors.h5"
OUTPUT_BASE = "results/experiment3_alignment"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_domain_vectors(h5_path):
    """Load all domain vectors. Returns dict[domain][comp][mode] → (28, 3584)."""
    vectors = {}
    with h5py.File(h5_path, "r") as f:
        for domain in DOMAINS:
            vectors[domain] = {}
            for comp in ["attn", "mlp"]:
                vectors[domain][comp] = {}
                for mode in POOLING_MODES:
                    key = f"domain_vectors/{domain}/{comp}/{mode}"
                    vectors[domain][comp][mode] = np.array(f[key], dtype=np.float32)
    return vectors


def load_adapter(domain, epoch):
    """Load LoRA adapter weights as float32 tensors."""
    path = os.path.join(ADAPTER_DIR, domain, f"epoch_{epoch}", "adapter_model.safetensors")
    weights = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).float()
    return weights


def get_lora_AB(weights, layer_idx, module_name):
    """Return (B, A) tensors for a specific module. B:(d_out,r), A:(r,d_in)."""
    block = "mlp" if module_name in MLP_MODULES else "self_attn"
    prefix = f"base_model.model.model.layers.{layer_idx}.{block}.{module_name}"
    A = weights[f"{prefix}.lora_A.weight"]
    B = weights[f"{prefix}.lora_B.weight"]
    return B, A


def compute_alignment(B, A, dv_unit, transpose=False):
    """
    Compute ρ = ‖ΔW · d̂‖ / ‖ΔW‖_F  where ΔW = scaling * B @ A.
    If transpose, computes ‖ΔW^T · d̂‖ / ‖ΔW‖_F instead.
    Returns (rho, frobenius_norm, output_norm).
    """
    # ΔW · d̂ = scaling * B @ (A @ d̂)  — never materialise full ΔW
    if transpose:
        # ΔW^T · d̂ = scaling * A^T @ (B^T @ d̂)
        inner = B.T @ dv_unit          # (r,)
        output = SCALING * (A.T @ inner)  # (d_in,)
    else:
        inner = A @ dv_unit              # (r,)
        output = SCALING * (B @ inner)    # (d_out,)

    output_norm = torch.norm(output).item()

    # ‖ΔW‖_F = scaling * ‖B @ A‖_F.  Efficient: ‖B@A‖_F^2 = tr(A^T B^T B A)
    BtB = B.T @ B   # (r, r)
    AAt = A @ A.T    # (r, r)  — actually we need A^T (B^T B) A trace
    frob_sq = torch.trace(AAt @ BtB).item()
    frob_norm = SCALING * (frob_sq ** 0.5)

    rho = output_norm / frob_norm if frob_norm > 1e-12 else 0.0
    return rho, frob_norm, output_norm


def efficient_svd_overlap(B, A, dv_unit, transpose=False, top_k=None):
    """
    SVD overlap of ΔW = scaling * B @ A with domain vector.
    Exploits low-rank structure (rank ≤ r) for efficiency.
    """
    r = A.shape[0]
    if top_k is None:
        top_k = r

    # SVD of A  (r × d_in) → U_a(r,r), S_a(r), Vh_a(r, d_in)
    U_a, S_a, Vh_a = torch.linalg.svd(A, full_matrices=False)

    # C = B @ U_a  (d_out, r);  CS = C * S_a
    C = B @ U_a
    CS = C * S_a.unsqueeze(0)

    # SVD of CS  (d_out × r) → U_c(d_out,r), S_c(r), Vh_c(r,r)
    U_c, S_c, Vh_c = torch.linalg.svd(CS, full_matrices=False)

    S_final = SCALING * S_c  # singular values of ΔW

    if transpose:
        # Left singular vectors of ΔW = U_c
        vectors = U_c[:, :top_k]  # (d_out, top_k)
    else:
        # Right singular vectors: rows of Vh_final = Vh_c @ Vh_a
        Vh_final = Vh_c @ Vh_a  # (r, d_in)
        vectors = Vh_final[:top_k, :].T  # (d_in, top_k)

    overlaps = (vectors.T @ dv_unit).abs()  # (top_k,)

    return {
        "singular_values": S_final[:top_k].tolist(),
        "overlaps": overlaps.tolist(),
        "total_overlap_sq": overlaps.pow(2).sum().item(),
        "top_k": top_k,
    }


def unit_vector(v_np):
    """L2-normalise a numpy vector, return torch tensor."""
    t = torch.from_numpy(v_np).float()
    n = torch.norm(t)
    if n < 1e-12:
        return t, 0.0
    return t / n, n.item()


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading domain vectors …")
    domain_vectors = load_domain_vectors(DOMAIN_VECTORS_PATH)

    for epoch in EPOCHS:
        print(f"\n{'='*60}\nEpoch {epoch}\n{'='*60}")

        # Pre-load all adapters for this epoch
        adapters = {}
        for domain in DOMAINS:
            print(f"  Loading adapter: {domain}/epoch_{epoch}")
            adapters[domain] = load_adapter(domain, epoch)

        for mode in POOLING_MODES:
            print(f"\n  Pooling mode: {mode}")

            for adapter_domain in DOMAINS:
                weights = adapters[adapter_domain]
                print(f"    Adapter: {adapter_domain}")

                # ── Per-module same-domain alignment ──────────────────────
                per_module = {}  # layer → module → metrics
                svd_data = {}    # layer → module → svd info

                for layer in range(NUM_LAYERS):
                    per_module[str(layer)] = {}
                    svd_data[str(layer)] = {}

                    for mod in ALL_MODULES:
                        comp = COMP_FOR_MODULE[mod]
                        dv_np = domain_vectors[adapter_domain][comp][mode][layer]
                        dv_unit, dv_norm = unit_vector(dv_np)
                        B, A = get_lora_AB(weights, layer, mod)
                        tr = mod in TRANSPOSE_MODULES

                        rho, frob, out_n = compute_alignment(B, A, dv_unit, transpose=tr)
                        per_module[str(layer)][mod] = {
                            "rho": rho,
                            "frobenius_norm": frob,
                            "output_norm": out_n,
                            "domain_vector_norm": dv_norm,
                            "transpose_used": tr,
                        }

                        svd_info = efficient_svd_overlap(B, A, dv_unit, transpose=tr)
                        svd_data[str(layer)][mod] = svd_info

                # Save per-module & SVD
                pm_path = os.path.join(OUTPUT_BASE, "per_module", f"epoch_{epoch}", mode, f"{adapter_domain}.json")
                save_json(per_module, pm_path)

                svd_path = os.path.join(OUTPUT_BASE, "svd_overlap", f"epoch_{epoch}", mode, f"{adapter_domain}.json")
                save_json(svd_data, svd_path)

                # ── Cross-domain alignment ────────────────────────────────
                cross = {}  # layer → module → {target_domain: rho}
                for layer in range(NUM_LAYERS):
                    cross[str(layer)] = {}
                    for mod in ALL_MODULES:
                        comp = COMP_FOR_MODULE[mod]
                        B, A = get_lora_AB(weights, layer, mod)
                        tr = mod in TRANSPOSE_MODULES

                        cross[str(layer)][mod] = {}
                        for target_domain in DOMAINS:
                            dv_np = domain_vectors[target_domain][comp][mode][layer]
                            dv_unit, _ = unit_vector(dv_np)
                            rho, _, _ = compute_alignment(B, A, dv_unit, transpose=tr)
                            cross[str(layer)][mod][target_domain] = rho

                cd_path = os.path.join(OUTPUT_BASE, "cross_domain", f"epoch_{epoch}", mode, f"{adapter_domain}.json")
                save_json(cross, cd_path)

                # ── Block-pooled alignment ────────────────────────────────
                block_pooled = {}
                for layer in range(NUM_LAYERS):
                    bp = {}
                    for block_name, modules in [("attn", ATTN_MODULES), ("mlp", MLP_MODULES)]:
                        total_w, weighted_rho = 0.0, 0.0
                        module_rhos = {}
                        for m in modules:
                            r = per_module[str(layer)][m]
                            w = r["frobenius_norm"]
                            total_w += w
                            weighted_rho += w * r["rho"]
                            module_rhos[m] = r["rho"]

                        pooled = weighted_rho / total_w if total_w > 1e-12 else 0.0

                        entry = {
                            "pooled_rho_frobenius_weighted": pooled,
                            "simple_mean_rho": float(np.mean(list(module_rhos.values()))),
                            "total_frobenius_norm": total_w,
                            "per_module_rho": module_rhos,
                        }

                        # MLP: also store version excluding down_proj
                        if block_name == "mlp":
                            w_no_d, rho_no_d = 0.0, 0.0
                            for m in ["gate_proj", "up_proj"]:
                                r = per_module[str(layer)][m]
                                w_no_d += r["frobenius_norm"]
                                rho_no_d += r["frobenius_norm"] * r["rho"]
                            entry["pooled_rho_no_down_proj"] = rho_no_d / w_no_d if w_no_d > 1e-12 else 0.0

                        bp[block_name] = entry
                    block_pooled[str(layer)] = bp

                bp_path = os.path.join(OUTPUT_BASE, "block_pooled", f"epoch_{epoch}", mode, f"{adapter_domain}.json")
                save_json(block_pooled, bp_path)

        # Free adapter memory
        del adapters
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"All results saved to {OUTPUT_BASE}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
