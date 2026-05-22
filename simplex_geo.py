#!/usr/bin/env python3
"""
simplex_geometry.py  —  Experiment 1: Domain Concept Vectors + Simplex Geometry

Phase 1: Residual stream domain concept vectors + simplex test (output_hidden_states).
Phase 2: Attention/MLP contribution concept vectors + simplex test (hooks on self_attn, mlp).
         Verifies additive decomposition identity (Eq. 5).

Usage:
    python simplex_geometry.py --data-dir ./arxiv_data --output-dir ./results/simplex
"""

import argparse, json, os, time
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ── Constants ────────────────────────────────────────────────────────────────
DOMAIN_FILES = {
    "cs": "cs.csv", "eess": "eess.csv", "math": "math.csv",
    "physics": "physics.csv", "q-bio": "q-bio.csv", "stat": "stat.csv",
}
DOMAINS = sorted(DOMAIN_FILES.keys())
K = len(DOMAINS)
IDEAL_OFF_DIAG = -1.0 / (K - 1)  # -0.2


def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 1: Domain concept vectors + simplex geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--data-dir", default="./arxiv_data")
    p.add_argument("--output-dir", default="./results/simplex")
    p.add_argument("--samples-per-domain", type=int, default=500)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Model + Tokenizer ───────────────────────────────────────────────────────
def load_model_and_tokenizer(args):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    print(f"\n{'='*60}\nLoading model: {args.model}\n{'='*60}")
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
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  {L} layers, d={d}\n")
    return model, tokenizer, L, d


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_and_tokenize(args, tokenizer):
    domain_data = {}
    for dom in DOMAINS:
        fp = os.path.join(args.data_dir, DOMAIN_FILES[dom])
        df = pd.read_csv(fp).dropna(subset=["text"]).reset_index(drop=True)
        if len(df) > args.samples_per_domain:
            df = df.sample(n=args.samples_per_domain, random_state=args.seed).reset_index(drop=True)
        enc = tokenizer(df["text"].tolist(), padding="max_length", truncation=True,
                        max_length=args.max_length, return_tensors="pt")
        domain_data[dom] = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        print(f"  {dom}: {len(df)} samples")
    return domain_data


def get_device(model):
    try:
        return model.model.embed_tokens.weight.device
    except AttributeError:
        return next(model.parameters()).device


# ── Simplex Utilities ────────────────────────────────────────────────────────
def ideal_simplex_matrix():
    S = np.full((K, K), IDEAL_OFF_DIAG)
    np.fill_diagonal(S, 1.0)
    return S

def simplex_deviation(cos_mat, ideal):
    return np.linalg.norm(cos_mat - ideal, ord="fro")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Residual Stream
# ══════════════════════════════════════════════════════════════════════════════
def phase1(model, domain_data, L, d, batch_size, output_dir):
    print(f"\n{'='*60}\nPhase 1: Residual Stream Concept Vectors\n{'='*60}")
    S = L + 1  # number of hidden states (embedding + L layers)
    dev = get_device(model)

    # Accumulate domain sums
    sums = {dom: np.zeros((S, d), dtype=np.float64) for dom in DOMAINS}
    counts = {dom: 0 for dom in DOMAINS}

    for dom in DOMAINS:
        ids = domain_data[dom]["input_ids"]
        mask = domain_data[dom]["attention_mask"]
        n = ids.size(0)
        for s in tqdm(range(0, n, batch_size), desc=f"  P1 [{dom}]", unit="b"):
            e = min(s + batch_size, n)
            with torch.no_grad():
                out = model(input_ids=ids[s:e].to(dev),
                            attention_mask=mask[s:e].to(dev),
                            output_hidden_states=True, use_cache=False)
            for li, hs in enumerate(out.hidden_states):
                sums[dom][li] += hs[:, -1, :].float().cpu().numpy().sum(axis=0)
            counts[dom] += (e - s)
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Domain means, global mean, concept vectors
    means = {dom: sums[dom] / counts[dom] for dom in DOMAINS}
    total = sum(counts.values())
    g_mean = sum(counts[dom] * means[dom] for dom in DOMAINS) / total
    cvecs = {dom: means[dom] - g_mean for dom in DOMAINS}  # (S, d)

    # Cosine matrices + simplex deviation at every layer
    ideal = ideal_simplex_matrix()
    cos_mats = np.zeros((S, K, K))
    s_devs = np.zeros(S)
    for ell in range(S):
        V = np.stack([cvecs[dom][ell] for dom in DOMAINS])
        cos_mats[ell] = sk_cosine(V)
        s_devs[ell] = simplex_deviation(cos_mats[ell], ideal)

    # Concept vector norms
    norms = {dom: np.linalg.norm(cvecs[dom], axis=1) for dom in DOMAINS}

    # Print
    print(f"\n  Concept vector norms:")
    for dom in DOMAINS:
        print(f"    {dom:>10s}: max={norms[dom].max():.4f} mean={norms[dom].mean():.4f}")
    print(f"\n  Simplex deviation (residual stream):")
    for ell in range(S):
        tag = "emb" if ell == 0 else f"L{ell-1}"
        print(f"    {tag:>5s}: {s_devs[ell]:.4f}")
    print(f"  Best: layer {np.argmin(s_devs)} (δ={s_devs.min():.4f})")

    # ── Save ─────────────────────────────────────────────────────────────
    cv_arr = np.stack([cvecs[dom] for dom in DOMAINS])        # (K, S, d)
    dm_arr = np.stack([means[dom] for dom in DOMAINS])        # (K, S, d)
    nm_arr = np.stack([norms[dom] for dom in DOMAINS])        # (K, S)

    np.savez_compressed(os.path.join(output_dir, "phase1_concept_vectors.npz"),
                        concept_vectors=cv_arr, domain_means=dm_arr,
                        global_mean=g_mean, domains=DOMAINS)
    np.savez_compressed(os.path.join(output_dir, "phase1_cosine_matrices.npz"),
                        cosine_matrices=cos_mats, simplex_deviations=s_devs,
                        ideal_matrix=ideal, domains=DOMAINS)
    np.savez_compressed(os.path.join(output_dir, "phase1_concept_norms.npz"),
                        norms=nm_arr, domains=DOMAINS)

    # ── Figures ──────────────────────────────────────────────────────────
    _plot_simplex_curve({"Residual Stream": s_devs}, L,
                        os.path.join(output_dir, "simplex_deviation_residual.png"))
    _plot_cosine_heatmaps(cos_mats, L, "Residual Stream",
                          os.path.join(output_dir, "cosine_heatmaps_residual.png"))
    _plot_norms(nm_arr, L, os.path.join(output_dir, "concept_norms_residual.png"))

    print(f"  ✓ Phase 1 complete.\n")
    return cvecs, means, g_mean


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Attention + MLP Contributions
# ══════════════════════════════════════════════════════════════════════════════
class ComponentStore:
    """Hooks on self_attn and mlp; captures last-token activations per batch."""
    def __init__(self, L):
        self.L = L
        self._handles = []
        self._attn = {}
        self._mlp = {}

    def register(self, model):
        for i, layer in enumerate(model.model.layers):
            self._handles.append(
                layer.self_attn.register_forward_hook(self._hook("attn", i, True)))
            self._handles.append(
                layer.mlp.register_forward_hook(self._hook("mlp", i, False)))
        print(f"  Registered {len(self._handles)} hooks (2×{self.L} layers)")

    def _hook(self, comp, idx, is_tuple):
        def fn(mod, inp, out):
            act = out[0] if is_tuple else out
            if isinstance(act, tuple):
                act = act[0]
            vec = act[:, -1, :].float().cpu().numpy()
            if comp == "attn":
                self._attn[idx] = vec
            else:
                self._mlp[idx] = vec
        return fn

    def pop(self):
        a, m = self._attn, self._mlp
        self._attn, self._mlp = {}, {}
        return a, m

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        print("  Hooks removed.")


def phase2(model, domain_data, L, d, batch_size, output_dir, res_cvecs):
    print(f"\n{'='*60}\nPhase 2: Attention + MLP Contributions\n{'='*60}")
    dev = get_device(model)
    store = ComponentStore(L)
    store.register(model)

    a_sums = {dom: np.zeros((L, d), dtype=np.float64) for dom in DOMAINS}
    m_sums = {dom: np.zeros((L, d), dtype=np.float64) for dom in DOMAINS}
    counts = {dom: 0 for dom in DOMAINS}

    for dom in DOMAINS:
        ids = domain_data[dom]["input_ids"]
        mask = domain_data[dom]["attention_mask"]
        n = ids.size(0)
        for s in tqdm(range(0, n, batch_size), desc=f"  P2 [{dom}]", unit="b"):
            e = min(s + batch_size, n)
            with torch.no_grad():
                model(input_ids=ids[s:e].to(dev), attention_mask=mask[s:e].to(dev),
                      use_cache=False, output_hidden_states=False)
            ba, bm = store.pop()
            for li in range(L):
                a_sums[dom][li] += ba[li].sum(axis=0)
                m_sums[dom][li] += bm[li].sum(axis=0)
            counts[dom] += (e - s)
            del ba, bm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    store.remove()

    # Means + concept vectors
    a_means = {dom: a_sums[dom] / counts[dom] for dom in DOMAINS}
    m_means = {dom: m_sums[dom] / counts[dom] for dom in DOMAINS}
    total = sum(counts.values())
    a_glob = sum(counts[dom] * a_means[dom] for dom in DOMAINS) / total
    m_glob = sum(counts[dom] * m_means[dom] for dom in DOMAINS) / total
    a_cv = {dom: a_means[dom] - a_glob for dom in DOMAINS}
    m_cv = {dom: m_means[dom] - m_glob for dom in DOMAINS}

    # ── Decomposition check (Eq. 5) ──────────────────────────────────────
    decomp_err = np.zeros(L)
    for ell in range(L):
        errs = []
        for dom in DOMAINS:
            lhs = res_cvecs[dom][ell + 1]
            rhs = res_cvecs[dom][ell] + a_cv[dom][ell] + m_cv[dom][ell]
            errs.append(np.max(np.abs(lhs - rhs)))
        decomp_err[ell] = max(errs)
    print(f"\n  Decomposition (Eq.5) max error: {decomp_err.max():.6e}")

    # ── Cosine + simplex deviation ───────────────────────────────────────
    ideal = ideal_simplex_matrix()
    a_cos = np.zeros((L, K, K))
    m_cos = np.zeros((L, K, K))
    a_sd = np.zeros(L)
    m_sd = np.zeros(L)
    for ell in range(L):
        Va = np.stack([a_cv[dom][ell] for dom in DOMAINS])
        Vm = np.stack([m_cv[dom][ell] for dom in DOMAINS])
        a_cos[ell] = sk_cosine(Va)
        m_cos[ell] = sk_cosine(Vm)
        a_sd[ell] = simplex_deviation(a_cos[ell], ideal)
        m_sd[ell] = simplex_deviation(m_cos[ell], ideal)

    # Norms
    a_norms = np.stack([np.linalg.norm(a_cv[dom], axis=1) for dom in DOMAINS])
    m_norms = np.stack([np.linalg.norm(m_cv[dom], axis=1) for dom in DOMAINS])

    # Print
    print(f"\n  Simplex deviation — Attention:")
    for ell in range(L):
        print(f"    L{ell:>2d}: {a_sd[ell]:.4f}")
    print(f"  Simplex deviation — MLP:")
    for ell in range(L):
        print(f"    L{ell:>2d}: {m_sd[ell]:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    np.savez_compressed(os.path.join(output_dir, "phase2_concept_vectors.npz"),
        attn_concept_vectors=np.stack([a_cv[dom] for dom in DOMAINS]),
        mlp_concept_vectors=np.stack([m_cv[dom] for dom in DOMAINS]),
        attn_domain_means=np.stack([a_means[dom] for dom in DOMAINS]),
        mlp_domain_means=np.stack([m_means[dom] for dom in DOMAINS]),
        attn_global_mean=a_glob, mlp_global_mean=m_glob, domains=DOMAINS)
    np.savez_compressed(os.path.join(output_dir, "phase2_cosine_matrices.npz"),
        attn_cosine_matrices=a_cos, mlp_cosine_matrices=m_cos,
        attn_simplex_deviations=a_sd, mlp_simplex_deviations=m_sd,
        ideal_matrix=ideal, domains=DOMAINS)
    np.savez_compressed(os.path.join(output_dir, "phase2_concept_norms.npz"),
        attn_norms=a_norms, mlp_norms=m_norms, domains=DOMAINS)
    np.save(os.path.join(output_dir, "decomposition_errors.npy"), decomp_err)

    # ── Combined JSON summary ────────────────────────────────────────────
    # Load phase1 simplex deviations for the combined summary
    p1 = np.load(os.path.join(output_dir, "phase1_cosine_matrices.npz"))
    res_sd = p1["simplex_deviations"]

    summary = {
        "domains": DOMAINS, "K": K, "ideal_off_diagonal": IDEAL_OFF_DIAG,
        "residual_simplex_deviations": res_sd.tolist(),
        "attn_simplex_deviations": a_sd.tolist(),
        "mlp_simplex_deviations": m_sd.tolist(),
        "decomposition_max_errors": decomp_err.tolist(),
        "residual_best_layer": int(np.argmin(res_sd)),
        "attn_best_layer": int(np.argmin(a_sd)),
        "mlp_best_layer": int(np.argmin(m_sd)),
    }
    with open(os.path.join(output_dir, "simplex_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Figures ──────────────────────────────────────────────────────────
    # Pad residual deviations (L+1) to align: use layers 1..L for comparison
    _plot_simplex_curve(
        {"Residual Stream": res_sd[1:], "Attention Δa": a_sd, "MLP Δm": m_sd},
        L, os.path.join(output_dir, "simplex_deviation_all.png"))
    _plot_cosine_heatmaps(a_cos, L, "Attention Contributions",
                          os.path.join(output_dir, "cosine_heatmaps_attn.png"))
    _plot_cosine_heatmaps(m_cos, L, "MLP Contributions",
                          os.path.join(output_dir, "cosine_heatmaps_mlp.png"))
    _plot_decomp(decomp_err, L, os.path.join(output_dir, "decomposition_check.png"))
    _plot_norms(a_norms, L, os.path.join(output_dir, "concept_norms_attn.png"),
                title="Attention Contribution Concept Vector Norms")
    _plot_norms(m_norms, L, os.path.join(output_dir, "concept_norms_mlp.png"),
                title="MLP Contribution Concept Vector Norms")

    print(f"  ✓ Phase 2 complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {"Residual Stream": "#2196F3", "Attention Δa": "#FF5722", "MLP Δm": "#4CAF50"}

def _plot_simplex_curve(curves, L, filepath):
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, vals in curves.items():
        x = np.arange(len(vals))
        ax.plot(x, vals, marker="o", markersize=3, label=label,
                color=COLORS.get(label, None), linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Simplex Deviation (Frobenius)")
    ax.set_title("Simplex Deviation Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def _plot_cosine_heatmaps(cos_mats, L, title_prefix, filepath):
    """Plot 6×6 cosine heatmaps at 5 evenly spaced layers."""
    n_layers = cos_mats.shape[0]
    indices = np.linspace(0, n_layers - 1, min(5, n_layers), dtype=int)
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        im = ax.imshow(cos_mats[idx], vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(K)); ax.set_xticklabels(DOMAINS, rotation=45, fontsize=7)
        ax.set_yticks(range(K)); ax.set_yticklabels(DOMAINS, fontsize=7)
        ax.set_title(f"Layer {idx}", fontsize=10)
        # Annotate cells
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cos_mats[idx, i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(cos_mats[idx, i, j]) > 0.5 else "black")
    fig.suptitle(f"{title_prefix} — Pairwise Cosine Similarity", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Cosine Similarity")
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def _plot_norms(norms_arr, L, filepath, title="Concept Vector Norms"):
    """norms_arr: (K, num_layers)"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, dom in enumerate(DOMAINS):
        ax.plot(np.arange(norms_arr.shape[1]), norms_arr[i], label=dom, linewidth=1.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("‖d_k^ℓ‖₂")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def _plot_decomp(errors, L, filepath):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(np.arange(L), errors, marker="o", markersize=3, color="#9C27B0")
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Max |d_k^{ℓ+1} − (d_k^ℓ + Δa + Δm)|")
    ax.set_title("Additive Decomposition Check (Eq. 5)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, L, d = load_model_and_tokenizer(args)
    domain_data = load_and_tokenize(args, tokenizer)

    t0 = time.time()

    # Phase 1 — residual stream concept vectors + simplex
    res_cvecs, _, _ = phase1(model, domain_data, L, d, args.batch_size, args.output_dir)

    # Phase 2 — attention/MLP contributions + simplex + decomposition
    phase2(model, domain_data, L, d, args.batch_size, args.output_dir, res_cvecs)

    elapsed = time.time() - t0
    print(f"{'='*60}")
    print(f"All done!  {elapsed:.1f}s total")
    print(f"Output: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()