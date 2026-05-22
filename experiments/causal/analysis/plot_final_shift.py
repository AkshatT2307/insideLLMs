#!/usr/bin/env python3
"""
exp3_plot_final_shift.py  (v3 — 4 components including random control)
"""
import os, sys, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.plot_style import plt, setup_style, save_fig, ATTN_COLOR, MLP_COLOR
import matplotlib.gridspec as gridspec

setup_style()

def main():
    results_dir = "./results/exp3_causal_alldomains/final_shift"
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    shift_tensor = torch.load(os.path.join(results_dir, "final_shift_scores.pt"), weights_only=True)
    with open(os.path.join(results_dir, "final_shift_results.json")) as f:
        summary = json.load(f)

    pair_keys = summary["pair_keys"]
    comps     = summary["components"]   # ["attn","mlp","rand_attn","rand_mlp"]
    n_pairs, n_comps, n_layers = shift_tensor.shape
    layers = np.arange(n_layers)

    STYLES = {
        "attn":      dict(color="#1f77b4", ls="-",  lw=2.5, label="Attention (domain)"),
        "mlp":       dict(color="#ff7f0e", ls="--", lw=2.5, label="MLP (domain)"),
        "rand_attn": dict(color="#1f77b4", ls=":",  lw=1.5, label="Attention (random ctrl)", alpha=0.6),
        "rand_mlp":  dict(color="#ff7f0e", ls=":",  lw=1.5, label="MLP (random ctrl)",       alpha=0.6),
    }

    # ── 1. Mean aggregate plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for ci, comp in enumerate(comps):
        mu  = shift_tensor[:, ci, :].mean(0).numpy()
        sd  = shift_tensor[:, ci, :].std(0).numpy()
        st  = STYLES[comp]
        ax.plot(layers, mu, label=st["label"], color=st["color"],
                ls=st["ls"], lw=st["lw"], alpha=st.get("alpha", 1.0))
        ax.fill_between(layers, mu - sd, mu + sd, color=st["color"],
                        alpha=0.08 if "rand" in comp else 0.15)

    ax.axhline(0, color="black", lw=0.8, alpha=0.3)
    ax.set_title("Final Domain Shift by Patch Layer\n(Mean ± 1 SD across 30 pairs, RMS-normalised alpha)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Patch Layer $\\ell^*$", fontsize=12)
    ax.set_ylabel("Final Domain Shift (at Layer 27)", fontsize=12)
    ax.set_xticks(layers[::2])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_mean_final_shift.png"))
    plt.close()
    print("Saved 01_mean_final_shift.png")

    # ── 2. L3 artifact test: random vs domain for MLP ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for ax, comp_pair, title in zip(
            axes,
            [("mlp", "rand_mlp"), ("attn", "rand_attn")],
            ["MLP: Domain vs Random Control", "Attention: Domain vs Random Control"]):
        c_dom,  c_rnd = comp_pair
        ci_dom  = comps.index(c_dom)
        ci_rnd  = comps.index(c_rnd)
        mu_dom  = shift_tensor[:, ci_dom, :].mean(0).numpy()
        sd_dom  = shift_tensor[:, ci_dom, :].std(0).numpy()
        mu_rnd  = shift_tensor[:, ci_rnd, :].mean(0).numpy()
        sd_rnd  = shift_tensor[:, ci_rnd, :].std(0).numpy()

        ax.plot(layers, mu_dom, color=STYLES[c_dom]["color"], lw=2.5,
                ls=STYLES[c_dom]["ls"], label=f"{c_dom} (domain)")
        ax.fill_between(layers, mu_dom - sd_dom, mu_dom + sd_dom,
                        color=STYLES[c_dom]["color"], alpha=0.15)
        ax.plot(layers, mu_rnd, color=STYLES[c_rnd]["color"], lw=1.5,
                ls=STYLES[c_rnd]["ls"], label=f"{c_rnd} (random)", alpha=0.7)
        ax.fill_between(layers, mu_rnd - sd_rnd, mu_rnd + sd_rnd,
                        color=STYLES[c_rnd]["color"], alpha=0.08)
        ax.axhline(0, color="black", lw=0.8, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Patch Layer $\\ell^*$", fontsize=11)
        ax.set_ylabel("Final Domain Shift", fontsize=11)
        ax.set_xticks(layers[::2])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, ls="--")

    plt.suptitle("Artifact Test: Does L3 spike appear with random vectors?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_artifact_control.png"), bbox_inches="tight")
    plt.close()
    print("Saved 03_artifact_control.png")

    # ── 3. 30-pair grid (attn + mlp only, exclude rand for readability) ───────
    fig, axes = plt.subplots(6, 5, figsize=(20, 20), dpi=150)
    axes = axes.flatten()
    for i, pair in enumerate(pair_keys):
        ax = axes[i]
        for comp in ["attn", "mlp"]:
            ci = comps.index(comp)
            st = STYLES[comp]
            ax.plot(layers, shift_tensor[i, ci].numpy(),
                    color=st["color"], ls=st["ls"], lw=1.5, label=comp)
        ax.axhline(0, color="black", lw=0.8, alpha=0.3)
        ax.set_title(pair, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2)
        if i >= 25: ax.set_xlabel("Layer")
        if i % 5 == 0: ax.set_ylabel("Final Shift")

    handles = [plt.Line2D([0],[0], color=STYLES[c]["color"], ls=STYLES[c]["ls"], lw=2,
                           label=STYLES[c]["label"])
               for c in ["attn", "mlp"]]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.99),
               fontsize=13, ncol=2)
    plt.suptitle("Final Domain Shift — All 30 Pairs (RMS-normalised alpha)",
                 fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(plots_dir, "02_final_shift_grid.png"))
    plt.close()
    print("Saved 02_final_shift_grid.png")

    # ── 4. Numerical summary ─────────────────────────────────────────────────
    print("\n=== Key Statistics ===")
    for ci, comp in enumerate(comps):
        for layer in [3, 18, 19, 27]:
            vals = shift_tensor[:, ci, layer]
            pct_pos = (vals > 0).float().mean().item() * 100
            print(f"  {comp:12s} L{layer:2d}: mean={vals.mean():+7.2f}  "
                  f"std={vals.std():6.2f}  CV={abs(vals.std()/vals.mean().clamp(min=1e-6)):5.2f}  "
                  f"pos={pct_pos:.0f}%")

if __name__ == "__main__":
    main()
