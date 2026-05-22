# Detailed Hypothesis Explanation

## Setup: What We're Measuring

We have a 7B-parameter Qwen model processing ~90K arXiv abstracts from 6 scientific domains (cs, math, physics, stat, q-bio, eess). For each abstract, we extract the internal activations at every layer ℓ ∈ {0, ..., 27}.

At each layer, the transformer computes:

```
x^ℓ = x^{ℓ-1} + Δa^ℓ + Δm^ℓ
       ↑              ↑        ↑
   residual     attention    MLP
   stream       contribution contribution
```

We extract **three separate vectors** per layer:
- `layer` = x^ℓ — the full residual stream (everything accumulated so far)
- `attn` = Δa^ℓ — what the attention sublayer at this layer *added*
- `mlp` = Δm^ℓ — what the MLP sublayer at this layer *added*

We do this in two pooling modes:
- `mean` — average across all token positions in the sequence
- `last` — take only the last token position

---

## The Five Quantities We Computed

### 1. Between-Class Scatter — tr(S_B)

```
tr(S_B)^ℓ = Σ_k  n_k  ||d_k^ℓ||²
```

**What it is**: For each domain k, compute the mean activation vector μ_k, then compute how far it is from the global mean: d_k = μ_k − μ. Square its norm, weight by sample count n_k, sum over all 6 domains.

**What it measures**: How **spread apart** the domain means are. If all 6 domain means are in the same location → tr(S_B) ≈ 0. If they're far apart → tr(S_B) is large.

**Intuition**: This is the **signal** — how much the activations differ *between* domains.

### 2. Within-Class Scatter — tr(S_W)

```
tr(S_W)^ℓ = Σ_k  Σ_{x ∈ D_k}  ||x^ℓ − μ_k^ℓ||²
```

**What it is**: For each sample x in domain k, compute how far it is from its own domain's mean μ_k. Square, sum over all samples in all domains.

**What it measures**: How **spread out** the samples are *within* each domain cluster. Large tr(S_W) means samples within the same domain are all over the place.

**Intuition**: This is the **noise** — how much variation exists that is NOT domain-related.

### 3. Fisher Discriminant Ratio — FDR = tr(S_B) / tr(S_W)

**What it is**: The ratio of signal to noise.

**What it measures**: How well-separated the domain clusters are, relative to how tight they are internally. High FDR = well-separated, low FDR = overlapping clusters.

**Key property**: FDR is **scale-invariant**. If you multiply all activations by a constant c, both S_B and S_W scale by c², so FDR stays the same. This means FDR already factors out the overall magnitude of activations.

### 4. Accurate FDR — tr(Ŝ_W⁻¹ · S_B)

**What it is**: Instead of just dividing traces, this computes the full matrix inverse of the regularized within-class scatter and multiplies it with the between-class scatter matrix. It accounts for correlations between dimensions (the basic FDR treats all dimensions independently).

**What it measures**: Separability in the most discriminative *directions*, not just total magnitude. A domain might be perfectly separable along one direction but noisy in all others — the accurate FDR captures this.

### 5. OvA FDR — Per-Class One-vs-All Scores

**What it is**: For each domain k separately, compute how well separated it is from all other domains combined, using the pooled within-class covariance.

**What it measures**: Which specific domains are driving the separability signal, and at which layers.

---

## What Each Plot Shows

### Plot A: Basic FDR Curves (3 curves × 2 modes)

| Config | What we see |
|--------|-------------|
| mean-pool | Attn, MLP, and Layer all fluctuate around FDR ≈ 0.15–0.35, with similar variability |
| last-token | All three rise with depth; MLP and Layer peak later (L23–24) than Attn (L22) |

**Takeaway**: At the FDR level (signal/noise ratio), **attention and MLP perform comparably**. Neither has dramatically sharper or flatter peaks. This refutes the original hypothesis.

### Plot B: Log-Scale S_B and S_W Decomposition

| Config | What we see |
|--------|-------------|
| S_B mean-pool | MLP's between-class scatter grows from ~10⁴ to ~10⁸ (4 orders of magnitude!). Attention stays at ~10⁴–10⁷. Full residual is largest. |
| S_W mean-pool | Same exponential growth pattern — MLP noise also goes from ~10⁵ to ~10⁹ |
| last-token | Same pattern, even more extreme |

**Takeaway**: MLP activations have **exponentially growing magnitudes** with depth. Both signal AND noise grow together, which is why the FDR ratio stays roughly constant. The raw numbers are vastly different but the ratio is similar.

### Plot C: Attn/MLP Ratio Decomposition (the key plot)

This plots two ratios on a log scale:
- **Red line**: tr(S_B^attn) / tr(S_B^mlp) — "How much between-class signal does attention contribute vs MLP?"
- **Teal line**: tr(S_W^attn) / tr(S_W^mlp) — "How much within-class noise does attention contribute vs MLP?"

| Config | What we see |
|--------|-------------|
| mean-pool | Both ratios start near 1.0 at layer 0, then **drop monotonically** to ~0.1 by layer 27 |
| last-token | Same monotonic decline from ~1.0 to ~0.05 |

**What this means**: At every layer, attention is contributing a **smaller and smaller fraction** of both signal and noise relative to MLP. By layer 27, MLP's signal is ~10× attention's signal, and MLP's noise is ~10× attention's noise.

### Plot D: Signal vs Noise Share (the "killer" plot)

This plots attention's **share** of the total (attn + mlp):
- **Red line**: attn_S_B / (attn_S_B + mlp_S_B) — attention's share of **signal**
- **Teal line**: attn_S_W / (attn_S_W + mlp_S_W) — attention's share of **noise**
- **Purple shading**: Where red > teal (attention has better signal-to-noise ratio than MLP)

| Config | What we see |
|--------|-------------|
| mean-pool | In layers 0–8, red is above teal (attention gets ~50–70% of signal but only ~40–50% of noise). In layers 20+, both converge to ~10–15%. |
| last-token | Same pattern, more pronounced. Layers 0–6: attention has clear signal advantage. |

**What this means**: In early layers, attention contributes a **disproportionately large share of signal relative to its noise** — it's a more efficient discriminator. In late layers, MLP dominates everything by sheer magnitude.

### Plot E: Localization Ratio Heatmap

For each domain k and layer ℓ: ρ_k = OvA_attn(k,ℓ) / OvA_mlp(k,ℓ)

| Color | Meaning |
|-------|---------|
| Red (ρ > 1) | Attention dominates for this domain at this layer |
| Blue (ρ < 1) | MLP dominates for this domain at this layer |

| Config | What we see |
|--------|-------------|
| mean-pool | Almost entirely blue. MLP dominates across all domains and layers. A few warm spots at layers 0–2 and 13–17. |
| last-token | Clear **left-red, right-blue gradient** across ALL 6 domains. Layers 0–6 are warm (attention), layers 16+ are cold (MLP). The handoff is universal. |

**What this means**: The handoff pattern is **not domain-specific** — it happens for cs, math, physics, stat, q-bio, and eess at the same layers. It's a fundamental property of how the transformer processes information.

---

## The Hypothesis, Precisely Stated

> **"In a decoder-only transformer, attention and MLP sublayers contribute to domain separability through distinct mechanisms: attention sublayers act as efficient, low-magnitude selectors that achieve a favorable signal-to-noise ratio in early layers, while MLP sublayers act as high-magnitude amplifiers whose exponentially growing activation norms carry domain signal through scale rather than discriminative precision. This produces a monotonic attention→MLP handoff in separability, where attention's share of both between-class signal and within-class noise declines steadily with depth."**

### Breaking this down into testable claims:

| Claim | Supporting evidence | Configs where it holds |
|-------|--------------------|-----------------------|
| 1. MLP activation norms grow exponentially with depth | Log-scale S_B and S_W plots: both grow ~4 orders of magnitude L0→L27 | All 3 components × 2 modes |
| 2. Attention activation norms grow much more slowly | Same plots: attention stays ~2 orders of magnitude growth | All configs |
| 3. FDR (signal/noise ratio) is comparable for attn and MLP | Basic FDR curves: CV(attn) ≈ CV(mlp), similar magnitudes | All 6 metric×mode combos |
| 4. Attention has a better signal-to-noise ratio in early layers | Signal-vs-noise share plot: red > teal in layers 0–8 | Both modes, more pronounced in last-token |
| 5. MLP dominates by scale, not efficiency | FDR is scale-invariant yet attn/MLP FDR ratio ≈ 1, despite MLP having 10–100× larger magnitudes | All configs |
| 6. The handoff is monotonic | Attn/MLP ratio plots: monotonic decline from ~1.0 to ~0.1 | Both modes |
| 7. The handoff is domain-universal | OvA heatmap: same red→blue gradient for all 6 domains | All 6 domains, clearest in last-token |

### What the running experiment (norm-normalized FDR) will test:

If we **remove norm entirely** (project all activations onto the unit sphere), then:
- If **attention's normalized FDR > MLP's normalized FDR** at most layers → confirms that attention is **directionally** more discriminative and MLP wins purely via scale
- If they're **equal after normalization** → both are equally discriminative per unit of activation, and MLP's advantage is entirely a scale artifact
- If **MLP's normalized FDR > attention's** → MLP is genuinely more discriminative even directionally, and the scale explanation is insufficient

---

## Why This Would Matter for the Literature

### What's currently known:
1. **MLPs as key-value memories** (Geva et al., 2021): MLPs store factual associations. But this says nothing about whether they're *efficient* discriminators or just *large-scale* carriers.
2. **Attention routes information** (Elhage et al., 2021): Attention heads specialize to move information between positions. But the signal-vs-noise efficiency angle is new.
3. **Representation grows with depth** (logit lens, tuned lens literature): Later layers produce better predictions. But the decomposition into "who contributes what" to class separation hasn't been done with Fisher statistics.

### What this adds:
- **Quantitative mechanism**: Not just "MLPs store knowledge" but "MLPs dominate because their activations are 10–100× larger in norm, carrying domain signal at scale"
- **The efficiency paradox**: Attention is actually a *more efficient* discriminator per unit of activation energy, but it operates at such small magnitude that MLP overwhelms it
- **Universal handoff**: The transition happens at the same layers for all domains, suggesting it's an architectural property, not task-specific

### Potential paper framing:
> *"Are MLPs better at domain encoding, or just louder? A Fisher-discriminant decomposition reveals that attention sublayers achieve higher signal-to-noise efficiency in early transformer layers, but MLP sublayers dominate total separability through exponential activation norm growth — a scale effect, not a discriminative one."*
