import torch, json
import numpy as np
res = torch.load("./results/exp3_causal_alldomains/final_shift/final_shift_scores.pt", weights_only=True)
with open("./results/exp3_causal_alldomains/final_shift/final_shift_results.json") as f:
    summ = json.load(f)
pairs = summ["pair_keys"]
mean_attn = res[:, 0, :].mean(dim=0).numpy()
mean_mlp = res[:, 1, :].mean(dim=0).numpy()

print(f"Attn Peak: {mean_attn.max():.2f} at layer {mean_attn.argmax()}")
print(f"MLP Peak:  {mean_mlp.max():.2f} at layer {mean_mlp.argmax()}")

for i in range(28):
    if mean_attn[i] > 3.0 or mean_mlp[i] > 3.0:
        print(f"L{i:02d}: Attn={mean_attn[i]:.2f}, MLP={mean_mlp[i]:.2f}")
