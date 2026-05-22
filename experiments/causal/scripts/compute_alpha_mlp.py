#!/usr/bin/env python3
"""
exp3_compute_alpha_mlp.py
Quick script to compute the projection magnitudes (alpha) for the MLP contributions
across all domains, needed for the final shift MLP patches.
"""
import argparse, os, torch
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
from hooks import ActivationStore

def compute_alpha_mlp(domain, df, model, tokenizer, store, args, global_mean_mlp, domain_unit_mlp):
    print(f"\nComputing alpha_mlp for {domain}...")
    dataset = tokenize_domain(df, tokenizer, max_length=args.max_length)
    loader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=4)
    
    proj_sum = torch.zeros(model.config.num_hidden_layers, dtype=torch.float64)
    total = 0
    
    for batch in tqdm(loader, desc=f"[alpha-mlp-{domain}]", dynamic_ncols=True):
        try:
            target_device = model.model.embed_tokens.weight.device
        except AttributeError:
            target_device = next(model.parameters()).device
            
        input_ids = batch["input_ids"].to(target_device)
        attn_mask = batch["attention_mask"].to(target_device)
        bs = input_ids.size(0)
        
        store.set_mask(attn_mask)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            
        results = store.get_batch_results()
        mlp_batch = results[f"mlp/{args.pooling}"].to(torch.float64) # B, L, d
        
        centred = mlp_batch - global_mean_mlp.unsqueeze(0)
        projections = (centred * domain_unit_mlp.unsqueeze(0)).sum(dim=2) # B, L
        proj_sum += projections.abs().sum(dim=0)
        total += bs
        
    return proj_sum / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    
    print("Loading model (device_map='auto')...")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    store = ActivationStore(num_layers=model.config.num_hidden_layers)
    store.register_hooks(model)
    
    domains = ["cs", "eess", "math", "physics", "q-bio", "stat"]
    print("Loading data...")
    domain_data = load_domain_data("./arxiv_data", 500, 42)
    
    vectors_dir = "./results/exp3_causal_alldomains"
    global_mean_mlp = torch.load(f"{vectors_dir}/global_mean_mlp.pt", weights_only=True)
    
    for domain in domains:
        domain_unit_mlp = torch.load(f"{vectors_dir}/{domain}/domain_vector_unit_mlp.pt", weights_only=True)
        alpha = compute_alpha_mlp(domain, domain_data[domain], model, tokenizer, store, args, global_mean_mlp, domain_unit_mlp)
        out_path = f"{vectors_dir}/{domain}/alpha_mlp_projection_magnitudes.pt"
        torch.save(alpha, out_path)
        print(f"✓ Saved alpha_mlp for {domain} ({alpha.mean():.4f} mean)")

if __name__ == "__main__":
    main()
