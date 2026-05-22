#!/usr/bin/env python3
"""
calculate_domain_vectors.py

Calculates domain concept vectors from saved HDF5 activation files.
"""

import os
import glob
import h5py
import numpy as np

def calculate_mean_batched(dataset, batch_size=1000):
    """Calculates the mean of an HDF5 dataset over axis 0 in batches."""
    N = dataset.shape[0]
    shape_rest = dataset.shape[1:]
    total_sum = np.zeros(shape_rest, dtype=np.float64)
    
    for i in range(0, N, batch_size):
        chunk = dataset[i:i+batch_size].astype(np.float64)
        total_sum += np.sum(chunk, axis=0)
        
    return total_sum / N

def main():
    activations_dir = "activations"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    out_file = os.path.join(results_dir, "domain_vectors.h5")
    
    h5_files = glob.glob(os.path.join(activations_dir, "*_activations.h5"))
    if not h5_files:
        print("No activation files found.")
        return
        
    domains = [os.path.basename(f).replace("_activations.h5", "") for f in h5_files]
    print(f"Found {len(domains)} domains: {', '.join(domains)}")
    
    components = ["attn", "mlp", "layer"]
    modes = ["last", "mean"]
    
    # Structure to hold means: domain_means[domain][component][mode] = mean_array
    domain_means = {d: {c: {m: None for m in modes} for c in components} for d in domains}
    sample_counts = {d: 0 for d in domains}
    
    print("\n1. Calculating domain means...")
    for file_path, domain in zip(h5_files, domains):
        print(f"  Processing {domain}...")
        with h5py.File(file_path, "r") as f:
            N = int(f.attrs["num_samples"])
            sample_counts[domain] = N
            
            for comp in components:
                for mode in modes:
                    key = f"{comp}/{mode}"
                    if key in f:
                        print(f"    - {key}")
                        mean_arr = calculate_mean_batched(f[key])
                        domain_means[domain][comp][mode] = mean_arr
                    else:
                        print(f"    - {key} (MISSING)")
                        
    print("\n2. Calculating global means and domain vectors...")
    
    with h5py.File(out_file, "w") as out_f:
        # Save sample counts as an attribute
        out_f.attrs["domains"] = domains
        for d, count in sample_counts.items():
            out_f.attrs[f"num_samples_{d}"] = count
            
        for comp in components:
            for mode in modes:
                # Check if all domains have this component/mode
                valid_domains = [d for d in domains if domain_means[d][comp][mode] is not None]
                if len(valid_domains) != len(domains):
                    print(f"  Skipping {comp}/{mode} due to missing data in some domains.")
                    continue
                
                print(f"  Processing {comp}/{mode}...")
                
                # Calculate global mean
                total_samples = sum(sample_counts[d] for d in domains)
                global_mean = np.zeros_like(domain_means[domains[0]][comp][mode])
                
                for d in domains:
                    weight = sample_counts[d] / total_samples
                    global_mean += domain_means[d][comp][mode] * weight
                
                # Store global mean
                gm_key = f"global_mean/{comp}/{mode}"
                out_f.create_dataset(gm_key, data=global_mean.astype(np.float16))
                
                # Calculate and store domain vectors
                for d in domains:
                    dv = domain_means[d][comp][mode] - global_mean
                    dv_key = f"domain_vectors/{d}/{comp}/{mode}"
                    out_f.create_dataset(dv_key, data=dv.astype(np.float16))
                    
    print(f"\nDone! Vectors saved to {out_file}")

if __name__ == "__main__":
    main()
