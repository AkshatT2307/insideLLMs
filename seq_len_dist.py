import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

def main():
    data_dir = "/home/psquare_a6000/Desktop/insideLLMs/arxiv_data"
    model_name = "Qwen/Qwen2.5-7B"
    
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    domain_files = {
        "cs": "cs.csv",
        "eess": "eess.csv",
        "math": "math.csv",
        "physics": "physics.csv",
        "q-bio": "q-bio.csv",
        "stat": "stat.csv",
    }
    
    print("\nSequence Length Distribution per Domain:\n" + "="*80)
    
    for domain, filename in domain_files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"[WARN] {filepath} not found.")
            continue
            
        df = pd.read_csv(filepath)
        df = df.dropna(subset=["text"])
        texts = df["text"].tolist()
        
        # Tokenize without truncation/padding to get true lengths
        # Fast tokenizer can process lists of strings efficiently
        encoded = tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        
        lengths = np.array(lengths)
        
        print(f"Domain: {domain.upper()} ({len(lengths):,} samples)")
        print(f"  Mean   : {lengths.mean():.1f}")
        print(f"  Std    : {lengths.std():.1f}")
        print(f"  Min    : {lengths.min()}")
        print(f"  25%    : {np.percentile(lengths, 25):.0f}")
        print(f"  Median : {np.percentile(lengths, 50):.0f}")
        print(f"  75%    : {np.percentile(lengths, 75):.0f}")
        print(f"  90%    : {np.percentile(lengths, 90):.0f}")
        print(f"  95%    : {np.percentile(lengths, 95):.0f}")
        print(f"  99%    : {np.percentile(lengths, 99):.0f}")
        print(f"  Max    : {lengths.max()}")
        print("-" * 80)

if __name__ == "__main__":
    main()

