#!/usr/bin/env python3
"""
prepare_data.py — Step 1: Prepare train/val/test splits per domain.

For each domain:
  1. Load CSV from arxiv_data/
  2. Tokenize all texts to measure token counts
  3. Sort by token count descending → take top N_total
  4. Shuffle deterministically, then split into train/val/test
  5. Save as HuggingFace Arrow datasets

Usage:
    python prepare_data.py                         # all domains
    python prepare_data.py --domains cs math       # specific domains
    python prepare_data.py --config custom.yaml    # custom config
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Add parent for utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, resolve_path, set_seed


# ── Domain → filename mapping ───────────────────────────────────────────────
DOMAIN_FILES = {
    "cs": "cs.csv",
    "eess": "eess.csv",
    "math": "math.csv",
    "physics": "physics.csv",
    "q-bio": "q-bio.csv",
    "stat": "stat.csv",
}


def parse_args():
    p = argparse.ArgumentParser(description="Prepare domain datasets.")
    p.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML."
    )
    p.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Process only these domains. Default: all from config.",
    )
    return p.parse_args()


def count_tokens_batch(texts: list, tokenizer, batch_size: int = 512) -> list:
    """Tokenize texts in batches and return token counts (without padding)."""
    counts = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk, padding=False, truncation=False, return_length=True
        )
        counts.extend(encoded["length"])
    return counts


def prepare_domain(
    domain: str,
    source_dir: str,
    output_dir: str,
    tokenizer,
    cfg_data: dict,
):
    """Prepare train/val/test splits for one domain."""
    filename = DOMAIN_FILES.get(domain)
    if filename is None:
        print(f"  [WARN] Unknown domain '{domain}' — skipping")
        return

    filepath = os.path.join(source_dir, filename)
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath} — skipping")
        return

    n_train = cfg_data["n_train"]
    n_val = cfg_data["n_val"]
    n_test = cfg_data["n_test"]
    n_total = n_train + n_val + n_test
    max_length = cfg_data["max_length"]
    seed = cfg_data["seed"]
    text_col = cfg_data["text_column"]

    print(f"\n{'─' * 60}")
    print(f"Domain: {domain}")
    print(f"{'─' * 60}")

    # ── Load CSV ─────────────────────────────────────────────────────────
    t0 = time.time()
    df = pd.read_csv(filepath)
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} samples in {time.time() - t0:.1f}s")

    # ── Count tokens ─────────────────────────────────────────────────────
    t0 = time.time()
    texts = df[text_col].tolist()
    token_counts = count_tokens_batch(texts, tokenizer)
    df["token_count"] = token_counts
    print(f"  Tokenized (for counting) in {time.time() - t0:.1f}s")

    # ── Sort by token count descending, take top N ───────────────────────
    df = df.sort_values("token_count", ascending=False).reset_index(drop=True)

    if len(df) < n_total:
        print(f"  [WARN] Only {len(df)} samples available, need {n_total}")
        n_total = len(df)
        # Proportional split
        ratio = n_train / (n_train + n_val + n_test)
        n_train = int(n_total * ratio)
        remaining = n_total - n_train
        n_val = remaining // 2
        n_test = remaining - n_val
        print(f"  Adjusted split: train={n_train}, val={n_val}, test={n_test}")

    df_top = df.head(n_total).copy()

    # ── Shuffle then split ───────────────────────────────────────────────
    df_top = df_top.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df_train = df_top.iloc[:n_train]
    df_val = df_top.iloc[n_train : n_train + n_val]
    df_test = df_top.iloc[n_train + n_val : n_train + n_val + n_test]

    # ── Print stats ──────────────────────────────────────────────────────
    for split_name, split_df in [
        ("train", df_train),
        ("val", df_val),
        ("test", df_test),
    ]:
        tc = split_df["token_count"]
        above_max = (tc >= max_length).sum()
        print(
            f"  {split_name:5s}: n={len(split_df):,}  "
            f"tokens — min={tc.min()}, median={tc.median():.0f}, "
            f"max={tc.max()}, mean={tc.mean():.0f}, "
            f"≥{max_length}: {above_max} ({100*above_max/len(split_df):.1f}%)"
        )

    # ── Save as Arrow datasets ───────────────────────────────────────────
    domain_dir = os.path.join(output_dir, domain)
    for split_name, split_df in [
        ("train", df_train),
        ("val", df_val),
        ("test", df_test),
    ]:
        split_dir = os.path.join(domain_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Keep only relevant columns
        ds = Dataset.from_pandas(
            split_df[[text_col, "token_count"]].reset_index(drop=True)
        )
        ds.save_to_disk(split_dir)
        print(f"  Saved: {split_dir}")


def main():
    args = parse_args()

    # Resolve config path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = resolve_path(args.config, script_dir)
    cfg = load_config(config_path)

    set_seed(cfg["data"]["seed"])

    # Resolve data paths
    source_dir = resolve_path(cfg["data"]["source_dir"], script_dir)
    output_dir = resolve_path(cfg["data"]["output_dir"], script_dir)

    domains = args.domains or cfg["data"]["domains"]

    print(f"\n{'=' * 60}")
    print(f"Data Preparation")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Domains: {domains}")
    print(f"  Split: train={cfg['data']['n_train']}, "
          f"val={cfg['data']['n_val']}, test={cfg['data']['n_test']}")
    print(f"  max_length: {cfg['data']['max_length']}")
    print(f"{'=' * 60}")

    # ── Load tokenizer (just for counting, no model needed) ──────────────
    print(f"\nLoading tokenizer: {cfg['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"], trust_remote_code=True
    )

    # ── Process each domain ──────────────────────────────────────────────
    t_start = time.time()
    for domain in domains:
        prepare_domain(
            domain=domain,
            source_dir=source_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            cfg_data=cfg["data"],
        )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"All done! {len(domains)} domains prepared in {elapsed:.1f}s")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
