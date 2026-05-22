#!/usr/bin/env python3
"""
evaluate_baseline.py — Step 2: Evaluate pretrained model on validation sets.

Loads the base model (no LoRA adapters) and evaluates cross-entropy loss
and perplexity on each domain's validation set.

Usage:
    python evaluate_baseline.py                        # all domains
    python evaluate_baseline.py --domains cs math      # specific domains
    python evaluate_baseline.py --config custom.yaml
"""

import argparse
import os
import sys
import time

import torch
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_config,
    resolve_path,
    set_seed,
    load_model_and_tokenizer,
    create_dataloader,
    evaluate_perplexity,
    save_json,
    get_device,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate baseline model.")
    p.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML."
    )
    p.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Evaluate only these domains. Default: all from config.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Which split to evaluate on.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = resolve_path(args.config, script_dir)
    cfg = load_config(config_path)

    set_seed(cfg["data"]["seed"])

    dataset_dir = resolve_path(cfg["data"]["output_dir"], script_dir)
    log_dir = resolve_path(cfg["paths"]["log_dir"], script_dir)
    domains = args.domains or cfg["data"]["domains"]

    print(f"\n{'=' * 60}")
    print(f"Baseline Evaluation")
    print(f"  Split: {args.split}")
    print(f"  Domains: {domains}")
    print(f"  Batch size: {cfg['training']['eval_batch_size']}")
    print(f"{'=' * 60}")

    # ── Load model ───────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg, for_training=False)
    device = get_device(cfg)

    # Wrap in DataParallel for multi-GPU eval
    if cfg["gpu"]["multi_gpu"] and torch.cuda.device_count() > 1:
        print(f"  Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # ── Evaluate each domain ─────────────────────────────────────────────
    results = {}
    t_start = time.time()

    for domain in domains:
        split_path = os.path.join(dataset_dir, domain, args.split)
        if not os.path.exists(split_path):
            print(f"\n  [WARN] {split_path} not found — skipping '{domain}'")
            continue

        print(f"\n{'─' * 60}")
        print(f"Evaluating: {domain} ({args.split})")
        print(f"{'─' * 60}")

        # Load dataset
        ds = load_from_disk(split_path)
        texts = ds["text"]
        print(f"  Loaded {len(texts)} samples")

        # Create dataloader
        dataloader = create_dataloader(
            texts=texts,
            tokenizer=tokenizer,
            max_length=cfg["data"]["max_length"],
            batch_size=cfg["training"]["eval_batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["dataloader_num_workers"],
        )

        # Evaluate
        t0 = time.time()
        metrics = evaluate_perplexity(model, dataloader, device)
        elapsed = time.time() - t0

        metrics["num_samples"] = len(texts)
        metrics["elapsed_seconds"] = round(elapsed, 2)

        results[domain] = metrics
        print(
            f"  Loss: {metrics['loss']:.4f}  |  "
            f"Perplexity: {metrics['perplexity']:.2f}  |  "
            f"Time: {elapsed:.1f}s"
        )

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{'Domain':<10} {'Loss':>10} {'Perplexity':>12} {'Samples':>10}")
    print(f"{'─' * 42}")
    for domain, m in results.items():
        print(
            f"{domain:<10} {m['loss']:>10.4f} {m['perplexity']:>12.2f} "
            f"{m['num_samples']:>10}"
        )
    print(f"{'=' * 60}")

    total_time = time.time() - t_start
    print(f"Total evaluation time: {total_time:.1f}s\n")

    # ── Save results ─────────────────────────────────────────────────────
    results_path = os.path.join(log_dir, f"baseline_{args.split}_results.json")
    save_json(results, results_path)


if __name__ == "__main__":
    main()
