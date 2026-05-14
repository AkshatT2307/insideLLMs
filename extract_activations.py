#!/usr/bin/env python3
"""
extract_activations.py — Main CLI for extracting LLM activations.

Extracts per-layer activations (attention, MLP, full layer) in both
pooling modes (last-token, mean) from a Qwen-family model and saves
them as one HDF5 file per domain.

Example
-------
    python extract_activations.py \
        --model Qwen/Qwen2.5-7B \
        --data-dir ./arxiv_data \
        --output-dir ./activations \
        --max-length 512 \
        --batch-size 32 \
        --samples-per-domain 3500 \
        --dtype bfloat16
"""

import argparse
import os
import sys
import time
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import load_domain_data, tokenize_domain, make_dataloader
from hooks import ActivationStore
from storage import create_hdf5, write_batch, close_hdf5


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-layer LLM activations for arXiv abstracts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="HuggingFace model identifier (must be a Qwen / LLaMA-style decoder).",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing per-domain CSV files (cs.csv, math.csv, …).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./activations",
        help="Directory where per-domain HDF5 files will be saved.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Fixed tokenizer sequence length (truncate / pad to this).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (tune for your GPU memory).",
    )
    p.add_argument(
        "--samples-per-domain",
        type=int,
        default=None,
        help=(
            "Max samples to use per domain. None → use all available. "
            "E.g. 3500 with 6 domains ≈ 21k total."
        ),
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map for model loading ('auto', 'cuda:0', etc.).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible domain sampling.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    p.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Process only these domains (space-separated). "
            "E.g. --domains cs math stat. None → all domains."
        ),
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# model + tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(
    model_name: str,
    dtype_str: str,
    device_map: str,
):
    """Load model in eval mode and configure tokenizer for left-padding."""

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_str]

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"  dtype  : {dtype_str}")
    print(f"  device : {device_map}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ── ensure left-padding (critical for last-token extraction) ─────────
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # Qwen models typically don't set a pad token; use eos as pad
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    # ── read architectural constants from config ─────────────────────────
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    print(f"  Model loaded — {num_layers} layers, d={hidden_dim}\n")

    return model, tokenizer, num_layers, hidden_dim


# ─────────────────────────────────────────────────────────────────────────────
# main extraction loop
# ─────────────────────────────────────────────────────────────────────────────
def extract_for_domain(
    domain: str,
    df,
    model,
    tokenizer,
    store: ActivationStore,
    args: argparse.Namespace,
    num_layers: int,
    hidden_dim: int,
):
    """Run inference over one domain and write activations to HDF5."""

    print(f"\n{'─'*60}")
    print(f"Processing domain: {domain}  ({len(df):,} samples)")
    print(f"{'─'*60}")

    # ── tokenize ─────────────────────────────────────────────────────────
    t0 = time.time()
    dataset = tokenize_domain(df, tokenizer, max_length=args.max_length)
    print(f"  Tokenized in {time.time() - t0:.1f}s")

    loader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── create HDF5 output file ──────────────────────────────────────────
    filepath = os.path.join(args.output_dir, f"{domain}_activations.h5")
    hf = create_hdf5(
        filepath=filepath,
        num_samples=len(dataset),
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        domain=domain,
        model_name=args.model,
        max_length=args.max_length,
    )
    print(f"  HDF5 file created: {filepath}")

    # ── inference + write ────────────────────────────────────────────────
    offset = 0
    t0 = time.time()

    pbar = tqdm(
        loader,
        desc=f"  [{domain}]",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in pbar:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        indices = batch["index"].tolist()

        # Determine the device to send inputs to.
        # With device_map="auto", model.device may not exist; use the
        # embedding layer's device which is where input_ids must land.
        try:
            target_device = model.model.embed_tokens.weight.device
        except AttributeError:
            target_device = next(model.parameters()).device

        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device)

        # Set the mask *before* the forward pass so hooks can access it
        store.set_mask(attention_mask)

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )

        # Retrieve pooled results (on CPU, float16)
        results = store.get_batch_results()

        # Write to HDF5
        offset = write_batch(hf, results, indices, offset)

        # Free GPU memory for this batch
        del input_ids, attention_mask, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.set_postfix({"written": f"{offset:,}/{len(dataset):,}"})

    close_hdf5(hf)
    elapsed = time.time() - t0
    throughput = len(dataset) / elapsed if elapsed > 0 else 0
    print(
        f"  ✓ Domain '{domain}' done — {len(dataset):,} samples, "
        f"{elapsed:.1f}s ({throughput:.0f} samples/s)"
    )
    print(f"  Saved to: {filepath}\n")


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── load model ───────────────────────────────────────────────────────
    model, tokenizer, num_layers, hidden_dim = load_model_and_tokenizer(
        args.model, args.dtype, args.device,
    )

    # ── register hooks ───────────────────────────────────────────────────
    store = ActivationStore(num_layers=num_layers)
    store.register_hooks(model)

    # ── load data ────────────────────────────────────────────────────────
    print("Loading dataset …")
    domain_data = load_domain_data(
        data_dir=args.data_dir,
        samples_per_domain=args.samples_per_domain,
        seed=args.seed,
    )

    # optionally filter to requested domains
    if args.domains:
        domain_data = {
            k: v for k, v in domain_data.items() if k in args.domains
        }
        if not domain_data:
            print(f"[ERROR] No matching domains found for: {args.domains}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── extract per domain ───────────────────────────────────────────────
    total_samples = sum(len(v) for v in domain_data.values())
    print(f"Starting extraction — {total_samples:,} samples across {len(domain_data)} domains\n")

    t_start = time.time()
    for domain, df in domain_data.items():
        extract_for_domain(
            domain=domain,
            df=df,
            model=model,
            tokenizer=tokenizer,
            store=store,
            args=args,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )

    store.remove_hooks()

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All done!  {total_samples:,} samples in {elapsed_total:.1f}s")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
