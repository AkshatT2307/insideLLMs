"""
utils.py — Shared utilities for the LoRA fine-tuning pipeline.

Provides:
  - Config loading from YAML
  - Model & tokenizer loading (single-GPU or DataParallel)
  - Seed setting for reproducibility
  - Tokenization collator for causal LM training
  - Perplexity / loss evaluation function
  - DataLoader creation from HuggingFace datasets
"""

import os
import math
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import shared model slug utility
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
try:
    from utils.model_loader import get_model_slug
except ImportError:
    import re
    def get_model_slug(model_name: str) -> str:
        slug = model_name.split("/")[-1].lower()
        slug = re.sub(r'[^a-z0-9._-]', '-', slug)
        return slug


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(path: str, base_dir: str) -> str:
    """Resolve a potentially relative path against a base directory."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(base_dir) / p
    return str(p.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops for reproducibility (minor perf cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Model & Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(cfg: dict, for_training: bool = False):
    """
    Load model and tokenizer from config.

    Parameters
    ----------
    cfg : dict
        Full config dict.
    for_training : bool
        If True, enables gradient checkpointing and does NOT wrap in
        DataParallel (LoRA + DP handled separately in finetune script).
        If False (eval mode), optionally wraps in DataParallel.

    Returns
    -------
    model, tokenizer
    """
    model_cfg = cfg["model"]
    gpu_cfg = cfg["gpu"]

    torch_dtype = DTYPE_MAP[model_cfg["dtype"]]

    print(f"\n{'=' * 60}")
    print(f"Loading model: {model_cfg['name']}")
    print(f"  dtype  : {model_cfg['dtype']}")
    print(f"  multi_gpu: {gpu_cfg['multi_gpu']}")
    print(f"{'=' * 60}\n")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], trust_remote_code=True
    )
    tokenizer.padding_side = "right"  # standard for causal LM training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ────────────────────────────────────────────────────────────
    device = gpu_cfg["device"]

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        dtype=torch_dtype,
        device_map=device if not gpu_cfg["multi_gpu"] else None,
        trust_remote_code=True,
    )

    if gpu_cfg["multi_gpu"]:
        # Move to primary GPU first, then wrap
        model = model.to("cuda:0")

    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Model loaded — {num_layers} layers, d={hidden_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class CausalLMDataset(Dataset):
    """
    Dataset for causal language modeling.
    Pre-tokenizes texts and creates input_ids / attention_mask / labels.
    Labels have padding tokens set to -100 so they are ignored in loss.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int,
    ):
        # Ensure texts is a plain Python list of strings
        # (HuggingFace datasets may return Arrow sequences)
        if not isinstance(texts, list):
            texts = list(texts)
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Labels = input_ids, but with padding masked to -100
        self.labels = self.encodings["input_ids"].clone()
        self.labels[self.encodings["attention_mask"] == 0] = -100

    def __len__(self) -> int:
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def create_dataloader(
    texts: List[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader from a list of texts."""
    dataset = CausalLMDataset(texts, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute average cross-entropy loss and perplexity on a dataloader.

    Uses the model's built-in loss computation (shifted labels internally).
    Padding tokens are masked via labels=-100.

    Returns
    -------
    dict with keys: "loss", "perplexity", "num_samples"
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc="  [eval]", unit="batch", dynamic_ncols=True, leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # If model is wrapped in DataParallel, it handles device placement
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Count non-padding tokens in labels (tokens that contribute to loss)
        # The model internally shifts labels, so valid tokens = labels != -100
        # minus the first position (which has no prediction target)
        n_valid = (labels[:, 1:] != -100).sum().item()

        # outputs.loss is mean over valid tokens in the batch
        loss = outputs.loss
        if loss.dim() > 0:  # DataParallel returns per-GPU losses
            loss = loss.mean()
        total_loss += loss.item() * n_valid
        total_tokens += n_valid
        
        # Update progress bar with running metrics
        run_loss = total_loss / max(total_tokens, 1)
        run_ppl = math.exp(min(run_loss, 100))
        pbar.set_postfix({"loss": f"{run_loss:.4f}", "ppl": f"{run_ppl:.2f}"})

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # cap to avoid overflow

    return {
        "loss": round(avg_loss, 6),
        "perplexity": round(perplexity, 4),
        "num_tokens": total_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def save_json(data: Any, filepath: str):
    """Save data as formatted JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {filepath}")


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_device(cfg: dict) -> torch.device:
    """Get the primary device from config."""
    if cfg["gpu"]["multi_gpu"]:
        return torch.device("cuda:0")
    return torch.device(cfg["gpu"]["device"])


def resolve_results_dir(cfg: dict, script_dir: str) -> str:
    """
    Resolve the model-specific results directory.

    Returns
    -------
    str : e.g. /path/to/experiments/finetuning/results/qwen2.5-7b
    """
    results_base = resolve_path(cfg.get("results_dir", "../results"), script_dir)
    model_slug = get_model_slug(cfg["model"]["name"])
    results_dir = os.path.join(results_base, model_slug)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
