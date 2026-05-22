"""
exp_utils.py — Shared utilities for Experiment 2 analysis scripts.

Provides:
  - LoRA adapter loading (safetensors → ΔW = B @ A)
  - Base model weight loading (from HF cache safetensor shards)
  - Module grouping (attention vs MLP, per-module)
  - Common constants
"""

import os
import json
import glob
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from safetensors import safe_open


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NUM_LAYERS = 28  # Default: Qwen2.5-7B. Override via set_num_layers() or config.

def set_num_layers(n: int):
    """Override NUM_LAYERS for different architectures."""
    global NUM_LAYERS
    NUM_LAYERS = n

# Attention modules (4)
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# MLP modules (3)
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]

ALL_MODULES = ATTN_MODULES + MLP_MODULES

# Module → group mapping
MODULE_GROUP = {}
for m in ATTN_MODULES:
    MODULE_GROUP[m] = "attn"
for m in MLP_MODULES:
    MODULE_GROUP[m] = "mlp"

# Default domains
DEFAULT_DOMAINS = ["cs", "eess", "math", "physics", "q-bio", "stat"]


# ─────────────────────────────────────────────────────────────────────────────
# LoRA key parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_lora_key(key: str) -> Optional[Tuple[int, str, str, str]]:
    """
    Parse a LoRA adapter safetensors key into (layer_idx, sublayer, module, ab).

    Example:
      'base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight'
      → (5, 'self_attn', 'q_proj', 'A')

      'base_model.model.model.layers.12.mlp.gate_proj.lora_B.weight'
      → (12, 'mlp', 'gate_proj', 'B')

    Returns None if the key doesn't match the expected pattern.
    """
    parts = key.split(".")
    try:
        layer_idx = int(parts[4])  # base_model.model.model.layers.<idx>
        sublayer = parts[5]         # self_attn or mlp
        module = parts[6]           # q_proj, k_proj, etc.
        ab = parts[7].split("_")[1].upper()  # lora_A → 'A', lora_B → 'B'
        return (layer_idx, sublayer, module, ab)
    except (IndexError, ValueError):
        return None


def base_model_key(layer_idx: int, sublayer: str, module: str) -> str:
    """
    Construct the base model weight key from layer info.
    
    Example: (5, 'self_attn', 'q_proj') → 'model.layers.5.self_attn.q_proj.weight'
    """
    return f"model.layers.{layer_idx}.{sublayer}.{module}.weight"


# ─────────────────────────────────────────────────────────────────────────────
# Load LoRA ΔW matrices
# ─────────────────────────────────────────────────────────────────────────────

def load_lora_deltas(
    adapter_dir: str,
    scaling: float = 1.0,
    device: str = "cpu",
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    Load a LoRA adapter and compute ΔW = (alpha/r) * B @ A for every module.

    Parameters
    ----------
    adapter_dir : str
        Path to the adapter directory (containing adapter_model.safetensors
        and adapter_config.json).
    scaling : float
        Override for LoRA scaling. If 1.0, will read alpha/r from config.
    device : str
        Device to load tensors onto.

    Returns
    -------
    dict mapping (layer_idx, module_name) → ΔW tensor on CPU
        e.g. {(0, 'q_proj'): Tensor(...), (0, 'k_proj'): Tensor(...), ...}
    """
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    config_path = os.path.join(adapter_dir, "adapter_config.json")

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")

    # Read LoRA config for scaling factor
    if scaling == 1.0 and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        r = cfg.get("r", 16)
        alpha = cfg.get("lora_alpha", 32)
        scaling = alpha / r

    # Load all A and B matrices, indexed by (layer, module)
    A_mats = {}  # (layer_idx, module) → A tensor
    B_mats = {}  # (layer_idx, module) → B tensor

    with safe_open(safetensors_path, framework="pt", device=device) as f:
        for key in f.keys():
            parsed = parse_lora_key(key)
            if parsed is None:
                continue
            layer_idx, sublayer, module, ab = parsed
            tensor = f.get_tensor(key)
            idx = (layer_idx, module)
            if ab == "A":
                A_mats[idx] = tensor
            else:
                B_mats[idx] = tensor

    # Compute ΔW = scaling * B @ A
    deltas = {}
    for idx in A_mats:
        if idx not in B_mats:
            continue
        A = A_mats[idx].float()  # (r, in_features)
        B = B_mats[idx].float()  # (out_features, r)
        delta_w = scaling * (B @ A)  # (out_features, in_features)
        deltas[idx] = delta_w.cpu()

    return deltas


# ─────────────────────────────────────────────────────────────────────────────
# Load base model weight norms
# ─────────────────────────────────────────────────────────────────────────────

def find_base_model_shards(model_name: str) -> List[str]:
    """
    Find the safetensor shard files for the base model in the HF cache.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_name.replace("/", "--")
    model_cache = os.path.join(cache_dir, model_dir_name, "snapshots")

    if not os.path.exists(model_cache):
        raise FileNotFoundError(
            f"Base model not found in HF cache: {model_cache}. "
            f"Make sure {model_name} is downloaded."
        )

    # Find the latest snapshot
    snapshots = sorted(os.listdir(model_cache))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {model_cache}")
    
    snapshot_dir = os.path.join(model_cache, snapshots[-1])
    shards = sorted(glob.glob(os.path.join(snapshot_dir, "model*.safetensors")))

    if not shards:
        raise FileNotFoundError(f"No safetensor shards in {snapshot_dir}")

    return shards


def load_base_weight_norms(
    model_name: str,
) -> Dict[Tuple[int, str], float]:
    """
    Load the Frobenius norm of each base model weight matrix for the target modules.

    Returns
    -------
    dict mapping (layer_idx, module_name) → ||W_base||_F (float)
    """
    shards = find_base_model_shards(model_name)
    norms = {}

    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Only look at the modules we care about
                for module in ALL_MODULES:
                    if f".{module}.weight" in key and "layers." in key:
                        # Extract layer index
                        parts = key.split(".")
                        try:
                            layer_idx = int(parts[2])  # model.layers.<idx>
                        except (IndexError, ValueError):
                            continue
                        
                        tensor = f.get_tensor(key).float()
                        norms[(layer_idx, module)] = tensor.norm().item()

    return norms


def load_base_weights(
    model_name: str,
    layers: Optional[List[int]] = None,
    modules: Optional[List[str]] = None,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    Load full base model weight tensors for specified layers/modules.
    WARNING: This is memory-intensive. Use sparingly.

    Parameters
    ----------
    model_name : str
    layers : list of int, optional
        Which layers to load (default: all)
    modules : list of str, optional
        Which modules to load (default: all target modules)

    Returns
    -------
    dict mapping (layer_idx, module_name) → W_base tensor
    """
    shards = find_base_model_shards(model_name)
    weights = {}
    target_modules = modules or ALL_MODULES
    target_layers = set(layers) if layers else set(range(NUM_LAYERS))

    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                for module in target_modules:
                    if f".{module}.weight" in key and "layers." in key:
                        parts = key.split(".")
                        try:
                            layer_idx = int(parts[2])
                        except (IndexError, ValueError):
                            continue
                        if layer_idx not in target_layers:
                            continue
                        weights[(layer_idx, module)] = f.get_tensor(key).float()

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Grouping helpers
# ─────────────────────────────────────────────────────────────────────────────

def group_by_layer_and_type(
    per_module: Dict[Tuple[int, str], float],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Reorganise a flat (layer, module) → value dict into:
      layer → { 'attn': {module: value, ...}, 'mlp': {module: value, ...} }
    """
    result = defaultdict(lambda: {"attn": {}, "mlp": {}})
    for (layer_idx, module), value in per_module.items():
        group = MODULE_GROUP.get(module, "unknown")
        result[layer_idx][group][module] = value
    return dict(result)


def save_json(data, path: str):
    """Save a dict/list to JSON with pretty printing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
