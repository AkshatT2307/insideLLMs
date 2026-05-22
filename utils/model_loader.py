"""
model_loader.py — Unified, model-agnostic model loading and architecture detection.

Supports Qwen, Llama, and Pythia model families.
The model name in config.yaml determines architecture-specific access paths.

Usage:
    from utils.model_loader import load_model_from_config, get_model_slug, get_layers

    cfg = yaml.safe_load(open("config.yaml"))
    model, tokenizer, model_info = load_model_from_config(cfg)
    layers = get_layers(model, model_info)
"""
import os
import re
from typing import Dict, Tuple, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture registry
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "qwen": {
        "layers_attr": ["model", "layers"],
        "embed_attr": ["model", "embed_tokens"],
        "attn_attr": "self_attn",
        "mlp_attr": "mlp",
    },
    "llama": {
        "layers_attr": ["model", "layers"],
        "embed_attr": ["model", "embed_tokens"],
        "attn_attr": "self_attn",
        "mlp_attr": "mlp",
    },
    "mistral": {
        "layers_attr": ["model", "layers"],
        "embed_attr": ["model", "embed_tokens"],
        "attn_attr": "self_attn",
        "mlp_attr": "mlp",
    },
    "pythia": {
        "layers_attr": ["gpt_neox", "layers"],
        "embed_attr": ["gpt_neox", "embed_in"],
        "attn_attr": "attention",
        "mlp_attr": "mlp",
    },
    "gpt_neox": {
        "layers_attr": ["gpt_neox", "layers"],
        "embed_attr": ["gpt_neox", "embed_in"],
        "attn_attr": "attention",
        "mlp_attr": "mlp",
    },
}


def _detect_family(model_name: str) -> str:
    """Detect model family from HuggingFace model name."""
    name_lower = model_name.lower()
    for family in MODEL_REGISTRY:
        if family in name_lower:
            return family
    # Fallback: try llama-style (most common architecture)
    print(f"  [model_loader] Unknown family for '{model_name}', assuming llama-style architecture")
    return "llama"


def _resolve_attr(obj: Any, attr_path: list) -> Any:
    """Resolve a dotted attribute path on an object."""
    for attr in attr_path:
        obj = getattr(obj, attr)
    return obj


def get_model_slug(model_name: str) -> str:
    """
    Convert HuggingFace model name to a filesystem-safe slug.

    Examples:
        'Qwen/Qwen2.5-7B'          → 'qwen2.5-7b'
        'meta-llama/Llama-3.1-8B'   → 'llama-3.1-8b'
        'EleutherAI/pythia-6.9b'    → 'pythia-6.9b'
    """
    # Take the part after the last /
    slug = model_name.split("/")[-1].lower()
    # Replace any problematic characters
    slug = re.sub(r'[^a-z0-9._-]', '-', slug)
    return slug


def load_model_from_config(
    cfg: dict,
    for_training: bool = False,
) -> Tuple[Any, Any, Dict]:
    """
    Load model and tokenizer from a config dict.

    Parameters
    ----------
    cfg : dict
        Must contain cfg["model"]["name"] and optionally:
        - cfg["model"]["dtype"] (default: "bfloat16")
        - cfg["model"]["device"] (default: "auto")
    for_training : bool
        If False, sets model.eval() and disables gradients.

    Returns
    -------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    model_info : dict
        Keys: 'name', 'slug', 'family', 'num_layers', 'hidden_dim',
              'arch_config' (from MODEL_REGISTRY)
    """
    model_name = cfg["model"]["name"]
    dtype_str = cfg["model"].get("dtype", "bfloat16")
    device_map = cfg["model"].get("device", "auto")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    family = _detect_family(model_name)
    slug = get_model_slug(model_name)

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"  Family : {family}")
    print(f"  Slug   : {slug}")
    print(f"  dtype  : {dtype_str}")
    print(f"  device : {device_map}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if not for_training:
        model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size

    print(f"  Model loaded — {num_layers} layers, d={hidden_dim}\n")

    model_info = {
        "name": model_name,
        "slug": slug,
        "family": family,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "arch_config": MODEL_REGISTRY[family],
    }

    return model, tokenizer, model_info


def get_layers(model: Any, model_info: Dict) -> Any:
    """Get the transformer layer list from a model."""
    attr_path = model_info["arch_config"]["layers_attr"]
    return _resolve_attr(model, attr_path)


def get_embed_tokens(model: Any, model_info: Dict) -> Any:
    """Get the embedding layer from a model."""
    attr_path = model_info["arch_config"]["embed_attr"]
    return _resolve_attr(model, attr_path)


def get_embed_device(model: Any, model_info: Dict) -> torch.device:
    """Get the device of the embedding layer (useful for multi-GPU)."""
    try:
        embed = get_embed_tokens(model, model_info)
        return embed.weight.device
    except AttributeError:
        return next(model.parameters()).device


def get_attn_module(layer, model_info: Dict):
    """Get the self-attention submodule from a transformer layer."""
    return getattr(layer, model_info["arch_config"]["attn_attr"])


def get_mlp_module(layer, model_info: Dict):
    """Get the MLP submodule from a transformer layer."""
    return getattr(layer, model_info["arch_config"]["mlp_attr"])
