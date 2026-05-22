"""
config.py — Paths, domain list, model constants.
"""
import os

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVATIONS_DIR = os.path.join(PROJECT_ROOT, "activations")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── Domains ──
DOMAINS = ["cs", "eess", "math", "physics", "q-bio", "stat"]

# ── Model constants (Qwen2.5-7B) ──
NUM_LAYERS = 28
HIDDEN_DIM = 3584

# ── Components stored in HDF5 ──
COMPONENTS = ["attn", "mlp", "layer"]

# ── Pooling mode ──
MODE = "last"  # primary analysis on last-token

# ── Random seed ──
SEED = 42
