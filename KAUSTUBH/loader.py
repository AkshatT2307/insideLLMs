"""
loader.py — HDF5 loading utilities for activation files.
"""
import os
import h5py
import numpy as np
from config import ACTIVATIONS_DIR, DOMAINS, COMPONENTS, MODE


def h5_path(domain: str) -> str:
    """Return the full path to a domain's HDF5 activation file."""
    return os.path.join(ACTIVATIONS_DIR, f"{domain}_activations.h5")


def get_num_samples(domain: str) -> int:
    """Return the number of samples in a domain's file."""
    with h5py.File(h5_path(domain), "r") as f:
        return int(f.attrs["num_samples"])


def load_component(domain: str, component: str, mode: str = MODE,
                   sample_slice=None) -> np.ndarray:
    """
    Load activations for one domain, one component.

    Parameters
    ----------
    domain : str
        One of DOMAINS.
    component : str
        One of 'attn', 'mlp', 'layer'.
    mode : str
        'last' or 'mean'.
    sample_slice : slice or None
        Which samples to load. None = all.

    Returns
    -------
    np.ndarray, shape (N, L, d), dtype float32
    """
    key = f"{component}/{mode}"
    with h5py.File(h5_path(domain), "r") as f:
        if sample_slice is None:
            data = f[key][:]
        else:
            data = f[key][sample_slice]
    return data.astype(np.float32)


def load_layer(domain: str, component: str, layer_idx: int,
               mode: str = MODE, sample_slice=None) -> np.ndarray:
    """
    Load activations for one domain, one component, one layer.

    Returns
    -------
    np.ndarray, shape (N, d), dtype float32
    """
    key = f"{component}/{mode}"
    with h5py.File(h5_path(domain), "r") as f:
        if sample_slice is None:
            data = f[key][:, layer_idx, :]
        else:
            data = f[key][sample_slice, layer_idx, :]
    return data.astype(np.float32)
