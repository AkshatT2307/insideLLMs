"""
loader.py — Config-driven HDF5 activation loading for probing experiments.

Replaces the old KAUSTUBH/loader.py which hardcoded paths from config.py.
Now reads paths from config.yaml and model_slug.

Usage:
    from loader import ActivationLoader
    al = ActivationLoader(results_dir="./results/qwen2.5-7b")
    X = al.load_layer("cs", "attn", 14)
"""
import os
import h5py
import numpy as np


class ActivationLoader:
    """HDF5 activation loader driven by a results directory path."""

    def __init__(self, results_dir: str, pooling: str = "last"):
        self.activations_dir = os.path.join(results_dir, "activations")
        self.pooling = pooling

    def h5_path(self, domain: str) -> str:
        """Return the full path to a domain's HDF5 activation file."""
        return os.path.join(self.activations_dir, f"{domain}_activations.h5")

    def get_num_samples(self, domain: str) -> int:
        """Return the number of samples in a domain's file."""
        with h5py.File(self.h5_path(domain), "r") as f:
            return int(f.attrs["num_samples"])

    def load_component(self, domain: str, component: str,
                       mode: str = None, sample_slice=None) -> np.ndarray:
        """
        Load activations for one domain, one component.

        Returns
        -------
        np.ndarray, shape (N, L, d), dtype float32
        """
        mode = mode or self.pooling
        key = f"{component}/{mode}"
        with h5py.File(self.h5_path(domain), "r") as f:
            if sample_slice is None:
                data = f[key][:]
            else:
                data = f[key][sample_slice]
        return data.astype(np.float32)

    def load_layer(self, domain: str, component: str, layer_idx: int,
                   mode: str = None, sample_slice=None) -> np.ndarray:
        """
        Load activations for one domain, one component, one layer.

        Returns
        -------
        np.ndarray, shape (N, d), dtype float32
        """
        mode = mode or self.pooling
        key = f"{component}/{mode}"
        with h5py.File(self.h5_path(domain), "r") as f:
            if sample_slice is None:
                data = f[key][:, layer_idx, :]
            else:
                data = f[key][sample_slice, layer_idx, :]
        return data.astype(np.float32)
