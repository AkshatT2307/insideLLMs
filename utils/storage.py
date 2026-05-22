"""
storage.py — HDF5-based storage for extracted activations.

Creates one HDF5 file per domain with pre-allocated, chunked, gzip-
compressed datasets.  Supports incremental batch-by-batch writes so
that only one batch of pooled activations needs to live in memory at
a time.

File layout
-----------
{domain}_activations.h5
├── attrs: domain, model, num_layers, hidden_dim, max_length, num_samples
├── metadata/
│   └── indices        (N,)          int64
├── attn/
│   ├── last           (N, L, d)     float16
│   └── mean           (N, L, d)     float16
├── mlp/
│   ├── last           (N, L, d)     float16
│   └── mean           (N, L, d)     float16
└── layer/
    ├── last           (N, L, d)     float16
    └── mean           (N, L, d)     float16
"""

import os
from typing import Dict, Optional

import h5py
import numpy as np
import torch

COMPONENTS = ("attn", "mlp", "layer")
MODES = ("last", "mean")


def create_hdf5(
    filepath: str,
    num_samples: int,
    num_layers: int,
    hidden_dim: int,
    domain: str,
    model_name: str,
    max_length: int,
    chunk_batch: int = 64,
    compression_level: int = 4,
) -> h5py.File:
    """
    Create and return an open HDF5 file with pre-allocated datasets.

    Parameters
    ----------
    filepath : str
        Full path for the .h5 file.
    num_samples : int
        Number of samples that will be written (N).
    num_layers : int
        Number of transformer layers (L).
    hidden_dim : int
        Model hidden dimension (d).
    domain : str
        Domain name stored as an attribute.
    model_name : str
        HuggingFace model identifier stored as an attribute.
    max_length : int
        Tokenizer max sequence length stored as an attribute.
    chunk_batch : int
        Number of samples per HDF5 chunk (first axis).
    compression_level : int
        gzip compression level (1-9).

    Returns
    -------
    h5py.File
        An *open* file handle — caller is responsible for closing it.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    f = h5py.File(filepath, "w")

    # ── file-level attributes ────────────────────────────────────────────
    f.attrs["domain"] = domain
    f.attrs["model"] = model_name
    f.attrs["num_layers"] = num_layers
    f.attrs["hidden_dim"] = hidden_dim
    f.attrs["max_length"] = max_length
    f.attrs["num_samples"] = num_samples

    # ── metadata ─────────────────────────────────────────────────────────
    meta = f.create_group("metadata")
    meta.create_dataset("indices", shape=(num_samples,), dtype="int64")

    # ── activation datasets ──────────────────────────────────────────────
    cb = min(chunk_batch, num_samples)
    chunk_shape = (cb, num_layers, hidden_dim)

    for comp in COMPONENTS:
        grp = f.create_group(comp)
        for mode in MODES:
            grp.create_dataset(
                mode,
                shape=(num_samples, num_layers, hidden_dim),
                dtype="float16",
                chunks=chunk_shape,
                compression="gzip",
                compression_opts=compression_level,
            )

    return f


def write_batch(
    hf: h5py.File,
    results: Dict[str, torch.Tensor],
    indices: list,
    offset: int,
) -> int:
    """
    Write one batch of pooled activations into the HDF5 file.

    Parameters
    ----------
    hf : h5py.File
        Open file returned by :func:`create_hdf5`.
    results : dict[str, Tensor]
        Output of ``ActivationStore.get_batch_results()``.
        Keys like ``"attn/last"``; values of shape ``(B, L, d)`` float16.
    indices : list[int]
        Original dataset row indices for this batch.
    offset : int
        Current write cursor (number of samples already written).

    Returns
    -------
    int
        New offset (``offset + batch_size``).
    """
    batch_size = len(indices)
    end = offset + batch_size

    # metadata indices
    hf["metadata/indices"][offset:end] = np.array(indices, dtype=np.int64)

    # activation tensors
    for key, tensor in results.items():
        # key is e.g. "attn/last"
        hf[key][offset:end] = tensor.numpy()

    return end


def close_hdf5(hf: h5py.File) -> None:
    """Flush and close the HDF5 file."""
    hf.flush()
    hf.close()
