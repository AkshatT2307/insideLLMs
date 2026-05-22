"""
data_utils.py — Dataset loading, stratified sampling, tokenization.

Loads per-domain CSV files from the arXiv abstracts dataset,
optionally samples a fixed number per domain, tokenizes with
left-padding + right-truncation, and returns PyTorch DataLoaders.
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ── domain → filename mapping ────────────────────────────────────────────────
DOMAIN_FILES = {
    "cs": "cs.csv",
    "eess": "eess.csv",
    "math": "math.csv",
    "physics": "physics.csv",
    "q-bio": "q-bio.csv",
    "stat": "stat.csv",
}


# ── PyTorch Dataset ──────────────────────────────────────────────────────────
class AbstractDataset(Dataset):
    """Holds pre-tokenized abstracts for a single domain."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        indices: List[int],
    ):
        self.input_ids = input_ids          # (N, max_length)
        self.attention_mask = attention_mask  # (N, max_length)
        self.indices = indices               # original row indices in the CSV

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "index": self.indices[idx],
        }


# ── loading ──────────────────────────────────────────────────────────────────
def load_domain_data(
    data_dir: str,
    samples_per_domain: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Load each domain CSV, optionally sampling a fixed number of rows.

    Parameters
    ----------
    data_dir : str
        Directory containing the per-domain CSV files.
    samples_per_domain : int or None
        If set, randomly sample this many rows per domain (fewer if the
        domain has less data).  None → keep all rows.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from domain name to its (possibly sampled) DataFrame.
    """
    domain_data: Dict[str, pd.DataFrame] = {}

    for domain, filename in DOMAIN_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"[WARN] {filepath} not found — skipping domain '{domain}'")
            continue

        df = pd.read_csv(filepath)
        df = df.dropna(subset=["text"]).reset_index(drop=True)

        if samples_per_domain is not None and len(df) > samples_per_domain:
            df = df.sample(n=samples_per_domain, random_state=seed).reset_index(drop=True)

        domain_data[domain] = df
        print(f"  Domain '{domain}': {len(df):,} samples loaded")

    total = sum(len(v) for v in domain_data.values())
    print(f"  Total across all domains: {total:,}\n")
    return domain_data


# ── tokenization ─────────────────────────────────────────────────────────────
def tokenize_domain(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 512,
    batch_size_tokenize: int = 1024,
) -> AbstractDataset:
    """
    Tokenize a domain DataFrame with left-padding and right-truncation.

    Left-padding ensures the *last* token in every sequence is always a
    real (non-pad) token — critical for correct "last token" activation
    extraction in causal LMs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``text`` column.
    tokenizer : PreTrainedTokenizerBase
        Must already have ``padding_side='left'`` set.
    max_length : int
        Fixed sequence length (truncate longer, pad shorter).
    batch_size_tokenize : int
        Tokenize in chunks to avoid OOM on very large DataFrames.

    Returns
    -------
    AbstractDataset
    """
    texts = df["text"].tolist()

    # Tokenize in chunks to limit peak memory
    all_input_ids = []
    all_attention_mask = []
    for start in range(0, len(texts), batch_size_tokenize):
        chunk = texts[start : start + batch_size_tokenize]
        enc = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        all_input_ids.append(enc["input_ids"])
        all_attention_mask.append(enc["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    indices = list(range(len(df)))

    return AbstractDataset(input_ids, attention_mask, indices)


def make_dataloader(
    dataset: AbstractDataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader (no shuffling — order must match HDF5 indices)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
