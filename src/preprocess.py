"""src/preprocess.py
Synthetic data generator / loader used for smoke & full experiment pipelines.
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict


# -----------------------------------------------------------------------------
#  Reproducibility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
#  Sliding window + summary helpers (placeholders for external use)
# -----------------------------------------------------------------------------

def sliding_windows(tokens: List[str], window: int, stride: int):
    for i in range(0, max(1, len(tokens) - window + 1), stride):
        yield tokens[i : i + window]


def generate_summaries(texts: List[str]):
    return [t[:100] + " … (summary)" for t in texts]


# -----------------------------------------------------------------------------
#  Dataset loader – returns synthetic WaterBench-like splits
# -----------------------------------------------------------------------------

class DatasetLoader:
    """Builds a small synthetic dataset when the real corpus is unavailable."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.seed = cfg.get("seed", 0)
        set_seed(self.seed)

    # ----------------------------------------------------------------------
    #  Main public API
    # ----------------------------------------------------------------------

    def load_waterbench(self) -> DatasetDict:
        """Returns a *datasets* `DatasetDict` with train / val / test keys."""
        size = self.cfg.get("dataset_size", {}).get("waterbench", 600)
        return _build_synthetic_dataset(size)


# -----------------------------------------------------------------------------
#  Helper – synthetic corpus construction
# -----------------------------------------------------------------------------


def _build_synthetic_dataset(total_size: int = 600) -> DatasetDict:
    """Creates a simple corpus with 50 % marked samples and trivial text."""
    texts = [f"Sample text number {i}" for i in range(total_size)]
    labels = [i % 2 for i in range(total_size)]  # alternate 0/1

    # Shuffle once for randomness
    idx = list(range(total_size))
    random.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]

    # 80 / 10 / 10 split
    n_train = int(0.8 * total_size)
    n_val = int(0.1 * total_size)

    splits = {
        "train": (texts[:n_train], labels[:n_train]),
        "val": (texts[n_train : n_train + n_val], labels[n_train : n_train + n_val]),
        "test": (texts[n_train + n_val :], labels[n_train + n_val :]),
    }

    ds_dict = {}
    for split, (txts, labs) in splits.items():
        ds_dict[split] = Dataset.from_dict({"text": txts, "is_marked": labs})
    return DatasetDict(ds_dict)
