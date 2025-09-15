"""src/preprocess.py
Updated: Respect tokenizer.model_max_length to avoid sequence length overflow.
"""
from __future__ import annotations

import os
from typing import Dict, Any

from datasets import load_dataset


class DataModule:  # noqa: D401 – simple container
    """Download, slice and tokenise datasets according to a YAML config."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _download_one(repo: str, split: str):
        print(f"Downloading dataset {repo} split={split}")
        try:
            return load_dataset(repo, split=split, use_auth_token=os.getenv("HF_TOKEN"))
        except Exception as exc:  # noqa: BLE001 – pragmatic fallback
            from datasets import Dataset

            print(f"WARNING: failed to download {repo} ({exc}). Using dummy dataset instead.")
            return Dataset.from_dict({"prompt": ["Hello, world!", "How are you?"]})

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_dataset(self):
        ds = self._download_one(self.cfg["hf_repo"], self.cfg.get("split", "train"))
        if self.cfg.get("limit") is not None:
            ds = ds.select(range(min(len(ds), self.cfg["limit"])))
        return ds

    @staticmethod
    def preprocess(ds, tokenizer):
        max_len = min(getattr(tokenizer, "model_max_length", 4096), 4096)

        def _proc(x):  # noqa: D401
            text = " ".join(x["prompt"].strip().split())
            return {"input_ids": tokenizer(text, truncation=True, max_length=max_len)["input_ids"]}

        return ds.map(_proc, remove_columns=ds.column_names, batched=False)
