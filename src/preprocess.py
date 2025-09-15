"""src/preprocess.py
Data loading & simple preprocessing utilities.  All dataset handling goes
through this *single* module so the rest of the code base stays clean.
"""
from __future__ import annotations

import os
from typing import Dict, Any

from datasets import load_dataset

LANG_TAG_FIELD = "lang"

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
        return load_dataset(repo, split=split, use_auth_token=os.getenv("HF_TOKEN"))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_dataset(self):
        main_ds = self._download_one(self.cfg["hf_repo"], self.cfg.get("split", "train"))
        if self.cfg.get("limit"):
            main_ds = main_ds.select(range(self.cfg["limit"]))
        return main_ds

    @staticmethod
    def preprocess(ds, tokenizer):
        def _proc(x):  # noqa: D401 – map helper
            text = x["prompt"].strip()
            text = " ".join(text.split())  # collapse whitespace
            return {"input_ids": tokenizer(text, truncation=True, max_length=4096)["input_ids"]}

        return ds.map(_proc, remove_columns=ds.column_names, batched=False)
