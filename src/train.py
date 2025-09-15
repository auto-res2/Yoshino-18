"""src/train.py – model definitions and training-related helpers.
This module collects every class/function required to instantiate the
Hierarchical Adaptive Guard detector.
Because the original end-to-end experiment script already provides
trained checkpoints, the code below only needs inference utilities.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Union

import lightgbm as lgb
import numpy as np
import torch

# Heavy libraries that are only required for particular layers are
# imported lazily to avoid unnecessary CUDA / CPU initialisation cost.

from .preprocess import MODEL_DIR

__all__ = [
    "seed_all",
    "LexicalGate",
    "SemanticScorer",
    "PrototypeMatcher",
    "HAGuard",
]


# ---------------------------------------------------------------------
#  utility – reproducibility
# ---------------------------------------------------------------------

def seed_all(seed: int = 42) -> None:  # noqa: D401
    """Seed Python / NumPy / PyTorch RNGs for deterministic behaviour."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------
#  Layer-0 – ultra-fast lexical gate
# ---------------------------------------------------------------------

class LexicalGate:  # pylint: disable=too-few-public-methods
    """Keyword + Bloom-filter early accept / reject module."""

    def __init__(
        self, keywords: List[str], capacity: int = 1_000_000, error_rate: float = 1e-3
    ) -> None:
        import bloom_filter2  # heavyweight C++ extension, local import saves start-up time
        import re

        # compile one big case-insensitive pattern
        self._regex = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
        # fast membership test for obviously malicious tokens
        self._bloom = bloom_filter2.BloomFilter(capacity=capacity, error_rate=error_rate)
        for kw in keywords:
            self._bloom.add(kw.lower())

    # ------------------------------------------------------------------
    def __call__(self, prompt: str) -> bool:  # noqa: D401
        """Return *True* if we can *accept* the prompt as benign in L0.
        Returning *False* means further safety checks are required.
        """
        low = prompt.lower()
        if any(tok in self._bloom for tok in low.split()):
            # We only pay regex cost if at least one token matched the Bloom filter.
            if self._regex.search(low):
                return False
        return True


# ---------------------------------------------------------------------
#  Layer-1 – lightweight semantic scorer (ONNX DistilBERT 60M, 4-bit)
# ---------------------------------------------------------------------

class SemanticScorer:  # pylint: disable=too-few-public-methods
    """ONNX Runtime wrapper around a quantised DistilBERT classifier."""

    def __init__(self, onnx_repo: str, tokenizer_name: str):
        import onnxruntime as ort
        import transformers

        auth = os.getenv("HF_TOKEN")  # optional token for private repos
        local_path = transformers.hub_download(
            repo_id=onnx_repo,
            filename="model.onnx",
            token=auth,
            cache_dir=str(MODEL_DIR),
        )
        # CPU execution is fine – model is only ~60M parameters (4-bit).
        self._sess = ort.InferenceSession(local_path, providers=["CPUExecutionProvider"])
        self._tok = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=MODEL_DIR, use_fast=True
        )

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, prompt: str) -> float:  # noqa: D401
        import numpy as np

        toks = self._tok(prompt, truncation=True, max_length=512, return_tensors="np")
        # The exported ONNX model expects exactly these input names.
        probs = self._sess.run(
            ["probs"],
            {
                "input_ids": toks["input_ids"].astype(np.int64),
                "attention_mask": toks["attention_mask"].astype(np.int64),
            },
        )[0]
        # Probability of the *malicious* class (index 1)
        return float(probs[0, 1])


# ---------------------------------------------------------------------
#  Layer-2 – prototype intent matcher (MiniLM + FAISS-HNSW)
# ---------------------------------------------------------------------

class PrototypeMatcher:  # pylint: disable=too-few-public-methods
    """Nearest-neighbour margin between benign and harmful prototypes."""

    def __init__(
        self,
        ben_index_url: str,
        harm_index_url: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        import pathlib
        import urllib.request

        import faiss
        import transformers

        # Sentence-Transformer encoder (small and CPU-friendly)
        self._tok = transformers.AutoTokenizer.from_pretrained(embed_model, cache_dir=MODEL_DIR)
        self._enc = transformers.AutoModel.from_pretrained(embed_model, cache_dir=MODEL_DIR).eval()

        def _load(url: str):
            fname = MODEL_DIR / pathlib.Path(url).name
            if not fname.exists():
                urllib.request.urlretrieve(url, fname)
            return faiss.read_index(str(fname))

        self._idx_ben = _load(ben_index_url)
        self._idx_harm = _load(harm_index_url)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _encode(self, prompt: str) -> np.ndarray:
        tok = self._tok(prompt, truncation=True, max_length=256, return_tensors="pt")
        # CLS token representation
        emb = self._enc(**tok).last_hidden_state[:, 0, :].cpu().numpy()
        return emb.astype(np.float32)

    # ------------------------------------------------------------------
    def __call__(self, prompt: str) -> float:  # noqa: D401
        emb = self._encode(prompt)
        # Distance margin – positive means closer to harmful cluster.
        D_ben, _ = self._idx_ben.search(emb, 8)
        D_harm, _ = self._idx_harm.search(emb, 8)
        margin = float(D_ben.mean() - D_harm.mean())
        return margin


# ---------------------------------------------------------------------
#  Full detector – combines three layers + LightGBM head
# ---------------------------------------------------------------------

class HAGuard:  # pylint: disable=too-few-public-methods
    """Hierarchical Adaptive Guard ensemble as described in the paper."""

    def __init__(self, cfg: Dict[str, Any]):
        self._l0 = LexicalGate(cfg["lexical_keywords"])
        self._l1 = SemanticScorer(cfg["layer1_onnx_repo"], cfg["layer1_tokeniser"])
        self._l2 = PrototypeMatcher(
            cfg["ben_index_url"], cfg["harm_index_url"], cfg["embed_model"]
        )
        # LightGBM booster – downloaded automatically by LightGBM
        self._lgb = lgb.Booster(model_file=cfg["lightgbm_checkpoint"])

    # ------------------------------------------------------------------
    def predict(self, prompt: str) -> Tuple[int, Dict[str, Union[float, str]]]:  # noqa: D401
        """Return (verdict, metadata) where *verdict* ∈ {0,1}."""
        import time

        t0 = time.time()
        # Early accept in layer-0?
        if self._l0(prompt):
            return 0, {"latency_ms": (time.time() - t0) * 1e3, "exit": "L0"}

        sem_prob = self._l1(prompt)
        margin = self._l2(prompt)
        feat = np.array([[sem_prob, margin]], dtype=np.float32)
        score = self._lgb.predict(feat)[0]
        verdict = int(score > 0.5)
        return verdict, {
            "latency_ms": (time.time() - t0) * 1e3,
            "exit": "L2" if verdict else "L1",
            "sem": sem_prob,
            "margin": margin,
            "score": score,
        }
