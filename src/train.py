"""src/train.py
Minimal training-side utilities and wrappers used across experiments.
No functional changes beyond directory refactor.
"""
from __future__ import annotations

import os  # kept for potential future use of env vars such as HF_TOKEN
from typing import Tuple, Dict, Any, ClassVar

import numpy as np
import torch
import gymnasium as gym  # lightweight; only used for action/obs spaces

# -----------------------------------------------------------------------------
#   DADSWrap – Decoding-Aware Diffused Smoothing (greatly simplified)
# -----------------------------------------------------------------------------


class DADSWrap(torch.nn.Module):
    """Very small diffusion-noise wrapper for smoke / CI tests.

    A *real* DADS implementation would be far more complex; this stub is just
    enough to keep the refactored project executable and self-contained during
    automated validation.
    """

    def __init__(self, base_model: torch.nn.Module, sigma: float = 0.12, steps: int = 4):
        super().__init__()
        self.model = base_model
        self.sigma = sigma
        self.steps = steps

        # identity "recovery" layer – lets us re-use HF logits w/o size changes
        self.recover_layer = torch.nn.Linear(
            base_model.config.vocab_size, base_model.config.vocab_size, bias=False
        )
        torch.nn.init.eye_(self.recover_layer.weight)
        self.recover_layer.requires_grad_(False)

    # ------------------------------------------------------------------
    #   Public API (mirror HuggingFace behaviour where possible)
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):  # noqa: D401
        return self.model.forward(*args, **kwargs)

    @torch.no_grad()
    def generate(self, **kwargs):  # noqa: D401 – mimic HF signature
        out = self.model.generate(
            **kwargs, output_scores=True, return_dict_in_generate=True
        )
        diffused_scores = []
        for step_logits in out.scores:
            noisy = step_logits
            for _ in range(self.steps):
                noisy = noisy + torch.randn_like(noisy) * self.sigma
            diffused_scores.append(self.recover_layer(noisy))
        out.scores = diffused_scores
        return out


# -----------------------------------------------------------------------------
#   ExDAR – Expert-Diversified Adaptive Routing (behaviour stub)
# -----------------------------------------------------------------------------


class ExDARWrapper(torch.nn.Module):
    """Stub that mimics routing diversity by averaging *votes* generations."""

    def __init__(self, base_model: torch.nn.Module, votes: int = 3, alpha: float = 1.0):
        super().__init__()
        self.model = base_model
        self.votes = max(1, votes)
        self.alpha = alpha  # kept for logging/traceability only

    @torch.no_grad()
    def generate(self, tokenizer, prompt_ids: torch.Tensor, **gen_kwargs):
        device = next(self.parameters()).device
        prompt_ids = prompt_ids.to(device)
        batch = prompt_ids.shape[0]
        prompt_rep = prompt_ids.repeat(self.votes, 1)
        outs = self.model.generate(
            input_ids=prompt_rep,
            **gen_kwargs,
            output_scores=True,
            return_dict_in_generate=True,
        )
        scores = torch.stack(list(outs.scores), dim=0)  # (seq, B*v, V)
        seq_len, big_B, vocab = scores.shape
        scores = scores.view(seq_len, self.votes, batch, vocab)
        mean_scores = scores.mean(dim=1)
        tokens = mean_scores.argmax(-1).transpose(0, 1)  # (B, seq)
        return tokens


# -----------------------------------------------------------------------------
#   BuMS – ultra-lightweight random search (no external RL deps)
# -----------------------------------------------------------------------------


class _BuMSEnv:
    """Toy environment capturing the (ε,σ,r,T,p,k) search space and constraints."""

    metadata: ClassVar[Dict[str, Any]] = {}

    def __init__(self, constraint_vram_gb: float, constraint_latency_ms: float):
        self.action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self._constraint_vram = constraint_vram_gb
        self._constraint_lat = constraint_latency_ms
        self.state = np.array([0.18, 0.12, 2.0, 0.9, 0.88, 3.0], dtype=np.float32)

    # ------------------------------ Env API ------------------------------
    def reset(self):
        return self.state.copy()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state = np.clip(
            self.state + action,
            [0.05, 0.05, 1.0, 0.6, 0.6, 1.0],
            [0.3, 0.25, 4.0, 1.3, 0.95, 5.0],
        )
        eps, sig, r, T, p, k = self.state
        vram = 6 + 0.5 * k + 8 * eps  # GB (toy proxy)
        latency = 100 + 50 * T + 30 * k  # ms/50-tok
        ruh = 1.0 - 0.5 * eps - 0.3 * sig + 0.05 * r
        penalty = 50 * max(0, vram - self._constraint_vram) + 20 * max(0, latency - self._constraint_lat)
        reward = ruh - penalty / 100.0
        info = {
            "vram": float(vram),
            "latency": float(latency),
            "ruh": float(ruh),
            "cfg": {
                "eps": float(eps),
                "sigma": float(sig),
                "r": float(r),
                "T": float(T),
                "p": float(p),
                "k": float(k),
            },
        }
        return self.state.copy(), float(reward), False, info


class BuMSSearch:
    """Very small random search used in smoke tests (no external RL deps)."""

    def __init__(self, vram_gb: float, latency_ms: float, episodes: int = 10):
        self.env = _BuMSEnv(vram_gb, latency_ms)
        self.episodes = max(1, episodes)

    def run(self) -> Tuple[Dict[str, Any], float]:
        best_score = -1e9
        best_cfg: Dict[str, Any] | None = None
        _ = self.env.reset()
        steps_total = self.episodes * 64
        for _ in range(steps_total):
            action = self.env.action_space.sample()
            _, reward, _done, info = self.env.step(action)
            if reward > best_score:
                best_score = reward
                best_cfg = info["cfg"]
        assert best_cfg is not None  # logical guarantee
        return best_cfg, best_score
