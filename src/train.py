"""src/train.py
Model wrappers (DADS, ExDAR) and the Budgeted Meta-Sampler search agent that are
used by the experiments.  No training logic is re-implemented – the project
focuses on evaluation so the wrappers are extremely lightweight.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch

# -----------------------------------------------------------------------------
#   DADSWrap – Decoding-Aware Diffused Smoothing
# -----------------------------------------------------------------------------

class DADSWrap(torch.nn.Module):
    """Very light-weight wrapper that performs Decoding-Aware Diffused Smoothing.
    A proper PAC-Bayes implementation is far more complex; here we only add
    Gaussian noise *sigma* for *steps* iterations and pass it through a frozen
    linear probe so downstream code can inspect the diffused logits.
    """

    def __init__(self, base_model: torch.nn.Module, sigma: float = 0.12, steps: int = 4):
        super().__init__()
        self.model = base_model
        self.sigma = sigma
        self.steps = steps

        # Identity linear probe that can (optionally) be fine-tuned later.
        self.recover_layer = torch.nn.Linear(
            base_model.config.vocab_size, base_model.config.vocab_size, bias=False
        )
        torch.nn.init.eye_(self.recover_layer.weight)
        self.recover_layer.requires_grad_(False)

    # ------------------------------------------------------------------
    # The standard LM forward is *not* touched, we only override generate.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):  # noqa: D401,E501 – signature mirrors base model
        return self.model.forward(*args, **kwargs)

    @torch.no_grad()
    def generate(self, **kwargs):  # noqa: D401 – keep HF signature intact
        outputs = self.model.generate(
            **kwargs, output_scores=True, return_dict_in_generate=True
        )
        logits_per_step = outputs.scores  # list[Tensor] (seq_len, B, V)
        diffused_scores = []
        for step_logits in logits_per_step:
            noisy = step_logits
            for _ in range(self.steps):
                noisy = noisy + torch.randn_like(noisy) * self.sigma
            recovered = self.recover_layer(noisy)
            diffused_scores.append(recovered)
        outputs.scores = diffused_scores
        return outputs

# -----------------------------------------------------------------------------
#   ExDAR – Expert-Diversified Adaptive Routing
# -----------------------------------------------------------------------------

class ExDARWrapper(torch.nn.Module):
    """Ensemble *votes* expert routes via Gumbel-Softmax.  At inference the logits
    from the *votes* replicas are averaged.  The heavy lifting (routing,
    MoE-specific tricks) is omitted for brevity – this is a *behavioural stub*
    that keeps the public interface identical to a HF causal-LM.
    """

    def __init__(self, base_model: torch.nn.Module, votes: int = 3, alpha: float = 1.0):
        super().__init__()
        self.model = base_model
        self.votes = votes
        self.alpha = alpha  # kept for traceability only

    @torch.no_grad()
    def generate(self, tokenizer, prompt_ids: torch.Tensor, **gen_kwargs):
        batch = prompt_ids.shape[0]
        prompt_rep = prompt_ids.repeat(self.votes, 1)
        outputs = self.model.generate(
            input_ids=prompt_rep,
            **gen_kwargs,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # Aggregate logits across votes
        scores = torch.stack(list(outputs.scores), dim=0)  # (seq, B*v, V)
        seq_len, bigB, vocab = scores.shape
        scores = scores.view(seq_len, self.votes, batch, vocab)
        mean_scores = scores.mean(dim=1)
        tokens = mean_scores.argmax(-1).transpose(0, 1)  # (B, seq)
        return tokens

# -----------------------------------------------------------------------------
#   BuMS – Budgeted Meta-Sampler search (tiny RL stub via SB3 PPO)
# -----------------------------------------------------------------------------

from stable_baselines3 import PPO  # heavy import but only when BuMS is used
from stable_baselines3.common.vec_env import DummyVecEnv  # correct import path

class _BuMSEnv:  # noqa: D401 – Gym-style env, minimal interface
    """A toy continuous control problem representing the search space of the
    budgeted meta-sampler.  The physics are *not* meaningful; they are a compact
    stand-in so every code path of the paper runs during smoke tests.
    """

    def __init__(self, constraint_vram_gb: float, constraint_latency_ms: float):
        import gymnasium as gym  # gymnasium is the standard backend for SB3 >=2.0

        self._gym = gym
        self.action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self._constraint_vram = constraint_vram_gb
        self._constraint_lat = constraint_latency_ms
        # state: (ε,σ,r,T,p,k)
        self.state = np.array([0.18, 0.12, 2.0, 0.9, 0.88, 3.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API (reset, step)
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # noqa: D401 – Gym API
        if seed is not None:
            np.random.seed(seed)
        return self.state.copy(), {}

    def step(self, action):  # noqa: D401 – Gym API
        self.state = np.clip(
            self.state + action,
            [0.05, 0.05, 1.0, 0.6, 0.6, 1.0],
            [0.3, 0.25, 4.0, 1.3, 0.95, 5.0],
        ).astype(np.float32)
        eps, sig, r, T, p, k = self.state
        vram = 6 + 0.5 * k + 8 * eps  # toy cost model
        latency = 100 + 50 * T + 30 * k
        ruh = 1.0 - 0.5 * eps - 0.3 * sig + 0.05 * r
        penalty = 50 * max(0, vram - self._constraint_vram) + 20 * max(0, latency - self._constraint_lat)
        reward = ruh - penalty / 100.0
        terminated = False  # continuing task
        truncated = False
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
        return self.state.copy(), reward, terminated, truncated, info

# -----------------------------------------------------------------------------
#   Public BuMS interface
# -----------------------------------------------------------------------------

class BuMSSearch:
    """Tiny PPO loop that optimises the _BuMSEnv – again, just for plumbing."""

    def __init__(self, vram_gb: float, latency_ms: float, episodes: int):
        self.env = DummyVecEnv([lambda: _BuMSEnv(vram_gb, latency_ms)])
        # Use a very small policy to keep the smoke test light-weight
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            n_steps=64,
            batch_size=32,
            learning_rate=3e-4,
        )
        self.episodes = episodes

    def run(self) -> Tuple[Dict[str, Any], float]:
        # Train PPO for the requested number of episodes (each episode ~= n_steps)
        self.model.learn(total_timesteps=self.episodes * 64)
        obs = self.env.reset()
        best_cfg, best_score = None, -1e9
        # Rollout a handful of steps to find the best-scoring configuration
        for _ in range(128):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, _, _, info = self.env.step(action)
            if reward > best_score:
                best_score = reward
                best_cfg = info[0]["cfg"]
        return best_cfg, float(best_score)
