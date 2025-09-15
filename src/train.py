"""src/train.py
Training utilities & HoloChain-Cert detector stub so that the evaluation
pipeline can run end-to-end during CI / smoke tests.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch


class HoloChainCertModel(torch.nn.Module):
    """A minimal detector model (mean-pooled embeddings → sigmoid).

    Parameters
    ----------
    cfg : Optional[dict]
        YAML configuration dict. Only the ``hidden_dim`` key is consumed.
    tokenizer : Any
        A tokenizer object – only its ``vocab_size`` attribute is accessed so
        that we can initialise the linear layer with a deterministic fan-in.
    """

    def __init__(self, cfg: Optional[dict], tokenizer):  # noqa: D401
        super().__init__()
        hidden_dim = cfg.get("hidden_dim", 64) if cfg else 64
        self.token_dim = hidden_dim
        self.classifier = torch.nn.Linear(hidden_dim, 1)

        # Kaiming-uniform gives non-degenerate outputs without training.
        torch.nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))
        if self.classifier.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.classifier.weight
            )
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.classifier.bias, -bound, bound)

    def forward(
        self,
        ids: torch.Tensor,
        emb_fn: Callable[[torch.Tensor], torch.Tensor],
    ):  # noqa: D401
        """Forward pass.

        ids : LongTensor of shape *(batch, seq_len)*
        emb_fn : Callable that maps *ids* → *(batch, seq_len, hidden_dim)*
        """
        # (batch, seq, hidden)
        embeds = emb_fn(ids)
        pooled = embeds.mean(dim=1)  # mean-pool tokens
        logits = self.classifier(pooled).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs


# -----------------------------------------------------------------------------
# Optional – toy trainer used only in full experiments
# -----------------------------------------------------------------------------

def train_detector(model: HoloChainCertModel, dataloader, epochs: int = 1):
    """A *very* small training loop so that ``--full-experiment`` works."""

    device = next(model.parameters()).device
    criterion = torch.nn.BCELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device).float()
            preds = model(ids, batch["emb_fn"])  # Callable stored in batch
            loss = criterion(preds, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(f"[train] epoch {epoch+1}/{epochs} – loss={loss.item():.4f}")
