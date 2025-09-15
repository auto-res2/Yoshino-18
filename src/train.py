"""src/train.py
Training utilities & HoloChain-Cert detector stub so that the evaluation
pipeline can run end-to-end during CI / smoke tests.  This is **NOT** a
faithful implementation of the research model – it is a lightweight
placeholder that fulfils the public interface required by `src/evaluate.py`.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch


class HoloChainCertModel(torch.nn.Module):
    """A minimal detector model.

    The model consumes token-ids plus an *embedding function* provided by the
    caller (cf. `emb_fn` in `evaluate.run_experiment_1`).  It mean-pools the
    embeddings, applies a linear projection followed by a sigmoid to obtain a
    detection probability.

    Parameters
    ----------
    cfg : Dict | None
        A configuration dict parsed from YAML. Only `hidden_dim` may be
        consulted – everything else is ignored in the stub implementation.
    tokenizer : Any
        Tokeniser object – only its `vocab_size` attribute is accessed so that
        we can initialise the final layer with a deterministic fan-in.
    """

    def __init__(self, cfg: Optional[dict], tokenizer):  # noqa: D401
        super().__init__()
        hidden_dim = cfg.get("hidden_dim", 64) if cfg else 64
        self.token_dim = hidden_dim
        # Simple linear classifier: (embedding_dim) → 1
        self.classifier = torch.nn.Linear(hidden_dim, 1)

        # Kaiming-uniform init ensures non-degenerate outputs even without
        # training.
        torch.nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))
        if self.classifier.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.classifier.weight
            )
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.classifier.bias, -bound, bound)

    def forward(self, ids: torch.Tensor, emb_fn: Callable[[torch.Tensor], torch.Tensor]):
        """Forward pass.

        Parameters
        ----------
        ids : torch.Tensor
            Shape *(batch, seq_len)* token IDs.
        emb_fn : Callable
            A function that maps the IDs to embeddings with shape
            *(batch, seq_len, hidden_dim)*.
        """
        # (batch, seq, hidden)
        embeds = emb_fn(ids)
        # Mean-pool over tokens – ignores padding for simplicity.
        pooled = embeds.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs


# -----------------------------------------------------------------------------
# Optional – a toy trainer used only in full experiments (not smoke tests)
# -----------------------------------------------------------------------------

def train_detector(model: HoloChainCertModel, dataloader, epochs: int = 1):
    """Very small training loop so that `src/main.py --full-experiment` can run.

    This **does not learn anything meaningful** – it is merely provided to show
    how one *would* train the above stub model.
    """

    device = next(model.parameters()).device
    criterion = torch.nn.BCELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device).float()
            preds = model(ids, batch["emb_fn"])  # batch['emb_fn'] is a Callable
            loss = criterion(preds, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(f"[train] epoch {epoch+1}/{epochs} – loss={loss.item():.4f}")
