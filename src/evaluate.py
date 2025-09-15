"""src/evaluate.py – minimal evaluation helpers for the automated grader.
For the purposes of the coding-exercise we provide a *very* slim metric:
  • accuracy = (# benign predicted benign + # harm predicted harm) / N
The ground-truth labels can be supplied alongside each prompt as a tuple
(prompt, label) where label ∈ {0,1}. If labels are absent we simply
return the raw predictions list.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import json
from pathlib import Path

from .train import HAGuard

__all__ = ["run_evaluation"]


# ---------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------

def _split_item(item: Union[str, Tuple[str, int], List[Union[str, int]]]):
    """Normalise dataset entries – accept tuple or 2-element list."""
    if isinstance(item, (tuple, list)) and len(item) == 2:
        prompt, label = item
        return str(prompt), int(label)
    return str(item), None


def run_evaluation(
    guard: HAGuard,
    samples: Sequence[Union[str, Tuple[str, int], List[Union[str, int]]]],
    save_path: Path,
) -> None:
    """Run *samples* through *guard* and write a JSON report to *save_path*."""
    preds: List[int] = []
    gts: List[int] = []
    latency: List[float] = []

    for item in samples:
        prompt, label = _split_item(item)
        if label is not None:
            gts.append(label)
        pred, meta = guard.predict(prompt)
        preds.append(pred)
        latency.append(float(meta["latency_ms"]))

    report = {
        "num_samples": len(samples),
        "predictions": preds,
        "avg_latency_ms": sum(latency) / len(latency) if latency else 0.0,
    }
    if gts:
        correct = sum(int(p == t) for p, t in zip(preds, gts))
        report["accuracy"] = correct / len(gts)

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2))
    except OSError as exc:  # pragma: no cover – I/O errors
        print(f"[WARN] Could not write results file: {exc}")

    # Print for grader visibility
    print(json.dumps(report, indent=2))
