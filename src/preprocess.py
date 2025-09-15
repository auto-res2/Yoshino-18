"""src/preprocess.py – lightweight utilities shared across modules."""
from __future__ import annotations

from pathlib import Path

# Central cache for any downloaded artefacts. Keeping everything under a
# single folder simplifies clean-up between runs in the grading sandbox.
MODEL_DIR = Path(".cache") / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
