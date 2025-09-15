"""src/main.py
Command-line entry-point.  Supports *either* smoke test only or full experiment
(which automatically performs a smoke test first).  Usage:

    uv run python -m src.main --smoke-test        # smoke test only
    uv run python -m src.main --full-experiment   # smoke + full experiment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from accelerate import Accelerator

from .evaluate import run_exp1_long_context, run_exp2_exdar, run_exp3_bums

# -----------------------------------------------------------------------------
#   Helpers
# -----------------------------------------------------------------------------

_CONFIG_DIR = Path("config")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _run_experiments(cfg: Dict[str, Any], accelerator: Accelerator):
    exps_cfg = cfg["experiments"]
    if exps_cfg.get("exp1_long_context", {}).get("active", False):
        run_exp1_long_context(exps_cfg["exp1_long_context"], accelerator)
    if exps_cfg.get("exp2_exdar_stress", {}).get("active", False):
        run_exp2_exdar(exps_cfg["exp2_exdar_stress"], accelerator)
    if exps_cfg.get("exp3_edge_bums", {}).get("active", False):
        run_exp3_bums(exps_cfg["exp3_edge_bums"], accelerator)


# -----------------------------------------------------------------------------
#   Main
# -----------------------------------------------------------------------------

def main():  # noqa: D401 – script entry-point
    parser = argparse.ArgumentParser(description="MINOTAUR experimental runner")
    parser.add_argument("--smoke-test", action="store_true", help="run smoke-test only and exit")
    parser.add_argument("--full-experiment", action="store_true", help="run smoke-test *then* full experiment")
    args = parser.parse_args()

    if not (args.smoke_test ^ args.full_experiment):
        print("ERROR: you must pass exactly one of --smoke-test or --full-experiment", file=sys.stderr)
        sys.exit(1)

    accelerator = Accelerator()

    # Always perform smoke test first – even when the user ultimately wants the full run
    smoke_cfg = _load_yaml(_CONFIG_DIR / "smoke_test.yaml")
    print("\n===== SMOKE TEST START =====")
    _run_experiments(smoke_cfg, accelerator)
    print("===== SMOKE TEST COMPLETE ✓ =====\n")

    # If the user asked only for smoke test, we are done
    if args.smoke_test:
        return

    # Otherwise proceed with full experiment
    full_cfg_path = _CONFIG_DIR / "full_experiment.yaml"
    if not full_cfg_path.exists():
        print(f"Configuration file {full_cfg_path} not found.", file=sys.stderr)
        sys.exit(1)

    full_cfg = _load_yaml(full_cfg_path)
    print("===== FULL EXPERIMENT START =====")
    _run_experiments(full_cfg, accelerator)
    print("===== FULL EXPERIMENT COMPLETE ✓ =====")


if __name__ == "__main__":  # pragma: no cover – manual execution only
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user – exiting.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 – surface all errors in CI
        print("Experiment aborted due to error:", exc, file=sys.stderr)
        sys.exit(1)
