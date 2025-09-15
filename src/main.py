"""src/main.py
Entry-point that orchestrates smoke-test and full-scale runs via command-line
flags.  Usage::

    python -m src.main --smoke-test        # quick CI run
    python -m src.main --full-experiment   # heavier, but still synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import yaml

# Local imports (absolute to *src*)
from src.evaluate import run_experiment_1, run_experiment_2, run_experiment_3


_CONFIG_DIR = Path("config")
_SMOKE_YAML = _CONFIG_DIR / "smoke_test.yaml"
_FULL_YAML = _CONFIG_DIR / "full_experiment.yaml"


# -----------------------------------------------------------------------------
#  CLI helpers
# -----------------------------------------------------------------------------

def _parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HoloChain-Cert experiments")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke-test", action="store_true", help="Run short smoke test")
    group.add_argument("--full-experiment", action="store_true", help="Run full synthetic experiment")
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
#  Config loader
# -----------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def _run(cfg: Dict):
    # For this refactor we always run the three experiments sequentially.
    run_experiment_1(cfg)
    run_experiment_2(cfg)
    run_experiment_3(cfg)


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])

    if args.smoke_test:
        cfg_path = _SMOKE_YAML
    elif args.full_experiment:
        cfg_path = _FULL_YAML
    else:
        print("[WARN] no flag provided – defaulting to smoke test")
        cfg_path = _SMOKE_YAML

    cfg = _load_yaml(cfg_path)
    print(f"=== Loaded config: {cfg_path} ===")
    _run(cfg)


if __name__ == "__main__":
    main()
