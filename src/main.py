"""src/main.py
Command-line entry-point.  Supports *either* smoke test or full experiment as
required:

    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from accelerate import Accelerator

from .evaluate import run_exp1_long_context, run_exp2_exdar, run_exp3_bums

# -----------------------------------------------------------------------------
#   CLI helpers
# -----------------------------------------------------------------------------

def _load_yaml(path: Path):
    with open(path, "r") as fp:
        return yaml.safe_load(fp)


def main():  # noqa: D401 – script entry-point
    parser = argparse.ArgumentParser(description="MINOTAUR experimental runner")
    parser.add_argument("--smoke-test", action="store_true", help="quick 3-minute sanity run")
    parser.add_argument("--full-experiment", action="store_true", help="complete, long run")
    args = parser.parse_args()

    if not (args.smoke_test ^ args.full_experiment):
        print("ERROR: you must pass exactly one of --smoke-test or --full-experiment", file=sys.stderr)
        sys.exit(1)

    cfg_file = Path("config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml")
    if not cfg_file.exists():
        print(f"Configuration file {cfg_file} not found.", file=sys.stderr)
        sys.exit(1)

    cfg = _load_yaml(cfg_file)
    accelerator = Accelerator()

    # Dispatch experiments in the order defined by the YAML
    exps_cfg = cfg["experiments"]
    if exps_cfg.get("exp1_long_context", {}).get("active", False):
        run_exp1_long_context(exps_cfg["exp1_long_context"], accelerator)
    if exps_cfg.get("exp2_exdar_stress", {}).get("active", False):
        run_exp2_exdar(exps_cfg["exp2_exdar_stress"], accelerator)
    if exps_cfg.get("exp3_edge_bums", {}).get("active", False):
        run_exp3_bums(exps_cfg["exp3_edge_bums"], accelerator)


if __name__ == "__main__":  # pragma: no cover – manual execution only
    main()
