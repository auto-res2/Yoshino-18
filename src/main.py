from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import yaml

from .train import HAGuard, seed_all
from .evaluate import run_evaluation

CONFIG_DIR = Path("config")
RESULT_DIR = Path(".research") / "iteration3"  # updated to mandatory path


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _load_yaml(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        sys.exit(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="HAGuard runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="run quick smoke test")
    group.add_argument(
        "--full-experiment", action="store_true", help="run full experiment configuration"
    )
    args = parser.parse_args()

    cfg_file = (
        CONFIG_DIR / "smoke_test.yaml" if args.smoke_test else CONFIG_DIR / "full_experiment.yaml"
    )
    cfg = _load_yaml(cfg_file)

    # ------------------------------------------------------------------
    #  Seeding for determinism
    # ------------------------------------------------------------------
    seed_all(42)

    # ------------------------------------------------------------------
    #  Instantiate Guard & run evaluation
    # ------------------------------------------------------------------
    guard = HAGuard(cfg)

    # The YAML keeps a tiny inline dataset for demonstration purposes.
    dataset = cfg.get("dataset", [])
    if not dataset:
        sys.exit("Config must include a non-empty 'dataset' key for this demo.")

    # ------------------------------------------------------------------
    #  Output bookkeeping – each run gets its own timestamped JSON.
    # ------------------------------------------------------------------
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tag = "smoke" if args.smoke_test else "full"
    out_file = RESULT_DIR / f"{tag}_results_{stamp}.json"

    run_evaluation(guard, dataset, out_file)


if __name__ == "__main__":
    main()
