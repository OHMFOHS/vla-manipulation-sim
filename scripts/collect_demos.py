"""Collect scripted-expert demonstrations for behaviour cloning.

uv run python scripts/collect_demos.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument(
        "--seed-start", type=int, default=0, help="demo seeds are [seed_start, seed_start+episodes)"
    )
    p.add_argument("--perception", choices=["vision", "oracle"], default="vision")
    p.add_argument("--out", default=None, help="default: training.yaml demo_path")
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()

    from robotics.common.logging_utils import setup_logging

    setup_logging(args.log_level)
    from robotics.common.config import TrainingConfig, load_config
    from robotics.evaluation.runner import EVAL_SEED_START
    from robotics.learning.collect import collect_demos

    if args.seed_start + args.episodes > EVAL_SEED_START:
        print(
            f"Demo seeds must stay below the evaluation seeds ({EVAL_SEED_START}) to avoid leakage."
        )
        return 2
    tcfg = load_config(TrainingConfig, args.config)
    stats = collect_demos(
        args.episodes, args.out or tcfg.demo_path, args.seed_start, args.perception, tcfg
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
