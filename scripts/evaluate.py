"""Automated benchmark over a fixed seed set.

    uv run python scripts/evaluate.py --episodes 50
    uv run python scripts/evaluate.py --episodes 50 --configs A_oracle_scripted B_vision_scripted
    uv run python scripts/evaluate.py --episodes 50 --configs D_vision_scripted_faults E_vision_scripted_faults_norecovery

Configs: A_oracle_scripted, B_vision_scripted, C_vision_bc (needs a checkpoint),
D_vision_scripted_faults, E_vision_scripted_faults_norecovery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument(
        "--configs", nargs="+", default=["A_oracle_scripted", "B_vision_scripted", "C_vision_bc"]
    )
    p.add_argument("--seed-start", type=int, default=None)
    p.add_argument("--instruction", default=None)
    p.add_argument("--checkpoint", default=None, help="override the BC checkpoint for C_vision_bc")
    p.add_argument("--out-dir", default="outputs/benchmark")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()

    from robotics.common.logging_utils import setup_logging

    setup_logging(args.log_level)
    from robotics.evaluation.runner import (
        DEFAULT_INSTRUCTION,
        EVAL_SEED_START,
        STANDARD_CONFIGS,
        eval_seeds,
        run_benchmark,
    )

    configs = []
    for name in args.configs:
        if name not in STANDARD_CONFIGS:
            print(f"unknown config {name}; choose from {list(STANDARD_CONFIGS)}")
            return 2
        c = STANDARD_CONFIGS[name]
        if c.controller == "bc":
            if args.checkpoint:
                c.checkpoint = args.checkpoint
            if not (ROOT / c.checkpoint).exists() and not Path(c.checkpoint).exists():
                print(
                    f"skipping {name}: checkpoint {c.checkpoint} not found (run collect_demos.py and train_bc.py)"
                )
                continue
        configs.append(c)
    if not configs:
        return 2
    seeds = eval_seeds(
        args.episodes, args.seed_start if args.seed_start is not None else EVAL_SEED_START
    )
    summaries = run_benchmark(configs, seeds, args.out_dir, args.instruction or DEFAULT_INSTRUCTION)
    print(f"\nResults ({len(seeds)} episodes per config, seeds {seeds[0]}..{seeds[-1]}):")
    header = f"{'config':40s} {'success':>8s} {'pick':>6s} {'place':>6s} {'recov':>6s} {'coll':>6s} {'actions':>8s} {'sim_s':>6s}"
    print(header)
    for name, s in summaries.items():
        print(
            f"{name:40s} {s['task_success_rate']:8.2f} {s['pick_success_rate']:6.2f} {s['placement_success_rate']:6.2f} "
            f"{s['avg_recovery_attempts']:6.2f} {s['collision_rate']:6.2f} {s['avg_manipulation_actions']:8.2f} "
            f"{s['avg_episode_sim_time_s']:6.1f}"
        )
    print(
        f"\nwritten to {ROOT / args.out_dir}: summary.json, summary.csv, episodes.csv, per_object.csv"
    )
    print(json.dumps({k: v["failure_reasons"] for k, v in summaries.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
