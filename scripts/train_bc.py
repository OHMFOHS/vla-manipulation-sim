"""Train the MLP behaviour-cloning policy.

uv run python scripts/train_bc.py
uv run python scripts/train_bc.py --epochs 100 --demos outputs/data/demos.npz
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--demos", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--max-episodes", type=int, default=None, help="train on the first N demo episodes only"
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()

    from robotics.common.logging_utils import setup_logging

    setup_logging(args.log_level)
    from robotics.common.config import TrainingConfig, load_config
    from robotics.learning.train import train_bc

    cfg = load_config(TrainingConfig, args.config)
    if args.demos:
        cfg.demo_path = args.demos
    if args.epochs:
        cfg.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    s = train_bc(cfg, args.out_dir, args.max_episodes)
    best = min(s["history"], key=lambda h: h["val_loss"])
    print(
        f"trained on {s['train_samples']} samples ({s['train_episodes']} episodes), "
        f"validated on {s['val_samples']} ({s['val_episodes']} episodes) in {s['train_time_s']:.0f}s\n"
        f"best epoch {best['epoch']}: val loss {best['val_loss']:.4f}, "
        f"xyz MSE {best['val_xyz_mse']:.4f}, gripper accuracy {best['val_grip_acc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
