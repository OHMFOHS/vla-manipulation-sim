"""Benchmark runner, demo collection, BC training and BC deployment (small but real)."""

from __future__ import annotations

import json

import numpy as np

from robotics.common.config import TrainingConfig
from robotics.evaluation.runner import BenchmarkConfig, run_benchmark
from robotics.learning.collect import collect_demos
from robotics.learning.train import train_bc


def test_benchmark_writes_results(tmp_path):
    cfg = BenchmarkConfig("smoke_oracle", "oracle")
    summaries = run_benchmark(
        [cfg], seeds=[10_010], out_dir=tmp_path, instruction="Put the milk into its bin."
    )
    s = summaries["smoke_oracle"]
    assert s["episodes"] == 1 and 0.0 <= s["task_success_rate"] <= 1.0
    for f in (
        "summary.json",
        "summary.csv",
        "episodes.csv",
        "per_object.csv",
        "episodes_smoke_oracle.jsonl",
    ):
        assert (tmp_path / f).exists(), f
    meta = json.loads((tmp_path / "summary.json").read_text())["meta"]
    assert meta["seeds"] == [10_010] and "robosuite" in meta["versions"]


def test_collect_train_and_deploy_bc(tmp_path):
    demo = tmp_path / "demos.npz"
    stats = collect_demos(
        2, demo, seed_start=0, perception="oracle", instruction="Put the milk into its bin."
    )
    assert stats["trajectories_kept"] >= 1
    d = np.load(demo)
    for key in ("observations", "actions", "timestamps", "task_ids"):
        assert key in d.files
    assert np.all(np.abs(d["actions"][:, :3]) <= 1.0)
    assert set(np.unique(d["actions"][:, 3])) <= {-1.0, 1.0}
    assert np.all(np.diff(d["timestamps"][d["task_ids"] == d["task_ids"][0]]) > 0)

    cfg = TrainingConfig(
        demo_path=str(demo), epochs=2, hidden_sizes=[32, 32], val_fraction=0.5, batch_size=256
    )
    summary = train_bc(cfg, out_dir=tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "bc_policy.pt").exists()
    assert summary["train_episodes"] == 1 and summary["val_episodes"] == 1  # episode-level split

    # Deploy the (barely trained) policy: it must run in closed loop and terminate cleanly.
    from robotics.agent import AgentOptions, ManipulationAgent

    agent = ManipulationAgent(
        AgentOptions(
            perception="oracle", controller="bc", checkpoint=str(tmp_path / "ckpt" / "bc_policy.pt")
        )
    )
    agent.cfg.recovery.max_recovery_attempts = 0
    agent.cfg.recovery.episode_step_budget = 900
    report = agent.run("Put the milk into its bin.", seed=10_020)
    assert report.controller == "bc"
    assert report.outcomes[0].attempts >= 1
    agent.close()
