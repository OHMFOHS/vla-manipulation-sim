"""Automated benchmark: run configurations over a fixed seed set and write CSV / JSON."""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from robotics.agent import AgentOptions, ManipulationAgent
from robotics.common.config import PROJECT_ROOT, SimulationConfig, default_simulation_config
from robotics.evaluation.metrics import episode_row, summarize
from robotics.recovery.executor import EpisodeReport

logger = logging.getLogger(__name__)

# Evaluation seeds are disjoint from demo-collection seeds (which start at 0).
EVAL_SEED_START = 10_000
DEFAULT_INSTRUCTION = "Sort all objects into their matching bins."


def eval_seeds(n: int, start: int = EVAL_SEED_START) -> list[int]:
    return list(range(start, start + n))


@dataclass
class BenchmarkConfig:
    name: str
    perception: str
    controller: str = "scripted"
    checkpoint: str | None = None
    faults: bool = False
    slip_probability: float = 0.0
    grasp_offset_probability: float = 0.0
    max_recovery_attempts: int = 2
    description: str = ""
    extra: dict = field(default_factory=dict)


STANDARD_CONFIGS = {
    "A_oracle_scripted": BenchmarkConfig(
        "A_oracle_scripted", "oracle", description="Oracle perception + scripted controller"
    ),
    "B_vision_scripted": BenchmarkConfig(
        "B_vision_scripted", "vision", description="RGB-D vision + scripted controller"
    ),
    "C_vision_bc": BenchmarkConfig(
        "C_vision_bc",
        "vision",
        "bc",
        "outputs/checkpoints/bc_policy.pt",
        description="RGB-D vision + learned BC policy",
    ),
    "C2_vision_bc_norecovery": BenchmarkConfig(
        "C2_vision_bc_norecovery",
        "vision",
        "bc",
        "outputs/checkpoints/bc_policy.pt",
        max_recovery_attempts=0,
        description="Vision + learned BC policy, recovery OFF (raw policy)",
    ),
    "D_vision_scripted_faults": BenchmarkConfig(
        "D_vision_scripted_faults",
        "vision",
        faults=True,
        slip_probability=0.3,
        grasp_offset_probability=0.3,
        description="Vision + scripted, injected gripper slips / grasp offsets, recovery ON (2 attempts)",
    ),
    "E_vision_scripted_faults_norecovery": BenchmarkConfig(
        "E_vision_scripted_faults_norecovery",
        "vision",
        faults=True,
        slip_probability=0.3,
        grasp_offset_probability=0.3,
        max_recovery_attempts=0,
        description="Same faults as D, recovery OFF (ablation)",
    ),
}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _versions() -> dict:
    import mujoco
    import numpy
    import robosuite
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "robosuite": robosuite.__version__,
        "mujoco": mujoco.__version__,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }


def run_config(
    bc: BenchmarkConfig,
    seeds: list[int],
    instruction: str = DEFAULT_INSTRUCTION,
    base_config: SimulationConfig | None = None,
) -> list[EpisodeReport]:
    cfg = (base_config or default_simulation_config()).model_copy(deep=True)
    cfg.faults.enabled = bc.faults
    cfg.faults.slip_probability = bc.slip_probability
    cfg.faults.grasp_offset_probability = bc.grasp_offset_probability
    cfg.recovery.max_recovery_attempts = bc.max_recovery_attempts
    agent = ManipulationAgent(
        AgentOptions(perception=bc.perception, controller=bc.controller, checkpoint=bc.checkpoint),
        sim_config=cfg,
    )
    reports = []
    try:
        for i, seed in enumerate(seeds):
            t0 = time.perf_counter()
            r = agent.run(instruction, seed)
            reports.append(r)
            logger.warning(
                "[%s] episode %d/%d seed=%d success=%s placed=%d/%d recoveries=%d (%.1fs)",
                bc.name,
                i + 1,
                len(seeds),
                seed,
                r.ground_truth["all_in_target"],
                sum(r.ground_truth["per_object_in_target"].values()),
                len(r.ground_truth["per_object_in_target"]),
                r.recovery_attempts,
                time.perf_counter() - t0,
            )
    finally:
        agent.close()
    return reports


def run_benchmark(
    configs: list[BenchmarkConfig],
    seeds: list[int],
    out_dir: str | Path = "outputs/benchmark",
    instruction: str = DEFAULT_INSTRUCTION,
    save_episode_logs: bool = True,
) -> dict:
    out = Path(out_dir)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    base_cfg = default_simulation_config()
    summaries = {}
    all_rows = []
    for bc in configs:
        reports = run_config(bc, seeds, instruction, base_cfg)
        summaries[bc.name] = {**summarize(bc.name, reports), "description": bc.description}
        rows = [episode_row(bc.name, r) for r in reports]
        all_rows += rows
        if save_episode_logs:
            with open(out / f"episodes_{bc.name}.jsonl", "w") as f:
                for r in reports:
                    f.write(json.dumps(r.to_dict(), default=str) + "\n")
        _write_outputs(out, summaries, all_rows, configs, seeds, instruction, base_cfg)
    return summaries


def _write_outputs(out: Path, summaries, rows, configs, seeds, instruction, base_cfg) -> None:
    with open(out / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    keys = [
        "episodes",
        "task_success_rate",
        "robosuite_success_rate",
        "pick_success_rate",
        "placement_success_rate",
        "avg_episode_sim_time_s",
        "avg_episode_wall_time_s",
        "avg_recovery_attempts",
        "collision_rate",
        "failure_rate",
        "avg_manipulation_actions",
        "avg_control_steps",
        "belief_ground_truth_agreement",
    ]
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", *keys])
        for name, s in summaries.items():
            writer.writerow(
                [name, *[round(s[k], 4) if isinstance(s[k], float) else s[k] for k in keys]]
            )
    with open(out / "per_object.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config",
                "object",
                "pick_attempts",
                "pick_success_rate",
                "placement_success_rate",
                "first_attempt_success_rate",
            ]
        )
        for name, s in summaries.items():
            for obj, m in s["per_object"].items():
                writer.writerow(
                    [
                        name,
                        obj,
                        m["pick_attempts"],
                        m["pick_success_rate"],
                        m["placement_success_rate"],
                        m["first_attempt_success_rate"],
                    ]
                )
    meta = {
        "instruction": instruction,
        "seeds": seeds,
        "configs": [c.__dict__ for c in configs],
        "simulation_config": base_cfg.model_dump(),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out / "summary.json", "w") as f:
        json.dump({"meta": meta, "results": summaries}, f, indent=2, default=str)
